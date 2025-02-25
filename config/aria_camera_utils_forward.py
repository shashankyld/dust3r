import os
import sys
import torch
from torch.optim import Adam
import numpy as np
import cv2
from tqdm import tqdm

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import camera_calibrations

camera_name = "camera-slam-right"
aria_fisheye_intrinsics_kb6 = {
    }

# The order of fish624 parameters is (fx = fy), cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3

def get_fisheye624_params(camera_name):
    """Get fisheye624 parameters from config for a specific camera."""
    if camera_name not in camera_calibrations:
        raise ValueError(f"Camera {camera_name} not found in calibrations")
        
    camera = camera_calibrations[camera_name]
    params = camera['projection_params']
    
    # The order of fish624 parameters is (fx = fy), cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3
    aria_fisheye_intrinsics = {
        'fx': params[0],
        'fy': params[0],  # fx = fy for Aria cameras
        'cy': params[1],
        'cx': params[2],
        'k0': params[3],
        'k1': params[4],
        'k2': params[5],
        'k3': params[6],
        'k4': params[7],
        'k5': params[8],
        'p0': params[9],
        'p1': params[10],
        's0': params[11],
        's1': params[12],
        's2': params[13],
        's3': params[14]
    }
    
    return aria_fisheye_intrinsics

def estimate_r_theta_phi(u, v, fx, fy, cx, cy, p0, p1, s0, s1, s2, s3, num_iterations=100):
    """Estimate r_theta and phi using optimization."""
    # Initial estimates assuming undistorted image
    u_norm = (u - cx) / fx
    v_norm = (v - cy) / fy
    
    # Initialize phi and r_theta
    phi_init = torch.atan2(v_norm, u_norm) # torch.atan2(y, x) = taninv(y/x)
    r_theta_init = torch.sqrt(u_norm**2 + v_norm**2)
    
    # Create parameters for optimization
    phi = torch.nn.Parameter(phi_init)
    r_theta = torch.nn.Parameter(r_theta_init)
    
    optimizer = Adam([phi, r_theta], lr=0.01)
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        # Compute predicted u, v using the current estimates
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        r_theta_sq = r_theta**2
        
        u_pred = fx * (r_theta * cos_phi + 
                      (2*p0*r_theta_sq*cos_phi**2 + p0*r_theta_sq + 2*p1*r_theta_sq*sin_phi*cos_phi) +
                      (s0*r_theta_sq + s1*r_theta_sq**2)) + cx
        
        v_pred = fy * (r_theta * sin_phi +
                      (2*p1*r_theta_sq*sin_phi**2 + p1*r_theta_sq + 2*p0*r_theta_sq*sin_phi*cos_phi) +
                      (s2*r_theta_sq + s3*r_theta_sq**2)) + cy
        
        # Compute loss
        loss = (u_pred - u)**2 + (v_pred - v)**2
        loss.backward()
        optimizer.step()
    
    return r_theta.detach(), phi.detach()

def estimate_theta(r_theta, k_params):
    """Estimate theta from r_theta using Newton's method."""
    theta = r_theta  # Initial guess
    
    for _ in range(50):  # Max iterations
        f_theta = theta
        df_theta = 1.0
        
        for i, k in enumerate(k_params):
            power = 2*i + 3
            f_theta += k * theta**power    
            df_theta += k * power * theta**(power-1)
        
        f_theta -= r_theta        
        if abs(f_theta) < 1e-6:
            break
            
        theta = theta - f_theta/df_theta
    
    return theta

def undistort_fisheye624(image, camera_name, destination_params, batch_size=1024, device="cuda"):
    """
    Undistort fisheye image to pinhole projection using GPU batching.
    
    Args:
        image: Input fisheye image
        camera_name: Name of the camera from config
        destination_params: Dictionary containing:
            - focal_length: [fx, fy]
            - principal_point: [cx, cy]
            - image_size: [width, height]
        batch_size: Number of pixels to process in parallel
        device: Device to use for computation ("cuda" or "cpu")
    """
    # Get fisheye parameters
    fisheye_params = get_fisheye624_params(camera_name)
    
    # Create tensors for input image coordinates
    height, width = image.shape[:2]
    dest_height, dest_width = destination_params['image_size']
    total_pixels = dest_height * dest_width


    print("Shape of the original image: ", image.shape)
    
    # Create output image
    undistorted = np.zeros((dest_height, dest_width, image.shape[2]), dtype=image.dtype)

    print("Shape of the undistorted image: ", undistorted.shape)
    
    
    # Convert parameters to torch tensors on GPU
    fx = torch.tensor(fisheye_params['fx'], dtype=torch.float32, device=device)
    fy = torch.tensor(fisheye_params['fy'], dtype=torch.float32, device=device)
    cx = torch.tensor(fisheye_params['cx'], dtype=torch.float32, device=device)
    cy = torch.tensor(fisheye_params['cy'], dtype=torch.float32, device=device)
    
    k_params = torch.tensor([fisheye_params[f'k{i}'] for i in range(6)], dtype=torch.float32, device=device)
    p0 = torch.tensor(fisheye_params['p0'], dtype=torch.float32, device=device)
    p1 = torch.tensor(fisheye_params['p1'], dtype=torch.float32, device=device)
    s0, s1, s2, s3 = [torch.tensor(fisheye_params[f's{i}'], dtype=torch.float32, device=device) for i in range(4)]
    
    # Destination camera parameters
    fx_dest, fy_dest = destination_params['focal_length']
    cx_dest, cy_dest = destination_params['principal_point']

    print("Source parameters: ", fx, fy, cx, cy, k_params, p0, p1, s0, s1, s2, s3)
    print("Destination parameters: ", fx_dest, fy_dest, cx_dest, cy_dest)

    # if True: 
    #     return undistorted
    
    # Adjust focal lengths to account for fisheye FOV
    # Fisheye cameras typically have ~180-degree FOV, while pinhole projection works best with narrower FOV
    fx_dest = fx_dest * 0.5  # Reduce focal length to increase FOV coverage
    fy_dest = fy_dest * 0.5
    
    # Scale principal point to match the adjusted focal length
    cx_dest = cx_dest * 1.0  # Keep center point
    cy_dest = cy_dest * 1.0

    # Create mesh grid with adjusted range and proper bounds
    y_coords, x_coords = torch.meshgrid(
        torch.arange(dest_height, device=device), 
        torch.arange(dest_width, device=device),
        indexing='ij'
    )
    
    # Keep coordinates in image bounds
    x_coords = torch.clamp(x_coords, 0, dest_width - 1)
    y_coords = torch.clamp(y_coords, 0, dest_height - 1)
    
    # Convert to normalized coordinates
    x_norm = (x_coords - cx_dest) / fx_dest
    y_norm = (y_coords - cy_dest) / fy_dest
    
    # Stack for batch processing
    pixels = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
    
    # Convert image to torch tensor
    image_tensor = torch.from_numpy(image).to(device)
    
    # Process in batches
    num_batches = (total_pixels + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Undistorting image"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_pixels)
            
            batch_pixels = pixels[start_idx:end_idx]
            
            # Adjust the ray projection calculation
            x_norm = (batch_pixels[:, 0] - cx_dest) / fx_dest
            y_norm = (batch_pixels[:, 1] - cy_dest) / fy_dest
            
            # Calculate radius with proper scaling
            r = torch.sqrt(x_norm**2 + y_norm**2)
            theta = torch.atan(r)  # This maps infinity to pi/2
            
            # Scale phi to maintain aspect ratio
            phi = torch.atan2(y_norm, x_norm)
            
            # Convert to fisheye coordinates using the forward model
            r_theta = theta.clone()
            for i, k in enumerate(k_params):
                r_theta = r_theta + k * theta**(2*i + 3)
            
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            r_theta_sq = r_theta**2
            
            # Calculate fisheye pixel coordinates
            u = fx * (r_theta * cos_phi + 
                     (2*p0*r_theta_sq*cos_phi**2 + p0*r_theta_sq + 2*p1*r_theta_sq*sin_phi*cos_phi) +
                     (s0*r_theta_sq + s1*r_theta_sq**2)) + cx
            
            v = fy * (r_theta * sin_phi +
                     (2*p1*r_theta_sq*sin_phi**2 + p1*r_theta_sq + 2*p0*r_theta_sq*sin_phi*cos_phi) +
                     (s2*r_theta_sq + s3*r_theta_sq**2)) + cy
            
            # Modify grid sample coordinates and add mask for valid pixels
            u_valid = (u >= 0) & (u < width)
            v_valid = (v >= 0) & (v < height)
            valid_pixels = u_valid & v_valid

            # print("u and v: ", u, v)

            # Normalize coordinates to [-1, 1] range with proper clamping
            u_normalized = torch.clamp(u / (width - 1), 0, 1) * 2 - 1
            v_normalized = torch.clamp(v / (height - 1), 0, 1) * 2 - 1
            
            # Create sampling grid
            grid = torch.stack([u_normalized, v_normalized], dim=1).view(1, -1, 1, 2)

            # print("Grid shape: ", grid.shape)
            
            # Sample from image using grid_sample with zero padding
            sampled = torch.nn.functional.grid_sample(
                image_tensor.permute(2, 0, 1).unsqueeze(0).float(), 
                grid,
                mode='bilinear',
                padding_mode='zeros',  # Changed from 'border' to 'zeros'
                align_corners=True
            )
            
            # Update output image with bounds checking
            y_idx = torch.clamp(batch_pixels[:, 1], 0, dest_height - 1).cpu().numpy().astype(int)
            x_idx = torch.clamp(batch_pixels[:, 0], 0, dest_width - 1).cpu().numpy().astype(int)
            valid_mask = valid_pixels.cpu().numpy()
            
            # Safety check for indices
            safe_mask = (y_idx < dest_height) & (x_idx < dest_width)
            valid_mask = valid_mask & safe_mask
            
            if valid_mask.any():
                sampled_values = sampled.squeeze().permute(1, 0).cpu().numpy()
                undistorted[y_idx[valid_mask], x_idx[valid_mask]] = sampled_values[valid_mask]
    
    return undistorted

if __name__ == "__main__":
    # Test the parameter extraction
    camera_name = "camera-slam-left"
    try:
        params = get_fisheye624_params(camera_name)
        print(f"Successfully loaded parameters for {camera_name}:")
        for key, value in params.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error loading parameters: {e}")

    """ 
    Math: 
    First we will get polar coordinates of point in real world associated with each pixel in the fisheye 624 image.
    Then we can estimate pixel value for that point in pinhole image. 

    1. Equations that define polar coordinates to fisheye u,v coordinates:
    u = fx * r_theta * cos(phi) 
        + fx * [2*p0*r_theta^2*cos^2(phi) + p0*r_theta^2 + 2*p1*r_theta^2*sin(phi)*cos(phi)]
        + fx * [s0*r_theta^2 + s1*r_theta^4]
        + cx
    v = fy * r_theta * sin(phi)
        + fy * [2*p1*r_theta^2*sin^2(phi) + p1*r_theta^2 + 2*p0*r_theta^2*sin(phi)*cos(phi)]
        + fy * [s2*r_theta^2 + s3*r_theta^4]
        + cy 

    2. Equations that define polar coordinates to pinhole u,v coordinates:
    u_pinhole = fx_pinhole * x + cx_pinhole 
    v_pinhole = fy_pinhole * y + cy_pinhole

    3. First estimate r_theta and phi from u,v coordinates of fisheye image using pytorch minimization of the two equations above.
    # Init values are 
    # theta is the angle between the ray from the camera center to the point and the positive z-axis.
    # phi is the angle between the ray from the image center to the pixel coordinate  and the positive x-axis in the image plane.

    # FOR INITIALIZATION ASSUME THE IMAGE IS A UNDISTORTED IMAGE, so phi is taninv[(v-cy/fy) * (fx/u-cx)], r_theta is sqrt((u-cx)^2/fx^2 + (v-cy)^2/fy^2)
    # Then we can use the equations above to estimate r_theta and phi using pytorch minimization.
    # Then also estimate theta from r_theta. 
    since, r_theta = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9 + k5*theta^11 

    # Then estimate u_pinhole, v_pinhole from theta, phi assuming the linear model of pinhole camera.

    
    """
    # Test the undistortion
    path = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/data/preprocessed_adt_data/Apartment_release_clean_seq131_M1292/images/R/image_R_0000.png"
    image = cv2.imread(path)
    
    # If the image is in lanscape - convert to portrait by rotating clockwise
    if image.shape[0] < image.shape[1]:
        print("image rotated")
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)



    destination_height = 640
    destination_width = 480

    # Adjust destination parameters to avoid border artifacts
    image_height, image_width = image.shape[:2]
    print("Height and width of the original image: ", image_height, image_width)
    scale_factor = 1  # Reduce output size to avoid edge artifacts
    source_params = get_fisheye624_params("camera-slam-left")
    print(f"Source parameters: {source_params}")

    f_scale_factor = 325/source_params["fx"]  # Adjust focal length to match the output size
    destination_fx, destination_fy = f_scale_factor * source_params['fx'], f_scale_factor * source_params['fy']


    # SETTING SAME FOCAL LENGTH AS THE SOURCE IMAGE FOR SIMPLE MATH
    destination_params = {
        'focal_length': [destination_fx, destination_fy],
        'principal_point': [destination_width // 2, destination_height // 2],
        'image_size': [
            int(destination_height * scale_factor),
            int(destination_width * scale_factor)
        ]
    }
    
    print(f"Destination parameters: {destination_params}")
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    undistorted = undistort_fisheye624(image, "camera-slam-left", destination_params, 
                                      batch_size=4096, device=device)

    
    save_path = f"/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/other_projects/dust3r/logs/test0000"
    if not os.path.exists(path):
        os.makedirs(path)
    name_tag = save_path.split("/")[-1].split(".")[0]
    cv2.imwrite(f"{save_path}/right_original_{name_tag}.png", image)

    # Convert principal points to integers for OpenCV
    source_cx = int(round(source_params['cx']))
    source_cy = int(round(source_params['cy']))
    dest_cx = int(round(destination_params['principal_point'][0]))
    dest_cy = int(round(destination_params['principal_point'][1]))
    
    # # Draw circles at principal points
    # cv2.circle(image, (source_cx, source_cy), 5, (0, 255, 0), -1)
    # cv2.circle(undistorted, (dest_cx, dest_cy), 5, (0, 255, 0), -1)
    
    # # Add some visual guides
    # # Draw horizontal and vertical lines through principal points
    # cv2.line(image, (0, source_cy), (image_width, source_cy), (0, 255, 0), 1)
    # cv2.line(image, (source_cx, 0), (source_cx, image_height), (0, 255, 0), 1)
    
    # cv2.line(undistorted, (0, dest_cy), (destination_width, dest_cy), (0, 255, 0), 1)
    # cv2.line(undistorted, (dest_cx, 0), (dest_cx, destination_height), (0, 255, 0), 1)
    
    # Rest of the display code
    cv2.imshow("Original", image)
    cv2.imshow("Undistorted", undistorted)
    
    # SAVE THE UNDISTORTED IMAGE _ USE NAME FROM THE IMAGE FILE - include logs/*/*
    
    # cv2.imwrite(f"/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/other_projects/dust3r/logs/test2/undistorted_{name_tag}.png", undistorted)
    print("saved image")
    # create path if not exists

    cv2.imwrite(f"{save_path}/right_undistorted_{name_tag}.png", undistorted)
    # Save Original Image as well 
    cv2.waitKey(0)