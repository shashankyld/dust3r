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

camera_name = "camera-slam-left"
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
        'cx': params[1],
        'cy': params[2],
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

def estimate_r_theta_phi(u_src, v_src, fx, fy, cx, cy, p0, p1, s0, s1, s2, s3, device, num_iterations=50):
    """Estimate r_theta and phi using optimization."""
    with torch.enable_grad():
        # Convert all inputs to tensors on device and enable gradients
        u = u_src.clone().detach().requires_grad_(True)
        v = v_src.clone().detach().requires_grad_(True)
        
        # Initialize parameters based on pinhole model
        u_norm = (u - cx) / fx
        v_norm = (v - cy) / fy
        
        # Initial estimates based on pinhole model
        phi = torch.atan2(v_norm, u_norm)
        r_theta = torch.sqrt(u_norm**2 + v_norm**2)
        
        # Convert to parameters
        phi = torch.nn.Parameter(phi)
        r_theta = torch.nn.Parameter(r_theta)
        
        # Setup optimizer
        optimizer = Adam([phi, r_theta], lr=0.01)
        
        # Target coordinates
        u_target = u.clone().detach()
        v_target = v.clone().detach()
        
        for iter_idx in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward computation
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            r_theta_sq = r_theta**2
            
            # Predict coordinates using fisheye model
            u_pred = fx * (r_theta * cos_phi + 
                       (2*p0*r_theta_sq*cos_phi**2 + p0*r_theta_sq + 2*p1*r_theta_sq*sin_phi*cos_phi) +
                       (s0*r_theta_sq + s1*r_theta_sq**2)) + cx
            
            v_pred = fy * (r_theta * sin_phi +
                       (2*p1*r_theta_sq*sin_phi**2 + p1*r_theta_sq + 2*p0*r_theta_sq*sin_phi*cos_phi) +
                       (s2*r_theta_sq + s3*r_theta_sq**2)) + cy
            
            # Compute loss
            loss = ((u_pred - u_target)**2 + (v_pred - v_target)**2).mean()
            
            if iter_idx % 10 == 0:
                print(f"Iteration {iter_idx}, Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        return r_theta.detach(), phi.detach()

def estimate_theta(r_theta, k_params, device, num_iterations=50):
    """Estimate theta from r_theta using PyTorch optimization."""
    with torch.enable_grad():
        # Initialize theta as a parameter
        theta = torch.nn.Parameter(r_theta.clone().detach())
        optimizer = Adam([theta], lr=0.01)
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward computation: r_theta = theta + k1*theta^3 + k2*theta^5 + ...
            r_theta_pred = theta.clone()
            for i, k in enumerate(k_params):
                power = 2*i + 3
                r_theta_pred = r_theta_pred + k * theta**power
            
            # Loss is the difference between predicted and target r_theta
            loss = ((r_theta_pred - r_theta)**2).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
    
    return theta.detach()

def undistort_fisheye624(image, camera_name, destination_params, batch_size=1024, device="cuda"):
    """Undistort fisheye image using optimization to find the mapping."""
    # Get fisheye parameters
    fisheye_params = get_fisheye624_params(camera_name)
    
    # Create tensors for input image coordinates
    src_height, src_width = image.shape[:2]
    dest_height, dest_width = destination_params['image_size']
    
    # Create output image (initialized to black)
    undistorted = np.zeros((dest_height, dest_width, image.shape[2]), dtype=image.dtype)
    
    # Get camera parameters
    fx = torch.tensor(fisheye_params['fx'], dtype=torch.float32, device=device)
    fy = torch.tensor(fisheye_params['fy'], dtype=torch.float32, device=device)
    cx = torch.tensor(fisheye_params['cx'], dtype=torch.float32, device=device)
    cy = torch.tensor(fisheye_params['cy'], dtype=torch.float32, device=device)
    
    # Get distortion parameters
    k_params = torch.tensor([fisheye_params[f'k{i}'] for i in range(6)], dtype=torch.float32, device=device)
    p0 = torch.tensor(fisheye_params['p0'], dtype=torch.float32, device=device)
    p1 = torch.tensor(fisheye_params['p1'], dtype=torch.float32, device=device)
    s0, s1, s2, s3 = [torch.tensor(fisheye_params[f's{i}'], dtype=torch.float32, device=device) for i in range(4)]
    
    # Destination camera parameters
    fx_dest, fy_dest = destination_params['focal_length']
    cx_dest, cy_dest = destination_params['principal_point']

    # Debug prints
    print("Source image shape:", image.shape)
    print("Destination image shape:", (dest_height, dest_width))
    print("Camera parameters:", fisheye_params)
    print("Destination parameters:", destination_params)
    
    # Create source pixel coordinates (we'll map FROM source TO destination)
    y_src, x_src = torch.meshgrid(
        torch.arange(src_height, device=device),
        torch.arange(src_width, device=device),
        indexing='ij'
    )
    source_pixels = torch.stack([x_src.flatten(), y_src.flatten()], dim=1)
    total_pixels = src_height * src_width
    
    # Process in batches
    num_batches = (total_pixels + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing source pixels"):
            print(f"\nProcessing batch {batch_idx}")
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_pixels)
            
            batch_pixels = source_pixels[start_idx:end_idx]
            
            # Convert coordinates to tensors and move to device
            u_src = batch_pixels[:, 0].float().to(device)
            v_src = batch_pixels[:, 1].float().to(device)
            
            try:
                print(f"Input shapes:")
                print(f"u_src: {u_src.shape}, v_src: {v_src.shape}")
                
                # Estimate r_theta and phi using optimization
                r_theta, phi = estimate_r_theta_phi(
                    u_src, v_src, fx, fy, cx, cy,
                    p0, p1, s0, s1, s2, s3,
                    device=device,
                    num_iterations=50
                )
                
                if r_theta is None or phi is None:
                    print(f"Skipping batch {batch_idx} due to optimization failure")
                    continue
                
                # Estimate theta from r_theta using optimization
                theta = estimate_theta(r_theta, k_params, device)
                
                # Convert to 3D coordinates
                x = torch.sin(theta) * torch.cos(phi)
                y = torch.sin(theta) * torch.sin(phi)
                z = torch.cos(theta)
                
                # Project to destination (pinhole) image plane
                z = torch.clamp(z, min=1e-6)  # Prevent division by zero
                x_norm = x / z
                y_norm = y / z
                
                u_dest = (fx_dest * x_norm + cx_dest).round().long()
                v_dest = (fy_dest * y_norm + cy_dest).round().long()
                
                # Check which destination coordinates are valid
                valid_mask = (u_dest >= 0) & (u_dest < dest_width) & \
                           (v_dest >= 0) & (v_dest < dest_height) & \
                           (z > 0)  # Only keep points in front of the camera
                
                # Get valid pixels
                u_src_valid = u_src[valid_mask].cpu().numpy().astype(int)
                v_src_valid = v_src[valid_mask].cpu().numpy().astype(int)
                u_dest_valid = u_dest[valid_mask].cpu().numpy()
                v_dest_valid = v_dest[valid_mask].cpu().numpy()
                
                # Debug first batch
                if batch_idx == 0:
                    print("First batch stats:")
                    print("Valid pixels:", len(valid_mask.nonzero()))
                    print("r_theta range:", r_theta.min().item(), r_theta.max().item())
                    print("theta range:", theta.min().item(), theta.max().item())
                    print("z range:", z.min().item(), z.max().item())
                    print("Source coords:", u_src_valid[:5], v_src_valid[:5])
                    print("Dest coords:", u_dest_valid[:5], v_dest_valid[:5])
                
                # Copy pixels from source to destination
                if len(valid_mask.nonzero()) > 0:
                    undistorted[v_dest_valid, u_dest_valid] = image[v_src_valid, u_src_valid]
                
            except Exception as e:
                print(f"Error in batch {batch_idx}:", str(e))
                continue
    
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
    path = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/other_projects/dust3r/logs/test2/rgb_image.png"
    image = cv2.imread(path)
    destination_height = 512 
    destination_width = 512

    # Adjust destination parameters to avoid border artifacts
    image_height, image_width = image.shape[:2]
    print("Height and width of the original image: ", image_height, image_width)
    scale_factor = 1  # Reduce output size to avoid edge artifacts
    
    destination_params = {
        'focal_length': [280, 280],  # Use source focal length for better results
        'principal_point': [destination_width // 2, destination_height // 2],
        'image_size': [destination_height, destination_width]
    }
    
    print(f"Destination parameters: {destination_params}")
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # camera-slam-left, camera-slam-right, camera-rgb
    
    undistorted = undistort_fisheye624(image, "camera-rgb", destination_params, 
                                      batch_size=16384, device=device)

    # ## FOR THE BLACK PIXEL, INTERPOLATE THE PIXEL VALUE FROM THE NEAREST PIXELS 
    # black_pixel = np.where(undistorted == 0)
    # for i in range(len(black_pixel[0])):
    #     u = black_pixel[1][i]
    #     v = black_pixel[0][i]
    #     if u == 0 or v == 0 or u == destination_width-1 or v == destination_height-1:
    #         continue
    #     undistorted[v, u] = (undistorted[v-1, u] + undistorted[v+1, u] + undistorted[v, u-1] + undistorted[v, u+1])/4




    cv2.imshow("Original", image)
    cv2.imshow("Undistorted", undistorted)

    # SAVE THE UNDISTORTED IMAGE _ USE NAME FROM THE IMAGE FILE
    name_tag = path.split("/")[-1].split(".")[0]
    cv2.imwrite(f"/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/other_projects/dust3r/logs/test2/backward_final_undistorted_{name_tag}.png", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


