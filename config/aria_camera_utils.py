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

def estimate_r_theta_phi(u, v, fx, fy, cx, cy, p0, p1, s0, s1, s2, s3, num_iterations=100):
    """Estimate r_theta and phi using optimization."""
    # Initial estimates assuming undistorted image
    u_norm = (u - cx) / fx
    v_norm = (v - cy) / fy
    
    # Initialize phi and r_theta
    phi_init = torch.atan2(v_norm, u_norm)
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
    
    # Create output image
    undistorted = np.zeros((dest_height, dest_width, image.shape[2]), dtype=image.dtype)
    
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
    
    # Create mesh grid of destination coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(dest_height, device=device), 
        torch.arange(dest_width, device=device),
        indexing='ij'  # Add indexing parameter
    )
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
            
            # Get normalized coordinates in pinhole camera
            x_norm = (batch_pixels[:, 0] - cx_dest) / fx_dest
            y_norm = (batch_pixels[:, 1] - cy_dest) / fy_dest
            
            # Convert to polar coordinates - Fix tensor type error
            denominator = torch.ones_like(x_norm, device=device)  # Create tensor of ones with same shape
            theta = torch.atan2(torch.sqrt(x_norm**2 + y_norm**2), denominator)
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
            
            # Grid sample for efficient bilinear interpolation
            u_normalized = (u / (width - 1)) * 2 - 1
            v_normalized = (v / (height - 1)) * 2 - 1
            grid = torch.stack([u_normalized, v_normalized], dim=1).view(1, -1, 1, 2)
            
            # Sample from image using grid_sample
            sampled = torch.nn.functional.grid_sample(
                image_tensor.permute(2, 0, 1).unsqueeze(0).float(), 
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            
            # Update output image - Fix CUDA to numpy conversion
            y_idx = batch_pixels[:, 1].cpu().numpy().astype(int)
            x_idx = batch_pixels[:, 0].cpu().numpy().astype(int)
            undistorted[y_idx, x_idx] = sampled.squeeze().permute(1, 0).cpu().numpy()
    
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
    image = cv2.imread("/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/other_projects/dust3r/config/distorted_example_kb6.png")
    destination_height = 880
    destination_width = 840

    destination_params = {
        'focal_length': [140, 140],
        'principal_point': [destination_width // 2, destination_height // 2],
        'image_size': [destination_width, destination_height]
    }
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    undistorted = undistort_fisheye624(image, "camera-slam-left", destination_params, 
                                      batch_size=4096, device=device)


    cv2.imshow("Original", image)
    cv2.imshow("Undistorted", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


