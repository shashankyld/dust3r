import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
class AriaCameraModelConverterTorch:
    def __init__(self, intrinsics_aria_fisheye, f_pinhole, c_pinhole, device="cuda"):
        """Initialize with PyTorch tensors on specified device"""
        self.device = device
        self.aria_intrinsics = {k: torch.tensor(v, dtype=torch.float32, device=device)
                               for k, v in intrinsics_aria_fisheye.items()}
        self.f_pinhole = torch.tensor(f_pinhole, dtype=torch.float32, device=device)
        self.c_pinhole = torch.tensor(c_pinhole, dtype=torch.float32, device=device)
        
    def _r_theta_torch(self, theta, k_params):
        """Vectorized r_theta calculation using PyTorch"""
        r_theta = theta
        theta_power = theta
        for k in k_params:
            theta_power = theta_power * (theta**2)
            r_theta = r_theta + k * theta_power
        return r_theta

    def project_fisheye_radtan_thinprism_batch(self, xyz_batch):
        """Improved batch projection with proper coordinate handling"""
        # Get parameters
        fx = self.aria_intrinsics['fx']
        fy = self.aria_intrinsics['fy']
        cx = self.aria_intrinsics['cx']
        cy = self.aria_intrinsics['cy']
        
        k_params = torch.stack([self.aria_intrinsics[f'k{i}'] for i in range(6)])
        p = torch.stack([self.aria_intrinsics['p0'], self.aria_intrinsics['p1']])
        s = torch.stack([self.aria_intrinsics[f's{i}'] for i in range(4)])

        # Normalize 3D points
        norm_xy = torch.norm(xyz_batch[:, :2], dim=1, keepdim=True)
        norm_xyz = torch.norm(xyz_batch, dim=1, keepdim=True)
        
        # Calculate theta (angle from optical axis)
        theta = torch.acos(torch.clamp(xyz_batch[:, 2:3] / norm_xyz, -1.0, 1.0))
        
        # Calculate phi (azimuthal angle)
        phi = torch.atan2(xyz_batch[:, 1:2], xyz_batch[:, 0:1])

        # Apply fisheye distortion model
        r_theta = theta.clone()
        theta_sq = theta * theta

        for i, k in enumerate(k_params):
            r_theta = r_theta + k * theta_sq.pow(i + 1)

        # Project to image plane
        x = r_theta * torch.cos(phi)
        y = r_theta * torch.sin(phi)

        # Apply tangential distortion
        r_sq = x*x + y*y
        tdx = p[0] * (3*x*x + r_sq) + 2*p[1]*x*y
        tdy = p[1] * (3*y*y + r_sq) + 2*p[0]*x*y

        # Apply thin prism distortion
        tpx = s[0]*r_sq + s[1]*r_sq*r_sq
        tpy = s[2]*r_sq + s[3]*r_sq*r_sq

        # Final projection
        u = fx * (x + tdx + tpx) + cx
        v = fy * (y + tdy + tpy) + cy

        return torch.stack([u, v], dim=1)

    def undistort_image_to_pinhole(self, distorted_image, output_resolution, batch_size=4096):
        """Undistort using proper fisheye to pinhole mapping"""
        # Convert image to tensor and normalize
        if not isinstance(distorted_image, torch.Tensor):
            distorted_image = torch.from_numpy(distorted_image).to(self.device).float() / 255.0
        
        output_width, output_height = output_resolution
        
        # Calculate proper FOV scaling
        max_fov = np.pi / 2  # 90 degrees for fisheye
        f_scale = 1.0 / np.tan(max_fov / 2)  # Proper focal length scaling
        
        # Create normalized pixel coordinates
        y_range = torch.linspace(-1, 1, output_height, device=self.device) 
        x_range = torch.linspace(-1, 1, output_width, device=self.device)
        y_coords, x_coords = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Scale coordinates based on focal length and principal point
        x_coords = x_coords * self.f_pinhole / f_scale
        y_coords = y_coords * self.f_pinhole / f_scale
        
        # Convert to 3D rays using proper spherical projection
        r = torch.sqrt(x_coords**2 + y_coords**2)
        theta = torch.atan2(r, self.f_pinhole)  # Proper angle calculation
        phi = torch.atan2(y_coords, x_coords)
        
        # Convert to 3D points on unit sphere
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        
        # Stack and reshape for batch processing
        rays = torch.stack([x, y, z], dim=-1)
        rays = rays.reshape(-1, 3)
        
        # Process in batches
        coords_list = []
        num_pixels = rays.shape[0]
        
        for i in range(0, num_pixels, batch_size):
            batch_rays = rays[i:i + batch_size]
            coords = self.project_fisheye_radtan_thinprism_batch(batch_rays)
            coords_list.append(coords)
        
        coords = torch.cat(coords_list, dim=0)
        
        # Normalize coordinates for grid_sample with proper bounds checking
        height, width = distorted_image.shape[:2]
        coords_normalized = torch.stack([
            torch.clamp((coords[:, 0] / (width - 1)) * 2 - 1, -1.1, 1.1),
            torch.clamp((coords[:, 1] / (height - 1)) * 2 - 1, -1.1, 1.1)
        ], dim=-1)
        
        # Reshape for grid_sample
        coords_normalized = coords_normalized.reshape(1, output_height, output_width, 2)
        
        # Prepare image for sampling
        if len(distorted_image.shape) == 3:
            distorted_image = distorted_image.permute(2, 0, 1).unsqueeze(0)
        
        # Sample with proper interpolation
        undistorted = F.grid_sample(
            distorted_image,
            coords_normalized,
            mode='bilinear',
            padding_mode='border',  # Change to 'border' for better edge handling
            align_corners=False
        )
        
        # Convert back to numpy with proper scaling
        result = undistorted.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result

if __name__ == "__main__":
    # Example Aria Fisheye intrinsics with 6 KB parameters
    aria_fisheye_intrinsics_kb6 = {
        'fx': 241.092,
        'fy': 241.092,
        'cy': 316.638,
        'cx': 237.025,
        'k0': -0.0255612,
        'k1': 0.0984733,
        'k2': -0.0675009,
        'k3': 0.00948425,
        'k4': 0.00233176,
        'k5': -0.000574631,
        'p0': 0.00114624,
        'p1': -0.00149885,
        's0': -0.00123338,
        's1': -0.000119675,
        's2': 0.0022857,
        's3': -5.6488e-05,
    }

    # Initialize converter
    original_cx = int(round(aria_fisheye_intrinsics_kb6['cx']))
    original_cy = int(round(aria_fisheye_intrinsics_kb6['cy']))
    original_principal_point = (original_cx, original_cy)
    
    pinhole_focal_length = 241.092
    pinhole_principal_point = (440, 440)
    
    # Convert principal points to integers for OpenCV
    pinhole_cx = int(round(pinhole_principal_point[0]))
    pinhole_cy = int(round(pinhole_principal_point[1]))
    pinhole_principal_point = (pinhole_cx, pinhole_cy)
    
    converter_kb6 = AriaCameraModelConverterTorch(
        aria_fisheye_intrinsics_kb6, 
        pinhole_focal_length, 
        pinhole_principal_point
    )

    # Load image with correct path
    img_path = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/other_projects/dust3r/logs/test1/slam_right_image.png"
    
    distorted_image = cv2.imread(img_path)
    if distorted_image is None:
        raise FileNotFoundError(f"Could not read image from {img_path}. Please check if file exists.")
        
    print(f"Loaded image shape: {distorted_image.shape}")

    # Adjust output parameters for better results
    output_resolution = (640, 480)  # Match input resolution
    pinhole_focal_length = aria_fisheye_intrinsics_kb6['fx'] * 0.8  # Reduce FOV slightly
    pinhole_principal_point = (output_resolution[0]//2, output_resolution[1]//2)

    # Process image
    undistorted_image_kb6 = converter_kb6.undistort_image_to_pinhole(
        distorted_image, 
        output_resolution
    )

    # Convert to uint8 for OpenCV operations
    undistorted_image_kb6 = (undistorted_image_kb6 * 255).astype(np.uint8)

    # Save output in the same directory as input
    output_dir = os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f"undistorted_kb6_{base_name}.png")
    
    cv2.imwrite(output_path, undistorted_image_kb6)
    print(f"Saved undistorted image to: {output_path}")

    # Display results with principal points
    print(f"Drawing original principal point at: {original_principal_point}")
    print(f"Drawing pinhole principal point at: {pinhole_principal_point}")
    
    # Draw circles on copies of the images to avoid modifying originals
    display_original = distorted_image.copy()
    display_undistorted = undistorted_image_kb6.copy()
    
    cv2.circle(display_original, original_principal_point, 5, (0, 0, 255), -1)
    cv2.circle(display_undistorted, pinhole_principal_point, 5, (0, 0, 255), -1)

    cv2.imshow("Original", display_original)
    cv2.imshow("Undistorted", display_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
