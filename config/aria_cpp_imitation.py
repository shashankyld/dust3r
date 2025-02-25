import os
import sys
import cv2
import numpy as np
from scipy.optimize import newton

# Add the parent directory to Python path (if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import camera_calibrations  # Assuming you have a config.py

def get_fisheye624_params(camera_name):
    """Get fisheye624 parameters from config for a specific camera."""
    if camera_name not in camera_calibrations:
        raise ValueError(f"Camera {camera_name} not found in calibrations")
        
    camera = camera_calibrations[camera_name]
    params = camera['projection_params']
    
    # Get individual parameters correctly
    aria_fisheye_intrinsics = {
        'fx': params[0],  # Single value
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

def fisheye_rad_tan_thin_prism_project(point_optical, params):
    """
    Projects a 3D point in camera coordinates to image coordinates
    using the fisheye model with radial, tangential, and thin-prism distortion.
    """
    fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = params

    # Make sure the point is not on the image plane
    if point_optical[2] == 0:
        return None, None

    # Normalize coordinates
    x, y = point_optical[:2] / point_optical[2]
    
    # Convert to polar coordinates
    r = np.sqrt(x*x + y*y)
    if r < 1e-8:
        return cu, cv
    
    # Calculate theta (angle from optical axis)
    theta = np.arctan(r)
    theta2 = theta * theta
    
    # Apply distortion model
    theta_d = theta * (1.0 + k0*theta2 + k1*theta2*theta2 + k2*theta2*theta2*theta2)
    
    # Scale factor
    scale = theta_d / r
    
    # Apply scaled coordinates
    xp = x * scale
    yp = y * scale
    
    # Apply tangential and thin prism distortion
    r2 = xp*xp + yp*yp
    
    # Tangential distortion
    tdx = p0 * (2*xp*xp + r2) + 2*p1*xp*yp
    tdy = p1 * (2*yp*yp + r2) + 2*p0*xp*yp
    
    # Thin prism distortion
    tpx = s0*r2 + s1*r2*r2
    tpy = s2*r2 + s3*r2*r2
    
    # Final projected coordinates
    u = fu * (xp + tdx + tpx) + cu
    v = fv * (yp + tdy + tpy) + cv
    
    return u, v

def fisheye_rad_tan_thin_prism_unproject(p, params):
    """
    Unprojects a pixel coordinate to a 3D point in camera coordinates
    using the fisheye model with radial, tangential, and thin-prism distortion.
    """
    fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = params

    # Get uvDistorted
    uv_distorted = (p - [cu, cv]) / fu  # Assuming single focal length

    # Get xr_yr from uvDistorted
    xr_yr = compute_xr_yr_from_uvDistorted(uv_distorted, params)

    # Early exit if point is in the center of the image
    xr_yr_norm = np.linalg.norm(xr_yr)
    if xr_yr_norm == 0:
        return np.array()

    # Otherwise, find theta
    theta = getThetaFromNorm_xr_yr(xr_yr_norm, params)

    # Get the point coordinates
    point3d_est = np.zeros(3)
    point3d_est[:2] = np.tan(theta) / xr_yr_norm * xr_yr
    point3d_est = 1.0

    return point3d_est

def compute_xr_yr_from_uvDistorted(uv_distorted, params):
    """
    Helper function to compute [x_r; y_r] from uvDistorted using Newton's method.
    """
    fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = params

    # Initial guess
    xr_yr = uv_distorted.copy()

    # Newton iterations
    for _ in range(10):  # Adjust max iterations as needed
        # Compute the estimated uvDistorted
        uv_distorted_est = xr_yr.copy()
        xr_yr_squared_norm = np.sum(xr_yr**2)

        # Add tangential distortion
        temp = 2 * np.dot(xr_yr, [p0, p1])
        uv_distorted_est += temp * xr_yr + xr_yr_squared_norm * np.array([p0, p1])

        # Add thin prism distortion
        radial_powers_2_and_4 = np.array([xr_yr_squared_norm, xr_yr_squared_norm**2])
        uv_distorted_est += np.dot([s0, s1], radial_powers_2_and_4)
        uv_distorted_est += np.dot([s2, s3], radial_powers_2_and_4)

        # Compute the Jacobian of uvDistorted wrt xr_yr
        duv_distorted_dxryr = np.eye(2)  # Initialize as identity
        # Add tangential part
        duv_distorted_dxryr += 6 * xr_yr * p0 + 2 * xr_yr * p1
        duv_distorted_dxryr += 2 * (xr_yr * p1 + xr_yr * p0)
        duv_distorted_dxryr += 2 * (xr_yr * p1 + xr_yr * p0)
        duv_distorted_dxryr += 6 * xr_yr * p1 + 2 * xr_yr * p0
        # Add thin prism part
        temp1 = 2 * (s0 + 2 * s1 * xr_yr_squared_norm)
        duv_distorted_dxryr += xr_yr * temp1
        duv_distorted_dxryr += xr_yr * temp1
        temp2 = 2 * (s2 + 2 * s3 * xr_yr_squared_norm)
        duv_distorted_dxryr += xr_yr * temp2
        duv_distorted_dxryr += xr_yr * temp2

        # Compute correction
        correction = np.linalg.solve(duv_distorted_dxryr, uv_distorted - uv_distorted_est)

        # Apply correction
        xr_yr += correction

        # Check for convergence (adjust tolerance as needed)
        if np.linalg.norm(correction) < 1e-6:
            break

    return xr_yr

def getThetaFromNorm_xr_yr(th_radial_desired, params):
    """
    Helper function to compute theta from the norm of [x_r; y_r] using Newton's method.
    """
    fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = params

    # Initial guess
    th = th_radial_desired

    # Newton iterations
    for _ in range(10):  # Adjust max iterations as needed
        theta_sq = th**2

        th_radial = th
        dthD_dth = 1.0
        theta2is = theta_sq
        for i, k in enumerate([k0, k1, k2, k3, k4, k5]):  # Assuming fisheye624
            th_radial += k * theta2is
            dthD_dth += (2 * i + 3) * k * theta2is
            theta2is *= theta_sq

        # Compute the correction
        step = (th_radial_desired - th_radial) / dthD_dth if abs(dthD_dth) > np.finfo(float).eps else \
               10 * np.finfo(float).eps if (th_radial_desired - th_radial) * dthD_dth > 0 else \
               -10 * np.finfo(float).eps

        # Apply correction
        th += step

        # Check for convergence (adjust tolerance as needed)
        if abs(step) < 1e-6:
            break

        # Revert to within 180 degrees FOV to avoid numerical overflow
        if abs(th) >= np.pi / 2.0:
            th = 0.999 * np.pi / 2.0

    return th

def undistort_fisheye_newton(image, camera_name, destination_params):
    """Undistorts a fisheye image to a pinhole image using Newton's method."""
    src_height, src_width = image.shape[:2]
    dest_height, dest_width = destination_params['image_size']
    
    # Fix output array creation
    undistorted = np.zeros((dest_height, dest_width, image.shape[2]), dtype=image.dtype)

    # Get camera parameters
    camera_params = get_fisheye624_params(camera_name)
    camera_params_list = [
        camera_params['fx'], camera_params['fy'],
        camera_params['cx'], camera_params['cy'],
        camera_params['k0'], camera_params['k1'],
        camera_params['k2'], camera_params['k3'],
        camera_params['k4'], camera_params['k5'],
        camera_params['p0'], camera_params['p1'],
        camera_params['s0'], camera_params['s1'],
        camera_params['s2'], camera_params['s3']
    ]

    # Destination camera parameters
    fx_dest, fy_dest = destination_params['focal_length']
    cx_dest, cy_dest = destination_params['principal_point']

    # Precompute coordinates
    print("Processing pixels...")
    y_coords, x_coords = np.mgrid[0:dest_height, 0:dest_width]
    for y_dest, x_dest in zip(y_coords.flatten(), x_coords.flatten()):
        # Convert to normalized coordinates
        x_norm = (x_dest - cx_dest) / fx_dest
        y_norm = (y_dest - cy_dest) / fy_dest

        # Compute ray direction
        r = np.sqrt(x_norm*x_norm + y_norm*y_norm)
        theta = np.arctan(r)
        phi = np.arctan2(y_norm, x_norm)

        try:
            # Project to fisheye coordinates
            point3d = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            u, v = fisheye_rad_tan_thin_prism_project(point3d, camera_params_list)
            
            if u is not None and v is not None:
                u = int(round(u))
                v = int(round(v))
                if 0 <= u < src_width and 0 <= v < src_height:
                    undistorted[y_dest, x_dest] = image[v, u]
                    
        except (ValueError, RuntimeError) as e:
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

    # Test the undistortion
    path = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/other_projects/dust3r/config/slam_right_image.png"
    image = cv2.imread(path)

    # Adjust destination parameters to avoid border artifacts
    image_height, image_width = image.shape[:2]
    print("Height and width of the original image: ", image_height, image_width)
    scale_factor = 1  # Reduce output size to avoid edge artifacts

    destination_params = {
        'focal_length':  [image_width // 2, image_height // 2],  # Assuming square pixels
        'principal_point': [image_width // 2, image_height // 2],  # Centered
        'image_size': [image_width // scale_factor, image_height // scale_factor]
    }

    print(f"Destination parameters: {destination_params}")

    undistorted = undistort_fisheye_newton(image, camera_name, destination_params)

    cv2.imshow("Original", image)
    cv2.imshow("Undistorted", undistorted)

    # Save the undistorted image
    name_tag = path.split("/")[-1].split(".")
    cv2.imwrite(f"undistorted_{name_tag}.png", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()