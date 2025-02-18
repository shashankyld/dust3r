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
    
    # The order of fish624 parameters is (fx = fy), cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3
    aria_fisheye_intrinsics = {
        'fx': params,
        'fy': params,  # fx = fy for Aria cameras
        'cx': params,
        'cy': params,
        'k0': params,
        'k1': params,
        'k2': params,
        'k3': params,
        'k4': params,
        'k5': params,
        'p0': params,
        'p1': params,
        's0': params,
        's1': params,
        's2': params,
        's3': params
    }
    
    return aria_fisheye_intrinsics

def fisheye_rad_tan_thin_prism_project(point_optical, params):
    """
    Projects a 3D point in camera coordinates to image coordinates
    using the fisheye model with radial, tangential, and thin-prism distortion.
    """
    fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3 = params

    # Make sure the point is not on the image plane
    if point_optical == 0:
        raise ValueError("Point cannot be on the image plane (z = 0)")

    # Compute [a; b] = [x/z; y/z]
    inv_z = 1.0 / point_optical
    ab = point_optical[:2] * inv_z

    # Compute the squares of the elements of ab
    ab_squared = ab**2

    # These will be used in multiple computations
    r_sq = ab_squared + ab_squared
    r = np.sqrt(r_sq)
    th = np.arctan(r)
    theta_sq = th**2

    # Compute the theta polynomial
    th_radial = 1.0
    theta2is = theta_sq
    for i, k in enumerate([k0, k1, k2, k3, k4, k5]):  # Assuming fisheye624
        th_radial += theta2is * k
        theta2is *= theta_sq

    # Compute th/r, using the limit for small values
    th_divr = 1.0 if r < np.finfo(float).eps else th / r

    # The distorted coordinates -- except for focal length and principal point
    # Start with the radial term:
    xr_yr = (th_radial * th_divr) * ab
    xr_yr_squared_norm = np.sum(xr_yr**2)

    # Start computing the output: first the radially-distorted terms,
    # then add more as needed
    uv_distorted = xr_yr.copy()

    # Add tangential distortion
    temp = 2 * np.dot(xr_yr, [p0, p1])
    uv_distorted += temp * xr_yr + xr_yr_squared_norm * np.array([p0, p1])

    # Add thin prism distortion
    radial_powers_2_and_4 = np.array([xr_yr_squared_norm, xr_yr_squared_norm**2])
    uv_distorted += np.dot([s0, s1], radial_powers_2_and_4)
    uv_distorted += np.dot([s2, s3], radial_powers_2_and_4)

    # Compute the return value
    return fu * uv_distorted + [cu, cv]  # Assuming single focal length


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
    """
    Undistorts a fisheye image to a pinhole image using Newton's method.
    """
    src_height, src_width = image.shape[:2]
    dest_height, dest_width = destination_params['image_size']
    undistorted = np.zeros((dest_height, dest_width, image.shape), dtype=image.dtype)

    # Get camera parameters
    camera_params = get_fisheye624_params(camera_name)
    fu, fv, cu, cv = camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy']
    k0, k1, k2, k3, k4, k5 = camera_params['k0'], camera_params['k1'], camera_params['k2'], camera_params['k3'], camera_params['k4'], camera_params['k5']
    p0, p1 = camera_params['p0'], camera_params['p1']
    s0, s1, s2, s3 = camera_params['s0'], camera_params['s1'], camera_params['s2'], camera_params['s3']
    camera_params_list = [fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3]

    # Destination camera parameters
    fx_dest, fy_dest = destination_params['focal_length']
    cx_dest, cy_dest = destination_params['principal_point']

    # Iterate over destination image pixels
    for y_dest in range(dest_height):
        for x_dest in range(dest_width):
            # 1. Convert destination pixel to normalized pinhole coordinates
            x_norm = (x_dest - cx_dest) / fx_dest
            y_norm = (y_dest - cy_dest) / fy_dest

            # 2. Convert to polar coordinates (theta, phi)
            theta = np.arctan2(np.sqrt(x_norm**2 + y_norm**2), 1)
            phi = np.arctan2(y_norm, x_norm)

            # 3. Use Newton's method to find r_theta from theta
            r_theta = newton(
                lambda r: getThetaFromNorm_xr_yr(r, camera_params_list) - theta,
                theta,  # Initial guess
            )

            # 4. Convert (r_theta, phi) to distorted (u, v) using the fisheye model
            u, v = fisheye_rad_tan_thin_prism_project(
                np.array([r_theta * np.cos(phi), r_theta * np.sin(phi), 1]),
                camera_params_list
            )

            # 5. Sample from the source image (with bounds checking)
            u = int(round(u))
            v = int(round(v))
            if 0 <= u < src_width and 0 <= v < src_height:
                undistorted[y_dest, x_dest] = image[v, u]

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