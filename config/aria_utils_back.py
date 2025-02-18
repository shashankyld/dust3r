import numpy as np
from scipy.optimize import newton
import cv2
class AriaCameraModelConverter:
    def __init__(self, intrinsics_aria_fisheye, f_pinhole, c_pinhole):
        """
        Initializes the converter with Aria fisheye intrinsics and desired pinhole parameters.

        Args:
            intrinsics_aria_fisheye (dict): Dictionary containing intrinsics for FisheyeRadTanThinPrism model.
                                            Expected keys: fx, fy, cx, cy, k0, k1, k2, k3, p0, p1, s0, s1, s2, s3
                                            Now supporting 6 KB params: k0, k1, k2, k3, k4, k5
            f_pinhole (float): Focal length for pinhole camera.
            c_pinhole (tuple): Principal point (cx, cy) for pinhole camera.
        """
        self.aria_intrinsics = intrinsics_aria_fisheye
        self.f_pinhole = f_pinhole
        self.c_pinhole = c_pinhole

        # Assume fx and fy are equal for Aria models as mentioned in the note.
        if 'fx' in self.aria_intrinsics and 'fy' in self.aria_intrinsics:
            if not np.isclose(self.aria_intrinsics['fx'], self.aria_intrinsics['fy']):
                print("Warning: fx and fy are not equal in provided Aria intrinsics. Using fx for both.")
                self.aria_intrinsics['fy'] = self.aria_intrinsics['fx']


    def _r_theta(self, theta, k):
        """Radial distortion polynomial r(theta) for KB and Fisheye models.
           Now supports up to 6 parameters (k0-k5)."""
        r_theta_val = theta
        theta_power = theta
        for ki in k:
            theta_power *= theta**2
            r_theta_val += ki * theta_power
        return r_theta_val

    def _inverse_r_theta_newton(self, r_val, k, initial_theta=0.1):
        # Add checks for numerical stability
        if r_val < 0:
            print("Warning: Negative r_val encountered")
            r_val = abs(r_val)
        
        if np.isclose(r_val, 0):
            return 0.0  # Return zero for zero input
            
        # Add maximum theta limit to prevent divergence
        MAX_THETA = np.pi  # Maximum reasonable theta value
        
        def func(theta):
            if theta > MAX_THETA:
                return float('inf')
            return self._r_theta(theta, k) - r_val
        def fprime(theta): # Derivative of r(theta) w.r.t theta
            deriv = 1.0
            theta_p = theta
            power = 3
            for ki in k:
                deriv += ki * power * theta_p
                theta_p *= theta**2
                power += 2
            return deriv

        try:
            theta_approx = newton(func, initial_theta, fprime=fprime, maxiter=100, tol=1e-9) # Increased maxiter and added tol
            if not np.isfinite(theta_approx) or theta_approx < 0: # Handle potential non-convergence or negative values
                raise RuntimeError("Newton-Raphson did not converge to a valid theta")
            return theta_approx
        except RuntimeError as e:
            print(f"Warning: Inverse r(theta) Newton method failed: {e}. Returning initial guess. Consider adjusting initial guess or distortion parameters.")
            return initial_theta # Return initial guess in case of failure


    # Projection functions for Aria Models

    def project_linear(self, xyz):
        """Linear (Pinhole) projection model."""
        x, y, z = xyz
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        u = fx * x / z + cx
        v = fy * y / z + cy
        return np.array([u, v])

    def project_spherical(self, xyz):
        """Spherical projection model."""
        x, y, z = xyz
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        u = fx * theta * np.cos(phi) + cx
        v = fy * theta * np.sin(phi) + cy
        return np.array([u, v])

    def project_kb3(self, xyz):
        """Kannala-Brandt K3/K4 projection model. Now supports up to 6 radial params."""
        x, y, z = xyz
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        # Dynamically handle number of k parameters. Assumes keys are k0, k1, k2...
        k_params = []
        k_index = 0
        while f'k{k_index}' in self.aria_intrinsics:
            k_params.append(self.aria_intrinsics[f'k{k_index}'])
            k_index += 1
        k = k_params

        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        r_val = self._r_theta(theta, k)
        ur = r_val * np.cos(phi)
        vr = r_val * np.sin(phi)
        u = fx * ur + cx
        v = fy * vr + cy
        return np.array([u, v])

    def project_fisheye62(self, xyz):
        """Fisheye62 projection model. Uses KB model with potentially 6 params."""
        x, y, z = xyz
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        # Dynamically handle k parameters like in project_kb3
        k_params = []
        k_index = 0
        while f'k{k_index}' in self.aria_intrinsics:
            k_params.append(self.aria_intrinsics[f'k{k_index}'])
            k_index += 1
        k = k_params
        p = [self.aria_intrinsics['p0'], self.aria_intrinsics['p1']]
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        r_val = self._r_theta(theta, k)
        ur = r_val * np.cos(phi)
        vr = r_val * np.sin(phi)
        tx = p[0] * (2 * ur**2 + r_val**2) + 2 * p[1] * ur * vr
        ty = p[1] * (2 * vr**2 + r_val**2) + 2 * p[0] * ur * vr
        u = fx * (ur + tx) + cx
        v = fy * (vr + ty) + cy
        return np.array([u, v])

    def project_fisheye_radtan_thinprism(self, xyz):
        """FisheyeRadTanThinPrism (Fisheye624) projection model. Uses KB model with potentially 6 params."""
        x, y, z = xyz
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        # Dynamically handle k parameters like in project_kb3
        k_params = []
        k_index = 0
        while f'k{k_index}' in self.aria_intrinsics:
            k_params.append(self.aria_intrinsics[f'k{k_index}'])
            k_index += 1
        k = k_params
        p = [self.aria_intrinsics['p0'], self.aria_intrinsics['p1']]
        s = [self.aria_intrinsics['s0'], self.aria_intrinsics['s1'], self.aria_intrinsics['s2'], self.aria_intrinsics['s3']]

        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        r_val = self._r_theta(theta, k)
        ur = r_val * np.cos(phi)
        vr = r_val * np.sin(phi)
        tx = p[0] * (2 * ur**2 + r_val**2) + 2 * p[1] * ur * vr
        ty = p[1] * (2 * vr**2 + r_val**2) + 2 * p[0] * ur * vr
        tpx = s[0] * r_val**2 + s[1] * r_val**4
        tpy = s[2] * r_val**2 + s[3] * r_val**4
        u = fx * (ur + tx + tpx) + cx
        v = fy * (vr + ty + tpy) + cy
        return np.array([u, v])


    # Unprojection functions for Aria Models (Output: theta, phi)

    def unproject_linear(self, uv):
        """Unproject from 2D pixel to polar coordinates (theta, phi) for Linear model."""
        u, v = uv
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        x_over_z = (u - cx) / fx
        y_over_z = (v - cy) / fy
        theta = np.arctan2(np.sqrt(x_over_z**2 + y_over_z**2), 1) # z=1 implicitly
        phi = np.arctan2(y_over_z, x_over_z)
        return theta, phi

    def unproject_spherical(self, uv):
        """Unproject from 2D pixel to polar coordinates (theta, phi) for Spherical model."""
        u, v = uv
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        theta_sq_x = ((u - cx) / fx)**2
        theta_sq_y = ((v - cy) / fy)**2
        theta = np.sqrt(theta_sq_x + theta_sq_y) # Corrected formula from prompt
        # phi = np.arctan2((v - cy) / fy, (u - cx) / fx) # Corrected arctan order.
        # Inverted order 
        phi = np.arctan2((u - cx) / fx, (v - cy) / fy)

        return theta, phi


    def unproject_kb3(self, uv):
        """Unproject from 2D pixel to polar coordinates (theta, phi) for KB3/KB4 model. Supports up to 6 radial params."""
        u, v = uv
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        # Dynamically handle k parameters like in project_kb3
        k_params = []
        k_index = 0
        while f'k{k_index}' in self.aria_intrinsics:
            k_params.append(self.aria_intrinsics[f'k{k_index}'])
            k_index += 1
        k = k_params
        phi = np.arctan2((v - cy) / fy, (u - cx) / fx) # Corrected arctan order.
        r_val_squared = ((u - cx) / fx)**2 + ((v - cy) / fy)**2
        r_val = np.sqrt(r_val_squared)

        # Solve for theta using Newton's method to invert r(theta)
        initial_theta_guess = r_val # Initial guess, could be improved
        theta = self._inverse_r_theta_newton(r_val, k, initial_theta_guess)

        return theta, phi

    def unproject_fisheye62(self, uv):
        """Unproject from 2D pixel to polar coordinates (theta, phi) for Fisheye62 model. Uses KB unprojection with potentially 6 params."""
        u_pixel, v_pixel = uv
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        p = [self.aria_intrinsics['p0'], self.aria_intrinsics['p1']]
        # Dynamically handle k parameters like in project_kb3
        k_params = []
        k_index = 0
        while f'k{k_index}' in self.aria_intrinsics:
            k_params.append(self.aria_intrinsics[f'k{k_index}'])
            k_index += 1
        k = k_params


        u_ratio = (u_pixel - cx) / fx
        v_ratio = (v_pixel - cy) / fy

        def fisheye62_inverse_func(r_uv):
            ur, vr = r_uv
            r_theta_val_current = np.sqrt(ur**2 + vr**2) # r(theta) based on current ur, vr -
            tx = p[0] * (2 * ur**2 + r_theta_val_current**2) + 2 * p[1] * ur * vr
            ty = p[1] * (2 * vr**2 + r_theta_val_current**2) + 2 * p[0] * ur * vr
            return [ur + tx - u_ratio, vr + ty - v_ratio] # Functions to drive to zero

        initial_guess = np.array([u_ratio, v_ratio]) #Initial guess could be improved.
        try:
            ur_vr_approx, _ = newton(fisheye62_inverse_func, initial_guess, maxiter=100, tol=1e-9) # Increased maxiter and added tol
            ur, vr = ur_vr_approx

            r_val = np.sqrt(ur**2 + vr**2)

            theta = self._inverse_r_theta_newton(r_val, k, initial_theta=np.arctan2(r_val, 1.0))  # Using r_val as initial guess for theta in KB3 unproject
            phi = np.arctan2(vr, ur)
            return theta, phi

        except RuntimeError as e:
            print(f"Warning: Fisheye62 unprojection Newton method failed: {e}. Returning KB3 unprojection as fallback with initial ratios as input.")
            # Fallback to KB3 unprojection with initial guess based on ratios in case of Newton failure in fisheye62 itself.
            # This approximates as if tangential distortion is negligible for unprojection if Newton fails.
            return self.unproject_kb3(uv) # Fallback to kb3 if fisheye62 fails, still using ratios as input


    def unproject_fisheye_radtan_thinprism(self, uv):
        """Unproject from 2D pixel to polar coordinates (theta, phi) for FisheyeRadTanThinPrism model. Uses KB unprojection with potentially 6 params."""
        u_pixel, v_pixel = uv
        fx, fy, cx, cy = self.aria_intrinsics['fx'], self.aria_intrinsics['fy'], self.aria_intrinsics['cx'], self.aria_intrinsics['cy']
        p = [self.aria_intrinsics['p0'], self.aria_intrinsics['p1']]
        s = [self.aria_intrinsics['s0'], self.aria_intrinsics['s1'], self.aria_intrinsics['s2'], self.aria_intrinsics['s3']]
        # Dynamically handle k parameters like in project_kb3
        k_params = []
        k_index = 0
        while f'k{k_index}' in self.aria_intrinsics:
            k_params.append(self.aria_intrinsics[f'k{k_index}'])
            k_index += 1
        k = k_params


        u_ratio = (u_pixel - cx) / fx
        v_ratio = (v_pixel - cy) / fy


        def fisheye_radtan_thinprism_inverse_func(r_uv):
            ur, vr = r_uv
            r_theta_val_current = np.sqrt(ur**2 + vr**2)
            tx = p[0] * (2 * ur**2 + r_theta_val_current**2) + 2 * p[1] * ur * vr
            ty = p[1] * (2 * vr**2 + r_theta_val_current**2) + 2 * p[0] * ur * vr
            tpx = s[0] * r_theta_val_current**2 + s[1] * r_theta_val_current**4
            tpy = s[2] * r_theta_val_current**2 + s[3] * r_theta_val_current**4
            return [ur + tx + tpx - u_ratio, vr + ty + tpy - v_ratio] # Functions to drive to zero

        initial_guess = np.array([u_ratio, v_ratio]) #Initial guess could be improved
        try:
            ur_vr_approx, _ = newton(fisheye_radtan_thinprism_inverse_func, initial_guess, maxiter=100, tol=1e-9) # Increased maxiter and added tol
            ur, vr = ur_vr_approx


            r_val = np.sqrt(ur**2 + vr**2)

            theta = self._inverse_r_theta_newton(r_val, k, initial_theta=np.arctan2(r_val, 1.0)) # Using r_val as initial guess for theta in KB3 unproject
            phi = np.arctan2(vr, ur)
            return theta, phi
        except RuntimeError as e:
            print(f"Warning: FisheyeRadTanThinPrism unprojection Newton method failed: {e}. Returning KB3 unprojection as fallback with initial ratios as input.")
            # Fallback to KB3 unprojection - approximating that tangential and thin prism distortions are negligible for unprojection if Newton fails.
            return self.unproject_kb3(uv) # Fallback to kb3 if fisheye624 fails, still using ratios as input


    # Distortion Correction - Aria Fisheye to Pinhole

    def undistort_image_to_pinhole(self, distorted_image, output_resolution):
        """
        Undistorts a given Aria fisheye distorted image to a pinhole image.

        Args:
            distorted_image (np.array): Input distorted image (height x width x channels).
            output_resolution (tuple): Desired (width, height) of the undistorted pinhole image.

        Returns:
            np.array: Undistorted pinhole image.
        """
        distorted_height, distorted_width = distorted_image.shape[:2]
        output_width, output_height = output_resolution

        undistorted_image = np.zeros((output_height, output_width, distorted_image.shape[2]), dtype=distorted_image.dtype) # Initialize output image
        

        pinhole_cx, pinhole_cy = self.c_pinhole
        pinhole_fx = self.f_pinhole
        pinhole_fy = self.f_pinhole # Assuming square pixels for pinhole

        x_map = np.zeros((output_height, output_width), dtype=np.float32)
        y_map = np.zeros((output_height, output_width), dtype=np.float32)

        for y_undistorted in range(output_height):
            for x_undistorted in range(output_width):
                # Pixel coordinates in undistorted pinhole image
                u_pinhole = x_undistorted
                v_pinhole = y_undistorted

                # Unproject from pinhole to normalized 3D direction (x/z, y/z, 1) - effectively theta, phi for linear model
                x_normalized = (u_pinhole - pinhole_cx) / pinhole_fx
                y_normalized = (v_pinhole - pinhole_cy) / pinhole_fy

                theta_pinhole = np.arctan2(np.sqrt(x_normalized**2 + y_normalized**2), 1)
                phi_pinhole = np.arctan2(y_normalized, x_normalized)


                # Project the 3D direction (represented by theta, phi) to distorted Aria fisheye image
                # We can use spherical model for intermediate 3D point representation as direction is important, not absolute scale.
                # For simplicity, assume z=1 for intermediate 3D point calculation.
                x_camera = np.tan(theta_pinhole) * np.cos(phi_pinhole)
                y_camera = np.tan(theta_pinhole) * np.sin(phi_pinhole)
                z_camera = 1.0

                uv_distorted = self.project_fisheye_radtan_thinprism(np.array([x_camera, y_camera, z_camera])) # Use full fisheye model for distortion


                x_distorted_float, y_distorted_float = uv_distorted

                # Bilinear interpolation - or simpler nearest neighbor for speed.
                x_distorted = int(np.round(x_distorted_float)) # Nearest neighbor for now - bilinear would be better quality
                y_distorted = int(np.round(y_distorted_float))


                if 0 <= x_distorted < distorted_width and 0 <= y_distorted < distorted_height:
                    undistorted_image[y_undistorted, x_undistorted, :] = distorted_image[y_distorted, x_distorted, :] # Nearest neighbor assignment


        return undistorted_image

    def project_points_to_aria(self, xyz_points, model_type='fisheye_radtan_thinprism'):
        """
        Projects a list of 3D points to 2D pixel coordinates using the specified Aria model.

        Args:
            xyz_points (np.array): (N, 3) array of 3D points.
            model_type (str):  'linear', 'spherical', 'kb3', 'fisheye62', 'fisheye_radtan_thinprism'.
                                Model to use for projection. Default: 'fisheye_radtan_thinprism'

        Returns:
            np.array: (N, 2) array of 2D pixel coordinates.
        """
        projection_function = {
            'linear': self.project_linear,
            'spherical': self.project_spherical,
            'kb3': self.project_kb3,
            'fisheye62': self.project_fisheye62,
            'fisheye_radtan_thinprism': self.project_fisheye_radtan_thinprism
        }.get(model_type.lower())

        if not projection_function:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from 'linear', 'spherical', 'kb3', 'fisheye62', 'fisheye_radtan_thinprism'")

        uv_points = []
        for xyz in xyz_points:
            uv_points.append(projection_function(xyz))
        return np.array(uv_points)


# --- Example Usage ---
if __name__ == '__main__':
    # Example Aria Fisheye intrinsics with 6 KB parameters (replace with your actual calibration)
    aria_fisheye_intrinsics_kb6 = {
        'fx': 241.092,  # Example values - replace with actual
        'fy': 241.092,
        'cx': 316.638,
        'cy': 237.025,
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
    pinhole_focal_length = 241.092 # Example - adjust as needed
    pinhole_principal_point = (440, 440) # Example - adjust as needed


    converter_kb6 = AriaCameraModelConverter(aria_fisheye_intrinsics_kb6, pinhole_focal_length, pinhole_principal_point)



    # --- Image Undistortion Example with KB6 params ---
    # Create a dummy distorted image (replace with your actual image loading)
    distorted_image = np.zeros((960, 1080, 3), dtype=np.uint8)
    # Box at center 
    distorted_image[460:500, 540:540+40, :] = [255, 0, 0]
    # One more box at top left different color and all other corners 
    distorted_image[20:60, 20:60, :] = [0, 255, 0]
    distorted_image[20:60, 1020:1060, :] = [0, 0, 255]
    distorted_image[920:960, 20:60, :] = [255, 255, 0]
    distorted_image[920:960, 1020:1060, :] = [255, 0, 255]

    # Add grid lines for visualization 
    distorted_image[::40, :, :] = 128
    distorted_image[:, ::40, :] = 128

    # Load from path 
    img_path = "slam_right_image.png"
    distorted_image = cv2.imread(img_path)
    print (distorted_image.shape)


    output_resolution = (880, 880) # Same as distorted resolution for now, adjust if needed. why dimenstions are swapped? = because of the row major order in numpy
    undistorted_image_pinhole_kb6 = converter_kb6.undistort_image_to_pinhole(distorted_image, output_resolution)


    import cv2
    cv2.imwrite("distorted_example_kb6.png", distorted_image) # Save dummy distorted image
    cv2.imwrite("undistorted_pinhole_example_kb6.png", undistorted_image_pinhole_kb6) # Save undistorted image
    print("Distorted and undistorted images saved as distorted_example_kb6.png and undistorted_pinhole_example_kb6.png (using KB6 params)")
