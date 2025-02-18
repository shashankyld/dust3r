import math

def project(x, y, z, fx, fy, cx, cy, k0, k1, k2, k3, p0, p1, s0, s1, s2, s3):
  """
  Projects a 3D point (x, y, z) in the camera's local frame 
  to a 2D pixel coordinate (u, v) using the FisheyeRadTanThinPrism model.
  """

  # 1. Convert to polar coordinates
  theta = math.atan2(math.sqrt(x**2 + y**2), z)
  phi = math.atan2(y, x)

  # 2. Radial distortion
  r_theta = theta + k0*theta**3 + k1*theta**5 + k2*theta**7 + k3*theta**9

  # 3. Tangential distortion
  ur = r_theta * math.cos(phi)
  vr = r_theta * math.sin(phi)
  tx = p0 * (2*ur**2 + r_theta**2) + 2*p1*ur*vr
  ty = p1 * (2*vr**2 + r_theta**2) + 2*p0*ur*vr

  # 4. Thin-prism distortion
  tpx = s0*r_theta**2 + s1*r_theta**4
  tpy = s2*r_theta**2 + s3*r_theta**4

  # 5. Final projection
  u = fx * (ur + tx + tpx) + cx
  v = fy * (vr + ty + tpy) + cy

  return u, v

def unproject_no_jacobian(u, v, fx, fy, cx, cy, k0, k1, k2, k3, p0, p1, s0, s1, s2, s3, 
                          num_iterations=10, tolerance=1e-6):
  """
  Unprojects a 2D pixel coordinate (u, v) to a 3D ray 
  using the FisheyeRadTanThinPrism model
  """

  # 1. Initial estimate (using KB3 model)
  phi = math.atan2(v - cy, u - cx)
  r_theta_approx = math.sqrt((u - cx)**2 / fx**2 + (v - cy)**2 / fy**2)
  theta_initial = r_theta_approx  # Initial guess for theta

  # 2. Newton's method (simplified)
  def r(theta):
    return theta + k0*theta**3 + k1*theta**5 + k2*theta**7 + k3*theta**9

  def r_prime(theta):
    return 1 + 3*k0*theta**2 + 5*k1*theta**4 + 7*k2*theta**6 + 9*k3*theta**8

  def f(theta):
    ur = r(theta) * math.cos(phi)
    vr = r(theta) * math.sin(phi)
    tx = p0 * (2*ur**2 + r(theta)**2) + 2*p1*ur*vr
    ty = p1 * (2*vr**2 + r(theta)**2) + 2*p0*ur*vr
    tpx = s0*r(theta)**2 + s1*r(theta)**4
    tpy = s2*r(theta)**2 + s3*r(theta)**4
    return fx * (ur + tx + tpx) + cx - u, fy * (vr + ty + tpy) + cy - v

  theta = theta_initial
  for _ in range(num_iterations):
    f_val = f(theta)
    
    delta = - f_val[0] / r_prime(theta)  
    theta = theta + delta
    if abs(delta) < tolerance:
      break


  return theta, phi