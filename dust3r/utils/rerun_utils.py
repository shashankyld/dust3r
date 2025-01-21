import rerun as rr
import numpy as np
import torch
import open3d as o3d

def visualize_in_rerun(optimizer):
    """
    Extracts necessary data from the PointCloudOptimizer and visualizes it using Rerun.

    Args:
        optimizer: The PointCloudOptimizer object containing the 3D data.
    """
    
    points_3d = optimizer.get_pts3d()
    
    points_flattened = []
    for pts, (h, w) in zip(points_3d, optimizer.imshapes):
        points_flattened.extend(pts[:h * w].reshape(-1, 3).cpu().numpy())
    
    poses = optimizer.get_im_poses()
    
    translations = []
    rotations = []
    
    for pose in poses:
        translation = pose[:3, 3].cpu().detach().numpy()
        rotation = pose[:3, :3].cpu().detach().numpy()  # Added detach()
        
        translations.append(translation)
        rotations.append(rotation)
    
    rr.init("PointCloudOptimizer")
    rr.log("point_cloud", rr.points3d(points_flattened, color=(0, 255, 0), radius=0.01))
    
    for i, (translation, rotation) in enumerate(zip(translations, rotations)):
        # Ensure rotation format is correct for rr.pose3d()
        rr.log(f"camera_pose_{i}", rr.pose3d(translation, rotation))  
    
    print(f"Visualized {len(points_flattened)} points and {len(translations)} camera poses in Rerun.")



