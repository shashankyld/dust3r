import os
import torch
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.viz import pts3d_to_trimesh, cat_meshes

def generate_and_save_pointcloud(image_files, model_weights, device, output_dir, image_size=512, niter=300, schedule="linear"):
    # Load images
    imgs = load_images(image_files, size=image_size, verbose=True)
    if len(imgs) == 1:
        imgs = [imgs[0], imgs[0].copy()]
        imgs[1]['idx'] = 1

    # Create pairs
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)

    # Load model
    model = torch.load(model_weights, map_location=device)

    # Run inference
    output = inference(pairs, model, device, batch_size=1, verbose=True)

    # Perform global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=True)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=0.01)

    # Get point cloud
    pts3d = to_numpy(scene.get_pts3d())
    masks = to_numpy(scene.get_masks())
    colors = to_numpy(scene.imgs)

    # Save as point cloud
    pts = pts3d[masks]
    col = colors[masks]
    pointcloud = pts3d_to_trimesh(colors=col, pts3d=pts3d, mask=masks)

    # Export
    outfile = os.path.join(output_dir, "pointcloud.ply")
    pointcloud.export(outfile)
    print(f"Point cloud saved to {outfile}")

# Example usage
image_files = ["/home/shashank/Downloads/tajmahal/dfsdfsdfsdf.jpeg"]  # Replace with actual paths
model_weights = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"  # Replace with actual model weight file
output_dir = "./output"
device = "cuda" if torch.cuda.is_available() else "cpu"

generate_and_save_pointcloud(image_files, model_weights, device, output_dir)
