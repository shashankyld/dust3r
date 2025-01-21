import os 
import config as cfg
from dust3r.model import AsymmetricCroCo3DStereo 
import logging 
from dust3r.demo import get_reconstructed_scene
import matplotlib.pyplot as plt
import open3d as o3d

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the output format
)

device = cfg.DEVICE 
weights_path = cfg.MODEL_PATH   
model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
img_size = cfg.IMAGE_SIZE

# Log the device, and model path
logging.info(f"Device: {device}")
logging.info(f"Model Path: {weights_path}")

# Output directory 
output_dir = cfg.OUTPUT_DIR 
# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)


logging.info(f"Output Directory: {output_dir}")

# Data directory
data_dir = cfg.DATA_DIR
logging.info(f"Data Directory: {data_dir}")
filelist = os.listdir(data_dir) #return full path
filelist = [os.path.join(data_dir, x) for x in filelist]
logging.info(f"Filelist: {filelist}")


# Get the reconstructed scene
scene, outfile, imgs = get_reconstructed_scene(
                            output_dir, model, device, cfg.SILENT, img_size, filelist, cfg.SCHEDULE,
                            cfg.NITER, cfg.MIN_CONF_THR, cfg.AS_POINTCLOUD, cfg.MASK_SKY,
                            cfg.CLEAN_DEPTH, cfg.TRANSPARENT_CAMS, cfg.CAM_SIZE, cfg.SCENEGRAPH_TYPE,
                            cfg.WINSIZE, cfg.REFID)

# Output is .glb file

print("images shape", len(imgs))
print("First set of images shape", imgs[0].shape, imgs[1].shape, imgs[2].shape, imgs[3].shape, imgs[4].shape, imgs[5].shape)
num_images = int(len(imgs)/3)
for i in range(num_images):
    ## Visualize the image, depth and confidence maps  at 3n-2, 3n-2, 3n indices
    img = imgs[3*i]
    depth = imgs[3*i+1]
    confidence = imgs[3*i+2]
    print(f"Image {i}")
    print(f"Image shape: {img.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print("\n")
    # Visualize the image, depth and confidence maps in a single plot 
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[1].imshow(depth)
    ax[1].set_title("Depth")
    ax[2].imshow(confidence)
    ax[2].set_title("Confidence")
    plt.show()

