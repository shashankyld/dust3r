# Config for Aria - Dust3r project

DEVICE = "cuda"

# TRAINING
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 100

# DATA
# DATA_DIR = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/Priliminary tests/oneimage"
DATA_DIR = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/Priliminary tests/test_rectified_upright"
# MODEL
MODEL_NAME = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
MODEL_PATH = f"checkpoints/{MODEL_NAME}.pth"

# INFERENCE
IMAGE_SIZE = 512 

# OUTPUT
OUTPUT_DIR = "temp_test/"

# CAMERA PARAMETERS 
CAMERA_TYPE = "aria_slam" # pinhole, aria_rgb

# ---  Parameters for get_reconstructed_scene ---

SILENT = False
SCHEDULE = 'linear'
NITER = 300
MIN_CONF_THR = 3
AS_POINTCLOUD = False
MASK_SKY = False
CLEAN_DEPTH = True
TRANSPARENT_CAMS = False
CAM_SIZE = 0.05
SCENEGRAPH_TYPE = 'complete'
WINSIZE = 1  # Only used if SCENEGRAPH_TYPE is 'swin'
REFID = 0    # Only used if SCENEGRAPH_TYPE is 'oneref'


# Known poses
KNOWN_POSES = [[[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]], 
                [[0.99339134, -0.05150564,  0.10257099, -0.00428072],
                [ 0.10364322,  0.78649219, -0.60884162, -0.01184173],
                [-0.04931251,  0.61544878,  0.78663275, -0.00511398],
                [ 0.0, 0.0, 0.0, 1.0]]]
