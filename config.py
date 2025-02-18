# Config for Aria - Dust3r project
import torch
import numpy as np

DEVICE = "cuda"

# TRAINING
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 100

# DATA
DATA_DIR = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/Priliminary tests/two_images"
# DATA_DIR = "/home/shashank/Documents/UniBonn/Sem5/aria-stereo-depth-completion/Priliminary tests/test_rectified_upright"
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


# Known poses - torch
# gt_relative_pose_rgb_l = [[ 0.99339134 -0.05150564  0.10257099 -0.00428072]
#  [ 0.10364322  0.78649219 -0.60884162 -0.01184173]
#  [-0.04931251  0.61544878  0.78663275 -0.00511398]
#  [ 0.          0.          0.          1.        ]]

# gt_relative_pose_r_l = [[ 0.99836471 -0.03532317  0.04494634  0.00567676]
#  [ 0.05208901  0.23820291 -0.96981757 -0.11215309]
#  [ 0.02355068  0.97057285  0.23965332 -0.08683892]
#  [ 0.          0.          0.          1.        ]]

# KNOWN_POSES = [
#     [
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ]
#     ,
#     [
#         [0.99339134, -0.05150564, 0.10257099, -0.00428072],
#         [0.10364322, 0.78649219, -0.60884162, -0.01184173],
#         [-0.04931251, 0.61544878, 0.78663275, -0.00511398],
#         [0.0, 0.0, 0.0, 1.0]
#     ], 
    
#     [
#         [0.99836471, -0.03532317, 0.04494634, 0.00567676],
#         [0.05208901, 0.23820291, -0.96981757, -0.11215309],
#         [0.02355068, 0.97057285, 0.23965332, -0.08683892],
#         [0.0, 0.0, 0.0, 1.0]
#     ]
# ]


# KNOWN_POSES = [
#     [
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ], 
#     [
#         [0.99836471, -0.03532317, 0.04494634, 0.00567676],
#         [0.05208901, 0.23820291, -0.96981757, -0.11215309],
#         [0.02355068, 0.97057285, 0.23965332, -0.08683892],
#         [0.0, 0.0, 0.0, 1.0]
#     ], 
#     [
#         [0.99836471, -0.03532317, 0.04494634, 0.00567676],
#         [0.05208901, 0.23820291, -0.96981757, -0.11215309],
#         [0.02355068, 0.97057285, 0.23965332, -0.08683892],
#         [0.0, 0.0, 0.0, 1.0]
#     ]
# ]

KNOWN_POSES = [
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], 
    [
        [0.99836471, -0.03532317, 0.04494634, 0.00567676],
        [0.05208901, 0.23820291, -0.96981757, -0.11215309],
        [0.02355068, 0.97057285, 0.23965332, -0.08683892],
        [0.0, 0.0, 0.0, 1.0]
    ], 
]

KNOWN_POSES = [
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], 
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ], 
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
]

KNOWN_POSES = torch.tensor(KNOWN_POSES, device=DEVICE)

# KNOWN_FOCALS = np.array([165, 165, 165]) # Focal length in pixels for slam -left, right, rgb assuming pinhole camera model 165, 165, 280
KNOWN_FOCALS = np.array([165, 165])
# Resolution of the slam-L image:  (480, 640)
# Resolution of the slam-R image:  (480, 640)
# Resolution of the rgb image:  (1408, 1408, 3)



# [f_u {f_v} c_u c_v [k_0: k_{numK-1}]  {p_0 p_1} {s_0 s_1 s_2 s_3}]
# 

camera_calibrations = {
    "camera-rgb": {
        "label": "camera-rgb",
        "model_name": "Fisheye624",
        "principal_point": [715.115, 716.715],
        "focal_length": [610.941, 610.941],
        "projection_params": [
            610.941, 715.115, 716.715, 0.406036, -0.489948, 0.174565, 
            1.13298, -1.70164, 0.651156, 0.000621147, 1.9322e-05, 
            -1.48553e-05, 0.000260123, -0.000658211, 3.7614e-05
        ],
        "image_size": [1408, 1408],
        "T_Device_Camera": {
            "translation": [-0.00428072, -0.0118417, -0.00511398],
            "quaternion": [0.32414, 0.0402123, 0.0410768, 0.944261]
        },
        "serial_number": "0450577b730301194401100000000000",
        "TimeOffsetSec_Device_Camera": 0
    },
    "camera-slam-left": {
        "label": "camera-slam-left",
        "model_name": "Fisheye624",
        "principal_point": [316.638, 237.025],
        "focal_length": [241.092, 241.092],
        "projection_params": [
            241.092, 316.638, 237.025, -0.0255612, 0.0984733, -0.0675009,
            0.00948425, 0.00233176, -0.000574631, 0.00114624, -0.00149885,
            -0.00123338, -0.000119675, 0.0022857, -5.6488e-05
        ],
        "image_size": [640, 480],
        "T_Device_Camera": {
            "translation": [0, -4.16334e-17, -1.04083e-17],
            "quaternion": [0, 0, 0, 1]
        },
        "serial_number": "0072510f1b0c07010700000826070001",
        "TimeOffsetSec_Device_Camera": 0
    },
    "camera-slam-right": {
        "label": "camera-slam-right",
        "model_name": "Fisheye624",
        "principal_point": [317.878, 238.205],
        "focal_length": [241.079, 241.079],
        "projection_params": [
            241.079, 317.878, 238.205, -0.0253498, 0.0975384, -0.0668994, 
            0.00968776, 0.00204258, -0.000503194, 0.00174441, -0.00104401,
            -0.00133432, -6.98614e-05, 0.00231518, 0.000196385
        ],
        "image_size": [640, 480],
        "T_Device_Camera": {
            "translation": [0.00567676, -0.112153, -0.0868389],
            "quaternion": [0.616545, 0.00679831, 0.0277746, 0.786801]
        },
        "serial_number": "0072510f1b0c07010700000819000001",
        "TimeOffsetSec_Device_Camera": 0
    }
}

# Add destination parameters for each camera
destination_params = {
    "camera-rgb": {
        'focal_length': [280, 280],
        'principal_point': [256,256],  # Center of destination image
        'image_size': [512, 512]  # Match original dimensions
    },
    "camera-slam-left": {
        'focal_length': [241.092, 241.092],  # Match source focal length
        'principal_point': [320, 240],  # Center of destination image
        'image_size': [640, 480]  # Standard dimensions
    },
    "camera-slam-right": {
        'focal_length': [241.092, 241.092],  # Match source focal length
        'principal_point': [320, 240],  # Center of destination image
        'image_size': [640, 480]  # Standard dimensions
    }
}