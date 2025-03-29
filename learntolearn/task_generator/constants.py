import numpy as np

# object mean sizes based on HS-Pose - in meters and in the sapien object coordinate system
# to get this, just transform by the corresponding rotation matrix defined in TON
MEAN_SIZES = { 
    "laptop": [
        0.315, 
        0.358,
        0.248
    ],
    "keyboard": [
        0.185,
        0.500,
        0.040
    ],
    "mouse": [
        0.100,
        0.060,
        0.040
    ],
    "display": [
        0.194,
        0.612,
        0.439
    ],
    "remote": [
        0.023,
        0.055,
        0.150
    ],
    "bottle": [
        0.070,
        0.070,
        0.185
    ],
    "mug": [
        0.115,
        0.080,
        0.100
    ],
    "plant": [
        0.293,
        0.301,
        0.383
    ],
    "book": [
        0.180,
        0.259,
        0.048
    ],
    "ball": [
        0.200,
        0.200,
        0.200
    ],
    "chair": [
        0.500,
        0.500,
        1.100
    ],
    "car": [
        1.400,
        0.620,
        0.500
    ],
    "sofa": [
        1.250,
        0.500,
        0.600
    ],
}

# in meters
EXTRA_PADDING = {
    "laptop": [
        0,
        0,
        0
    ],
    "keyboard": [
        0,
        0,
        0.0035
    ],
    "mouse": [
        0,
        0,
        0
    ],
    "display": [
        0,
        0.0025,
        0
    ],
    "remote": [
        0,
        0,
        0
    ],
    "bottle": [
        0,
        0,
        0
    ],
    "mug": [
        0,
        0,
        0
    ],
    "plant": [
        0,
        0,
        0
    ],
    "book": [
        0,
        0,
        0
    ],
    "ball": [
        0,
        0,
        0
    ],
    "chair": [
        0,
        0,
        0
    ],
    "car": [
        0,
        0,
        0
    ],
    "sofa": [
        0,
        0,
        0
    ],
}

# based on YOLOv8
CATEGORY_IDS = {
    "car": 2,
    "laptop": 63,
    "bottle": 39,
    "chair": 56,
    "camera": 0, # not in YOLOv8
    "keyboard": 66,
    "mouse": 64,
    "display": 62, # as TV
    "remote": 65,
    "mug": 41, # as cup
    "plant": 58,
    "book": 73,
    "ball": 32, # as sports ball
}

# Camera configuration
WIDTH = 640
HEIGHT = 480
NEAR = 0.1
FAR = 5.0
FOVY = np.deg2rad(42.5)
DEPTH_SCALE = 12500.0

# Other configurations
NUM_POSES = np.random.randint(20, 40)
BBOX3D_PADDING = 0.10
SAMPLING_RADIUS_RANGE = (0.15, 0.45)

# for nerf training
SAMPLING_THETA_RANGE = (0, 2*np.pi)
SAMPLING_PHI_RANGE = (0, np.pi) 

## MODES:
# choose one
CHOOSE_ONE_RANDOM = True
THIRD_PARTY_ONLY = False

# for held-out test set
IS_TEST = True
NUM_TEST_POSES = 6