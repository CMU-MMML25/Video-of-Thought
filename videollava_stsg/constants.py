CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image token constants (from original VideoLLaVA)
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video token constants (from original VideoLLaVA)
# Important: We keep the existing VideoLLaVA tokens to maintain compatibility
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<im_patch>"  # Using VideoLLaVA's version for compatibility 
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"
VIDEO_TOKEN_INDEX = -300  # Using a different index from IMAGE_TOKEN_INDEX to avoid conflicts

# Scene Graph token constants (new for STSG)
SG_TOKEN_INDEX = -400  # Using a different index to avoid conflicts with both image and video
DEFAULT_SG_TOKEN = "<scene_graph>"  # Token that will be added to tokenizer
DEFAULT_SG_PATCH_TOKEN = "<sg_patch>"
DEFAULT_SG_START_TOKEN = "<sg_start>"
DEFAULT_SG_END_TOKEN = "<sg_end>"
SG_PLACEHOLDER = "<sg-placeholder>"

# Max lengths
MAX_IMAGE_LENGTH = 16
# Using VideoLLaVA's setting for now, but this could be increased for multi-frame videos
MAX_VIDEO_LENGTH = 1  
MAX_SG_LENGTH = 16

PAD_LENGTH = 620