# Debug settings
DISPLAY_DEBUG_MESSAGES = True

# Marker settings
IDS_CORNER_MARKERS = [0,1,2,3]
ID_GOAL_MARKER = 4
ID_ROBOT_MARKER = 5

# Camera settings
CAMERA_ID = 1
ITERATIONS_MAP_CREATION = 50
ITERATIONS_REAL_TIME_DETECTION = 4

# Environment settings
w_cm = 29.7*4
h_cm = 21*3
w_px = 640
h_px = round(w_px * h_cm / w_cm)
NUMBER_OF_OBSTACLES = 2
ROBOT_SIZE = 50

# Global navigation settings
DIST_FROM_GOAL_THRESH = 5
ANGLE_THRESH = 0.2
NOMINAL_SPEED = 75
K_ANGLE = 50
K_TRAJ = 300

# Local navigation settings
DIST_THRESH_LOCAL = 100
NOMINAL_SPEED_LOCAL = 80
SPEED_GAIN_LOCAL = [6, 4, -2, -6, -8]
ANGLE_THRESH_LOCAL = 0.2
K_ANGLE_LOCAL = 50
K_OBSTACLE_LOCAL = 0.01
LOCAL_AVOIDANCE_DELAY = 0.5

# Kidnapping settings
KIDNAPPING_CONV_THRESH = 4