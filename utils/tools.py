import numpy as np

def angle_diff_rel(angle_goal, angle_current):
    """Function which gives the delta angle in range [-pi;pi].
    
    Args:
        angle_goal: The angle we want to be at 
        angle_current: The angle we are currently at
    
    Returns:
        angle_diff: The difference between the two angles in range [-pi;pi]
    """
    if 0 < angle_goal <= np.pi:
        if angle_goal <= angle_current <= angle_goal + np.pi:
            angle_diff = angle_current - angle_goal
        else:
            if angle_goal + np.pi <= angle_current <= 2 * np.pi:
                angle_diff = angle_current - angle_goal - 2 * np.pi
            else:
                angle_diff = angle_current - angle_goal
    else:
        if angle_goal - np.pi <= angle_current <= angle_goal:
            angle_diff = angle_current - angle_goal
        else:
            if 0 <= angle_current <= angle_goal- np.pi:
                angle_diff = angle_current - angle_goal + 2 * np.pi
            else:
                angle_diff = angle_current - angle_goal
    return angle_diff