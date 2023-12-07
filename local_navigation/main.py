from utils.settings import *
from utils.tools import *
from time import *

class Local:
    def __init__(self):
        self.active_avoidance = False
        self.last_obstacle_time = 0
    
    def local_obstacle(self, prox_horizontal):
        #test for local obstacles and update status
        if(max(prox_horizontal) > DIST_THRESH_LOCAL):
            self.active_avoidance = True
            
        #leave local avoidance if no obstacle detected for a certain time
        elif(self.active_avoidance):
            if(self.last_obstacle_time == 0):
                self.last_obstacle_time = time()
            elif(time() - self.last_obstacle_time > LOCAL_AVOIDANCE_DELAY):
                self.last_obstacle_time = 0
                self.active_avoidance = False       

        return self.active_avoidance
    
    def local_controller(self, prox_horizontal, angle_current, angle_goal):
        #set rotation considering next point to reach
        current_angle_theta = -angle_diff_rel(angle_goal, angle_current)
        spLeft  = NOMINAL_SPEED_LOCAL - K_ANGLE_LOCAL * current_angle_theta
        spRight = NOMINAL_SPEED_LOCAL + K_ANGLE_LOCAL * current_angle_theta
        
        #adjust rotation considering local obstacle
        for i in range(5):
            spLeft += prox_horizontal[i] * SPEED_GAIN_LOCAL[i] * K_OBSTACLE_LOCAL
            spRight += prox_horizontal[i] * SPEED_GAIN_LOCAL[4 - i] * K_OBSTACLE_LOCAL
        
        #avoid beeing static  
        if (spLeft == 0) and (spRight == 0):
            spLeft = -2
            spRight = -2
        
        return spLeft, spRight