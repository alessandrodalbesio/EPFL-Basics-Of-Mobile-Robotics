from utils.settings import *

class Local:
    def __init__(self):
        self.active_avoidance = False
        self.current_angle_theta = 0
    
    def local_obstacle(self, prox_horizontal, angle_current, angle_goal):
        #test for local obstacles and update status
        obstacle_detected = max(prox_horizontal) > DIST_THRESH_LOCAL
        if(obstacle_detected):
            self.active_avoidance = True
        #leave local avoidance if in the good direction
        elif(self.active_avoidance):
            self.__calculate_angle_teta(angle_current, angle_goal)
            self.active_avoidance = abs(self.current_angle_theta) >= ANGLE_THRESH_LOCAL
                
        return 
    
    def local_controller(self, prox_horizontal):
        #set speed considering next point to reach
        spLeft  = NOMINAL_SPEED_LOCAL - K_ANGLE_LOCAL * self.current_angle_theta
        spRight = NOMINAL_SPEED_LOCAL + K_ANGLE_LOCAL * self.current_angle_theta
        
        #adjust speed considering local obstacle
        for i in range(5):
            spLeft += prox_horizontal[i] * SPEED_GAIN_LOCAL[i] * K_OBSTACLE_LOCAL
            spRight += prox_horizontal[i] * SPEED_GAIN_LOCAL[4 - i] * K_OBSTACLE_LOCAL
        
        #avoid beeing static  
        if (spLeft == 0) and (spRight == 0):
            spLeft = -2
            spRight = -2
        
        return spLeft, spRight
    
    def __calculate_angle_teta(self, angle_current, angle_goal):
        #difference between current and wanted angle
        self.current_angle_theta = angle_current - angle_goal
        return