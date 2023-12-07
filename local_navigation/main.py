from utils.settings import *
from utils.tools import *
from time import *

class Local:
    """ Local controller class 
    
    Public Methods:
        local_obstacle (boolean): check if we should use the local controller
        local_controller (float, float): gives desired speed from local controller
    """
    
    def __init__(self):
        """Constructor of the Local class.
        """
        
        #Define some attributes
        self.__active_avoidance = False
        self.__last_obstacle_time = 0
    
    def local_obstacle(self, prox_horizontal):
        """Function which tests if we should use the local controller.
        
        Args:
            prox_horizontal (list): Datas of the horizontal proximity sensors. 
        
        Returns:
            active_avoidance (boolean): If we use the local controller.
        """
        
        #test for local obstacles and update avoidance status
        if(max(prox_horizontal) > DIST_THRESH_LOCAL):
            self.__active_avoidance = True
            
        #leave local avoidance if no obstacle detected for a certain time
        elif(self.__active_avoidance):
            if(self.__last_obstacle_time == 0):
                self.__last_obstacle_time = time()
            elif(time() - self.__last_obstacle_time > LOCAL_AVOIDANCE_DELAY):
                self.__last_obstacle_time = 0
                self.__active_avoidance = False       

        return self.__active_avoidance
    
    def local_controller(self, prox_horizontal, angle_current, angle_goal):
        """Controller which makes the robot avoid local obstacles while going to ne next goal.

        Args:
            prox_horizontal (list): Datas of the horizontal proximity sensors.
            angle_current (float): estimation of the current angle of the robot
            angle_goal (float): the angle the robot has to follow to reach next goal

        Returns:
            spLeft: desired speed of left wheel
            spRight: desired speed of right wheel
        """
        
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