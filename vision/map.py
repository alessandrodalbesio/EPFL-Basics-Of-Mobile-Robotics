import numpy as np
import matplotlib.pyplot as plt

class Map:
    """ Map of the environment 
    
    Attributes:
        h (float): Height of the environment [cm]
        w (float): Width of the environment [cm]
        obstacles (list): List of obstacles. This is a numpy array of numpy arrays of shape (n,2) where n is the number of vertices of the obstacle given in clockwise order. 
                          The obstacle are already enlarged by the radius of the robot. All the vertices are expressed in cm.
    """
    def __init__(self,h,w,obstacles):
        # Dimensions of the environment
        self.h = h
        self.w = w

        # Obstacles of the environment
        self.obstacles = obstacles

    def plot(self, initialPoint=None, finalPoint=None, path=None):
        """Plot the map and the obstacles"""
        
        # Create a figure
        fig = plt.figure()

        # Plot a geometric shape with the coordinates of the obstacles
        for obstacle in self.obstacles:
            plot_points = obstacle.copy() # Copy the obstacle to avoid modifying the original one
            # Add the last point to close the shape
            plot_points = np.vstack((plot_points,plot_points[0]))
            # Plot the shape
            plt.fill(plot_points[:,0],plot_points[:,1],color='black')

        # Plot the path
        if path is not None:
            path = np.array(path)
            plt.plot(path[:,0],path[:,1],marker='o',color='blue')

        # Plot the initial and final points if they are given
        if initialPoint is not None and finalPoint is not None:
            plt.plot(initialPoint[0],initialPoint[1],marker='8',color='green', markersize=10)
            plt.plot(finalPoint[0],finalPoint[1],marker='X',color='red', markersize=20)

        # Set the limits of the plot
        plt.axis('equal')
        plt.xlim(0,self.w)
        plt.ylim(0,self.h)
        plt.show()


def createMap():
    """Create the map of the environment

    Returns:
        Map: Get a map object containing the information about the env size and the obstacles 
    """
    h = 100
    w = 100
    obstacles = []
    obstacles.append(np.array([[20,20],[0,30],[50,50]]))
    obstacles.append(np.array([[40,10],[80,40],[70,10]]))
    obstacles.append(np.array([[0,60],[60,60],[20,80]]))
    obstacles.append(np.array([[100,60],[100,100],[80,70],[80,60]]))
    return Map(h,w,obstacles)

def getInitialFinalPoints():
    """Get the initial and final points of the environment

    Returns:
        np.array([x,y]): Initial point [cm]
        np.array([x,y]): Final point [cm]
    """
    return np.array([5,5]),np.array([40,90])

def cameraRobotSensing():
    """Get the position and the otientation of the robot. The position and orientation is refreshed at a rate of 30Hz

    Returns:
        np.array([x,y]): Position of the robot [cm]
        np.array([x,y]): Orientation of the robot (unit vector)
    """
    return np.array([0,0]), np.array([0,1])