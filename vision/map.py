# Append the path
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.marker import Marker

# Import generic modules
import numpy as np
import matplotlib.pyplot as plt
from utils.settings import *

# Import camera modules
import cv2

# Import geometry modules
from shapely import Polygon, BufferCapStyle, BufferJoinStyle

# Map class definition
class Map:
    """ Map of the environment 
    
    Public Attributes:
        h (float): Height of the environment [cm]
        w (float): Width of the environment [cm]
        obstacles (list): List of obstacles. This is a numpy array of numpy arrays of shape (n,2) where n is the number of vertices of the obstacle given in clockwise order. 
                          The obstacle are already enlarged by the radius of the robot. All the vertices are expressed in cm.
    Internal Attributes:
        h_px (int): Height of the environment [px]
        w_px (int): Width of the environment [px]
        camera (Camera): Camera object
    """
    def __init__(self, camera, h_px = h_px, h_cm = h_cm, w_px = w_px, w_cm = w_cm, number_of_obstacles = NUMBER_OF_OBSTACLES, robot_size = ROBOT_SIZE):
        """ Constructor of the Map class

        Args:
            camera (Camera): Camera object
            numberOfObstacles (int, optional): Number of obstacles in the environment (need to be tuned based on the environment). Defaults to 2.
            robotSize (int, optional): Robot size (need to be tuned based on the robot). Defaults to 75.
        """
        # Define some attributes
        self.camera = camera
        self.obstacles = []
        self.obstacles_original = []

        # Define the size of the environment
        self.number_obstacles = number_of_obstacles
        self.robotSize = robot_size
        self.h_px = h_px
        self.w_px = w_px
        self.h = h_cm
        self.w = w_cm


    def convertToPx(self, points):
        """ Convert the coordinates from cm to pixels

        Args:
            points (np.array((n,2))): Points in cm

        Returns:
            np.array((n,2)): Points in pixels
        """
        pxPoints = []
        for p in points:
            pxPoints.append([int(p[0]*self.w_px/self.w),int(p[1]*self.h_px/self.h)])
        return np.array(pxPoints)

    def convertToCm(self, points):
        """ Convert the coordinates from pixels to cm

        Args:
            points (np.array((n,2))): Points in pixels

        Returns:
            np.array((n,2)): Points in cm
        """
        cmPoints = []
        for p in points:
            cmPoints.append([p[0]*self.w/self.w_px,p[1]*self.h/self.h_px])
        return np.array(cmPoints)

    def findObstacles(self, number_of_obstacles=None, robot_size=None):
        """ Find the obstacles in the environment. They can be accessed through the obstacles attribute. The obstacles are already enlarged by the radius of the robot. All the vertices are expressed in cm. """
        
        if number_of_obstacles is None:
            number_of_obstacles = self.number_obstacles
        if robot_size is None:
            robot_size = self.robotSize

        ## Find the obstacles in the image ##
        # Get a binary frame from the camera
        _, frameCut = self.camera.get_frame()
        # Convert the image to grayscale
        gray = cv2.cvtColor(frameCut, cv2.COLOR_BGR2GRAY)
        # Apply a threshold to the image
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # Invert the mask
        mask = cv2.bitwise_not(mask)
        # Create a temporary binary image
        temp = np.zeros_like(mask)
        # Find the contours of the binary image
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Order the contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Get the first N_obstacles contours
        contours = contours[:number_of_obstacles]
        # Plot the contours on the binary image
        cv2.drawContours(temp,contours,-1,(255,255,255),-1)

        ## Smooth the obstacles in the image ##
        # Find convex hulls to smooth the obstacles in the image
        hulls = [cv2.convexHull(c) for c in contours]
        # Plot the convex hulls on the binary image
        cv2.drawContours(temp,hulls,-1,(255,255,255),-1)
        # Find the contours of the convex hulls
        contours, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Approximate the contours with a polygon
        contours = [cv2.approxPolyDP(c, 0.01*cv2.arcLength(c,True), True) for c in contours]
    
        ## Make the obstacles compatible with the path planning algorithm and enlarge them ##
        pols = []
        for c in contours:
            # Create a shapely polygon from the contour
            pol = Polygon([p[0] for p in c])
            points = np.array(pol.exterior.coords)
            # Add the vertices to the list of obstacles
            self.obstacles_original.append(points)
            # Enlarge the polygon
            pol = pol.buffer(robot_size, cap_style=BufferCapStyle.square, join_style=BufferJoinStyle.mitre)
            # Add the vertices of the polygon to the list
            pols.append(pol)
        
        # Convert the polygons to numpy arrays
        self.obstacles = []
        for pol in pols:
            # Get the vertices of the polygon
            points = np.array(pol.exterior.coords)
            # Add the vertices to the list of obstacles
            self.obstacles.append(points)
        
        # Iterate through the obstacles and remove the points that are outside the image
        for i in range(len(self.obstacles)):
            # Get the vertices of the obstacle
            points = self.obstacles[i]
            # Invert the y axis
            points[:,1] = self.h_px - points[:,1]
            # Update the obstacle
            self.obstacles[i] = points        
        

    def getInitialFinalData(self):
        """ Get the initial and final points of the environment

        Returns:
            np.array([x,y]): Initial point [px]
            np.array([x,y]): Final point [px]
        """
        marker = Marker()
        # Define the region where the markers are
        self.markersRegion = marker.detect(self.camera, n_iterations=ITERATIONS_MAP_CREATION)
        # Finf the marker with ID 4
        for key in self.markersRegion.keys():
            if key == 5:
                # Convert the points to the new reference system
                regionPoints = self.camera.originToFieldReference(self.markersRegion[key]["points"])
                # Compute the center of the region
                center = np.around(np.mean(regionPoints,axis=0))
                # Set the initial point
                initialPoint = center
                # Get the initial orientation
                initialOrientation = self.getRobotOrientation(regionPoints[0],regionPoints[1])
            if key == 4:
                # Convert the points to the new reference system
                regionPoints = self.camera.originToFieldReference(self.markersRegion[key]["points"])
                # Compute the center of the region
                center = np.around(np.mean(regionPoints,axis=0))
                # Set the final point
                finalPoint = center

        return initialPoint, initialOrientation, finalPoint       

    def getRobotOrientation(self, initialPoint, finalPoint):
        """ Get the orientation of the robot

        Args:
            initialPoint (np.array([x,y])): Initial point [px]
            finalPoint (np.array([x,y])): Final point [px]

        Returns:
            int: Orientation of the robot (in radians)
        """
        # Compute the angle of the vector with respect to the x axis
        vector = finalPoint-initialPoint
        orientation = np.arctan2(vector[1],vector[0])
        orientation = (orientation + 2*np.pi)%(2*np.pi)
        return orientation


    def cameraRobotSensing(self, isInCm=False):
        """ Get the position and the otientation of the robot. The position and orientation is refreshed at a rate of 30Hz

        Returns:
            np.array([x,y]): Position of the robot [cm]
            int: Orientation of the robot (in radians)
        """ 
        marker = Marker()
        # Define the region where the markers are
        self.markersRegion = marker.detect(self.camera, n_iterations=ITERATIONS_REAL_TIME_DETECTION)
        # Return None if no markers is detected in the map
        if self.markersRegion is None:
            return None,None
        
        position = None
        orientation = None
        # Iterate through the markers
        for key in self.markersRegion.keys():
            if key == ID_ROBOT_MARKER:
                # Estimate position of the robot
                points = self.camera.originToFieldReference(self.markersRegion[key]["points"])
                # Compute the center of the region
                center = np.around(np.mean(points,axis=0))
                # Set the initial point
                position = center

                # Compute the angle of the vector with respect to the x axis
                orientation = self.getRobotOrientation(points[0],points[1])
        
        if isInCm:
            position = self.convertToCm([position])[0]

        return position, orientation
        

    def plot(self, initialPoint=None, finalPoint=None):
        """Plot the map and the obstacles"""
        
        # Create a figure with the same size as the map
        plt.figure()

        # Plot a geometric shape with the coordinates of the obstacles
        for obstacle in self.obstacles:
            plot_points = obstacle.copy() # Copy the obstacle to avoid modifying the original one
            # Add the last point to close the shape
            plot_points = np.vstack((plot_points,plot_points[0]))
            # Plot the shape
            plt.fill(plot_points[:,0],plot_points[:,1],color='black')

        for obstacle in self.obstacles_original:
            plot_points = obstacle.copy()
            # Add the last point to close the shape
            plot_points = np.vstack((plot_points,plot_points[0]))
            # Plot the shape
            plt.plot(plot_points[:,0],plot_points[:,1],color='red')

        # Plot the initial and final points if they are given
        if initialPoint is not None and finalPoint is not None:
            plt.plot(initialPoint[0],initialPoint[1],marker='8',color='green', markersize=10)
            plt.plot(finalPoint[0],finalPoint[1],marker='X',color='red', markersize=20)

        

        # Set the limits of the plot
        plt.axis('equal')
        # Don't add the axis
        plt.axis('off')
        plt.show()