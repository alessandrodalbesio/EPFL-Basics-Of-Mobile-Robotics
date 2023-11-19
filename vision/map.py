# Append the path
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.marker import Marker

# Import generic modules
import numpy as np
import matplotlib.pyplot as plt
import utils.logger as logger
from utils.settings import w_px, h_px, w_cm, h_cm

# Import camera modules
import cv2
from vision.camera import Camera

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
    # Public attributes
    h = h_cm # Height of the environment [cm]
    w = w_cm # Width of the environment [cm]
    obstacles = [] # List of obstacles
    obstacles_original = [] # List of obstacles

    # Internal attributes
    h_px = h_px # Height of the environment [px]
    w_px = w_px # Width of the environment [px]
    camera = None

    def __init__(self, camera, numberOfObstacles=2, robotSize=25):
        """ Constructor of the Map class

        Args:
            camera (Camera): Camera object
            numberOfObstacles (int, optional): Number of obstacles in the environment (need to be tuned based on the environment). Defaults to 2.
            robotSize (int, optional): Robot size (need to be tuned based on the robot). Defaults to 75.
        """
        # Set
        self.camera = camera
        self.nObstacles = numberOfObstacles
        self.robotSize = robotSize

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

    def findObstacles(self):
        """ Find the obstacles in the environment. They can be accessed through the obstacles attribute. The obstacles are already enlarged by the radius of the robot. All the vertices are expressed in cm. """

        ## Find the obstacles in the image ##
        # Get a binary frame from the camera
        _, frameCut = self.camera.get_frame()
        # Convert the image to grayscale
        gray = cv2.cvtColor(frameCut, cv2.COLOR_BGR2GRAY)
        # Apply a threshold to the image
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) # [TODO] Value still to be tuned
        # Invert the mask
        mask = cv2.bitwise_not(mask)
        # Create a temporary binary image
        temp = np.zeros_like(mask)
        # Find the contours of the binary image
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Order the contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Get the first N_obstacles contours
        contours = contours[:self.nObstacles]
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
            # Invert the y axis
            points[:,1] = self.h_px - points[:,1]
            # Add the vertices to the list of obstacles
            self.obstacles_original.append(points)
            # Enlarge the polygon
            pol = pol.buffer(self.robotSize, cap_style=BufferCapStyle.square, join_style=BufferJoinStyle.mitre)
            # Add the vertices of the polygon to the list
            pols.append(pol)

        # Verify if any of the polygons intersect
        for i in range(len(pols)):
            for j in range(i+1,len(pols)):
                if pols[i].intersects(pols[j]):
                    # If they intersect, take the union of the two polygons
                    pols[i] = pols[i].union(pols[j])
                    # Remove the second polygon
                    pols.pop(j)
                    # Decrement the index
                    j -= 1
        
        # Convert the polygons to numpy arrays
        self.obstacles = []
        for pol in pols:
            # Get the vertices of the polygon
            points = np.array(pol.exterior.coords)
            # Invert the y axis
            points[:,1] = self.h_px - points[:,1]
            # Add the vertices to the list of obstacles
            self.obstacles.append(points)

        ## Convert the coordinates from pixels to cm ##
        # self.obstacles = self.convertToCm(self.obstacles)

    def getInitialFinalPoints(self):
        """ Get the initial and final points of the environment

        Returns:
            np.array([x,y]): Initial point [cm]
            np.array([x,y]): Final point [cm]
        """
        marker = Marker()
        # Define the region where the markers are
        self.markersRegion = marker.detect(self.camera, n_iterations=20)
        # Finf the marker with ID 4
        for key in self.markersRegion.keys():
            if key == 5:
                # Convert the points to the new reference system
                regionPoints = self.camera.originToFieldReference(self.markersRegion[key]["points"])
                # Compute the center of the region
                center = np.around(np.mean(regionPoints,axis=0))
                # Invert the y axis
                center[1] = self.h_px - center[1]
                # Set the initial point
                initialPoint = center
            if key == 4:
                # Convert the points to the new reference system
                regionPoints = self.camera.originToFieldReference(self.markersRegion[key]["points"])
                # Compute the center of the region
                center = np.around(np.mean(regionPoints,axis=0))
                # Invert the y axis
                center[1] = self.h_px - center[1]
                # Set the final point
                finalPoint = center

        return initialPoint, finalPoint

    def cameraRobotSensing(self):
        """ Get the position and the otientation of the robot. The position and orientation is refreshed at a rate of 30Hz

        Returns:
            np.array([x,y]): Position of the robot [cm]
            np.array([x,y]): Orientation of the robot (unit vector)
        """ 
        marker = Marker()
        # Define the region where the markers are
        self.markersRegion = marker.detect(self.camera, n_iterations=4)
        position = np.zeros(2)
        orientation = 0
        # Iterate through the markers
        for key in self.markersRegion.keys():
            if key == 5:
                # Estimate position of the robot
                points = self.camera.originToFieldReference(self.markersRegion[key]["points"])
                # Compute the center of the region
                center = self.camera._invertYaxis([np.around(np.mean(points,axis=0))])[0]
                # Set the initial point
                position = center

                # Find the median in the segment that goes from the point p[2] to p[3]
                vector = points[1]-points[0]
                # Compute the angle of the vector with respect to the x axis
                orientation = np.arctan2(vector[1],vector[0])
                orientation = orientation if orientation > 0 else orientation + 2*np.pi
                
        return position, orientation
        

    def plot(self, initialPoint=None, finalPoint=None, path=None):
        """Plot the map and the obstacles"""
        
        # Create a figure with the same size as the map
        fig = plt.figure()

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
        # Don't add the axis
        plt.axis('off')
        plt.show()