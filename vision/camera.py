# Append the path of the parent directory
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import generic modules
import numpy as np
from utils.settings import h_px, w_px,IDS_CORNER_MARKERS
from utils.logger import logger

# Import camera modules
import cv2

# Import marker modules
from vision.marker import Marker

# Class definition
class Camera:
    """ Class for the camera """

    ## Management methods ##
    def __init__(self):
        # Get the camera
        self.camera = cv2.VideoCapture(0)
        
        # Define h of the frame
        self.h = h_px
        self.w = w_px

        # Estimate the region of interest
        self.fieldArea = None
        self.detectedMarkers = None
        self.fieldAreaEstimation()

        # Define some parameters for real time display
        self._obstacles = None
        self._startPosition = None
        self._goalPosition = None
        self._robotEstimatedPosition = None
        self._robotEstimatedOrientation = None

        # Log
        logger.info("Camera initialized")


    def fieldAreaEstimation(self):
        """ Method that estimate based on the corner markers the region of the fieldArea """
        # Create the marker object
        marker = Marker()
        # Define the region where the markers are
        self.markersRegion = marker.detect(self,n_iterations=250)
        # Remove the markers that are not used for corner detection
        self.markersRegion = {key: self.markersRegion[key] for key in self.markersRegion.keys() if key in IDS_CORNER_MARKERS}
        # Get only the origin of the markers
        self.fieldArea = [corner["points"][0] for corner in self.markersRegion.values() if corner["num_samples"] > 0]
        # Define the target points
        target_points = [[self.w,self.h],[0,self.h],[0,0],[self.w,0]]
        # Define the transformation matrix
        self.matrix = cv2.getPerspectiveTransform(np.float32(self.fieldArea),np.float32(target_points))
        # Define the region of the markers in the new reference system (this is used to remove most of the markers from the image in the binary frame computation)
        self.detectedMarkers = {}
        for key in self.markersRegion.keys():
            self.detectedMarkers[key] = self._originToCutFrame(self.markersRegion[key]["points"])
        
        # Log
        logger.info("Field area estimated")

    def release(self):
        """ Clear the camera """
        self.camera.release()
        cv2.destroyAllWindows()

        # Log
        logger.info("Camera released")

    def calibration(self):
        """ Method to calibrate the camera

        Raises:
            NotImplementedError: This method is not implemented yet [TODO]
        """
        raise NotImplementedError

    ## Methods for real time display ##
    # Obstacles
    @property
    def obstacles(self):
        # Invert y axis to make it equal to the system reference
        if self._obstacles is not None:
            for i in range(len(self._obstacles)):
                for j in range(len(self._obstacles[i])):
                    self._obstacles[i][j][1] = self.h - self._obstacles[i][j][1]

        # Return the obstacles
        return self._obstacles
    
    @obstacles.setter
    def obstacles(self,value):
        """ Set the obstacles

        Args:
            value (list(np.array((n,2)))): Obstacles (each obstacle is a list of points)
        """
        # Invert y axis to make it compatible with opencv
        for i in range(len(value)):
            value[i] = self._invertYaxis(value[i])
        
        # Set the obstacles
        self._obstacles = value

    # Start position
    @property
    def startPosition(self):
        # Invert y axis
        if self._startPosition is not None:
            self._startPosition[1] = self.h - self._startPosition[1]
        
        # Return the start position
        return self._startPosition
    
    @startPosition.setter
    def startPosition(self,value):
        """ Set the start position

        Args:
            value (np.array(2,)): Start position
        """
        # Invert the y axis to make it compatible with opencv
        value = self._invertYaxis([value])[0]

        # Set the start position
        self._startPosition = value
    
    # Goal position
    @property
    def goalPosition(self):
        # Invert y axis
        if self._goalPosition is not None:
            self._goalPosition[1] = self.h - self._goalPosition[1]

        # Return the goal position
        return self._goalPosition
    
    @goalPosition.setter
    def goalPosition(self,value):
        """ Set the goal position

        Args:
            value (np.array((2,))): Goal position
        """
        # Invert y axis to make it compatible with opencv
        value = self._invertYaxis([value])[0]

        # Set the goal position
        self._goalPosition = value
    
    # Estimated position
    @property
    def robotEstimatedPosition(self):
        # Invert y axis
        if self._robotEstimatedPosition is not None:
            self._robotEstimatedPosition[1] = self.h - self._robotEstimatedPosition[1]

        # Return the estimated position
        return self._robotEstimatedPosition
    
    @robotEstimatedPosition.setter
    def robotEstimatedPosition(self,value):
        """ Set the estimated position

        Args:
            value (np.array((2,))): Estimated position
        """
        # Invert y axis to make it compatible with opencv
        value = self._invertYaxis([value])[0]

        # Set the estimated position
        self._robotEstimatedPosition = value

    # Estimated orientation
    @property
    def robotEstimatedOrientation(self):
        # Return the estimated Orientation
        return self._robotEstimatedOrientation
    
    @robotEstimatedOrientation.setter
    def robotEstimatedOrientation(self,value):
        """ Set the estimated Orientation

        Args:
            value (np.array((2,))): Estimated Orientation
        """
        # Set the estimated Orientation
        self._robotEstimatedOrientation = value

    # Display method
    def display(self):
        """ Display the frame in real time. Run this method in a while loop to display the frame in real time. If the user press q (and this method returns True), the loop should be stopped.

        Returns:
            bool: True if the user pressed q, False otherwise
        """
        # Get the frame 
        _, frameCut = self.get_frame()

        # Display the obstacles
        if self._obstacles is not None:
            for obstacle in self._obstacles:
                for p in obstacle:
                    p = tuple(p.astype(int))
                    cv2.circle(frameCut,p,5,(0,0,255),5)

        # Display the start position
        if self._startPosition is not None:
            p = tuple(self._startPosition.astype(int))
            cv2.circle(frameCut,p,5,(0,255,0),5)
            cv2.putText(frameCut,"Start position",p,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # Display the goal position
        if self._goalPosition is not None:
            p = tuple(self._goalPosition.astype(int))
            cv2.circle(frameCut,p,5,(0,255,0),5)
            cv2.putText(frameCut,"Goal position",p,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # Display the estimated position
        if self._robotEstimatedPosition is not None and self._robotEstimatedOrientation is not None:
            p = tuple(self._robotEstimatedPosition.astype(int))
            cv2.circle(frameCut,p,5,(255,0,0),5)

            o = self._robotEstimatedOrientation
            o = (p[0]+int(50*o[0]),p[1]+int(50*o[1]))
            cv2.arrowedLine(frameCut,p,o,(255,0,0),5)

        # Display the frame
        cv2.imshow("Frame cut",frameCut)
        return cv2.waitKey(1) & 0xFF == ord('q')

    ## Frame management methods ##

    def get_frame(self):
        """ Get the frame from the camera

        Returns:
            np.array((h,w,3)): Frame
            np.array((h,w,3)): Cutted frame
        """
        # Get the frame
        _, frame = self.camera.read()
        # Cut the frame to get only the fieldArea
        frameCut = None
        if self.fieldArea is not None:
            frameCut = frame.copy()
            frameCut = cv2.warpPerspective(frameCut,self.matrix,(self.w,self.h))
        # Return the frame
        return frame, frameCut

    ## Utils methods ##
    def _invertYaxis(self,points):
        """ Invert the y axis of the points

        Args:
            points (np.array((2,n))): Points to invert

        Returns:
            np.array((2,n)): Inverted points
        """
        # Invert y axis
        for i in range(len(points)):
            points[i][1] = self.h - points[i][1]

        # Return the points
        return points

    def _originToCutFrame(self,points):
        """ Convert the points into fieldArea from the origin reference system to the destination reference system

        Args:
            points (np.array((2,n))): Points to convert

        Returns:
            np.array((2,n)): Converted points
        """
        # Convert the points
        points = cv2.perspectiveTransform(np.float32([points]),self.matrix)[0]

        # Return the points
        return points
