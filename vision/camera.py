##### Modules importing #####
# Append the path of the parent directory
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import generic modules
import numpy as np
from utils.settings import h_px, w_px,IDS_CORNER_MARKERS, CAMERA_ID, ITERATIONS_MAP_CREATION
from utils.exceptions import NotEnoughMarkers, DoNotAccessError

# Import camera modules
import cv2

# Import marker modules
from vision.marker import Marker

##### Class definition #####
class Camera:
    """ This class implement all the method that are used to manage the camera. 
    Don't use the getter methods to compute the position of the robot, obstacles, etc. but use the Map class instead.

    Raises:
        NotEnoughMarkers: If not all the markers are detected
        NotImplementedError: If the method is not implemented yet

    Returns:
        Camera: Camera object
    """

    ## Management methods ##
    def __init__(self,frame_path=None):
        # Get the camera
        self.camera = None
        self.camera_frame = None
        if frame_path == None:
            self.camera = cv2.VideoCapture(CAMERA_ID)
        else:
            self.camera_frame = cv2.imread(frame_path)

        # Define h of the frame
        self.h_px = h_px
        self.w_px = w_px

        # Estimate the region of interest
        self.fieldArea = None
        self.fieldAreaEstimation()

        # Define some parameters for real time display
        self._obstacles = None
        self._startPosition = None
        self._goalPosition = None
        self._robotEstimatedPosition = None
        self._robotEstimatedOrientation = None


    def fieldAreaEstimation(self):
        """ Method that estimate based on the corner markers the region of the fieldArea """
        # Create the marker object
        marker = Marker()

        ## Markers detection ##
        # Define the region where the markers are
        if self.camera is not None:
            self.markersRegion = marker.detect(cam=self,n_iterations=ITERATIONS_MAP_CREATION)
        else: 
            self.markersRegion = marker.detect(cam=self,n_iterations=1)
        # Remove the markers that are not used for corner detection
        self.markersRegion = {key: self.markersRegion[key] for key in self.markersRegion.keys() if key in IDS_CORNER_MARKERS}
        # Check that all the markers are detected
        if len(self.markersRegion) != 4:
            raise NotEnoughMarkers("Not all the markers have been detected. Please check that all the markers are visible and that the camera is not too close to the fieldArea.")
        
        ## Definition of the new reference system ##
        # Get only the origin of the markers
        self.fieldArea = [corner["points"][0] for corner in self.markersRegion.values() if corner["num_samples"] > 0]
        # Define the target points
        target_points = [[0,self.h_px],[self.w_px,self.h_px],[self.w_px,0],[0,0]]
        # Define the transformation matrix
        self.matrix = cv2.getPerspectiveTransform(np.float32(self.fieldArea),np.float32(target_points))

    def release(self):
        """ Clear the camera """
        if self.camera == None:
            return
        ## Release the camera ##
        self.camera.release()
        cv2.destroyAllWindows()

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
        raise DoNotAccessError
    
    @obstacles.setter
    def obstacles(self,value):
        """ Set the obstacles

        Args:
            value (list(np.array((n,2)))): Obstacles (each obstacle is a list of points)
        """
        # Invert y axis to make it compatible with opencv visualization
        if value is not None:
            for obstacle in value:
                obstacle = self._invertYaxis(obstacle)
                
        # Set the obstacles
        self._obstacles = value

    # Start position
    @property
    def startPosition(self):
        raise DoNotAccessError
    
    @startPosition.setter
    def startPosition(self,value):
        """ Set the start position

        Args:
            value (np.array((2,))): Start position
        """
        # Invert the y axis to make it compatible with opencv
        value = self._invertYaxis([value])[0]

        # Set the start position
        self._startPosition = value
    
    # Goal position
    @property
    def goalPosition(self):
        raise DoNotAccessError
    
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
        raise DoNotAccessError
    
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
        raise DoNotAccessError
    
    @robotEstimatedOrientation.setter
    def robotEstimatedOrientation(self,value):
        """ Set the estimated Orientation

        Args:
            value (np.array((2,))): Estimated Orientation
        """
        # Invert the y sign (because the y axis is inverted)
        # value[1] = -value[1]

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
                    cv2.circle(frameCut,p,5,(0,0,255),-1)

        # Display the start position
        if self._startPosition is not None:
            p = tuple(self._startPosition.astype(int))
            cv2.circle(frameCut,p,5,(0,255,0),-1)
            cv2.putText(frameCut,"Start",p,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # Display the goal position
        if self._goalPosition is not None:
            p = tuple(self._goalPosition.astype(int))
            cv2.circle(frameCut,p,5,(0,255,0),-1)
            cv2.putText(frameCut,"Goal",p,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # Display the estimated position
        if self._robotEstimatedPosition is not None and self._robotEstimatedOrientation is not None:
            # Display the position
            p = tuple(self._robotEstimatedPosition.astype(int))
            cv2.circle(frameCut,p,5,(255,0,0),-1)

            # Display the orientation
            o = self._robotEstimatedOrientation
            o = tuple(np.array([p[0] + 50*np.cos(o), p[1] + 50*np.sin(o)]).astype(int))
            cv2.arrowedLine(frameCut,p,o,(255,0,0),2)


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
        if self.camera_frame is not None:
            frame = self.camera_frame
        else:
            _, frame = self.camera.read()
        
        # Cut the frame to get only the fieldArea
        frameCut = None
        if self.fieldArea is not None:
            frameCut = frame.copy()
            frameCut = cv2.warpPerspective(frameCut,self.matrix,(self.w_px,self.h_px))
        
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
            points[i][1] = self.h_px - points[i][1]

        # Return the points
        return points

    def originToFieldReference(self,points):
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