##### Modules importing #####
# Append the path of the parent directory
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import generic modules
import numpy as np
from utils.settings import *

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
    def __init__(self,frame_path=None,h_cm=h_cm,w_cm=w_cm,h_px=h_px,w_px=w_px,save_states=False):
        # Get the camera
        self.camera = None
        self.camera_frame = None
        if frame_path == None:
            self.camera = cv2.VideoCapture(CAMERA_ID)
        else:
            self.camera_frame = cv2.imread(frame_path)

        # Define the sizes
        self.h_cm = h_cm
        self.w_cm = w_cm
        self.h_px = h_px
        self.w_px = w_px

        # Estimate the region of interest
        self.fieldArea = None
        self.fieldAreaEstimation()

        # Define some parameters for real time display
        self.startPosition = None
        self.goalPosition = None
        self.robotEstimatedPosition = None
        self.robotEstimatedOrientation = None
        self.robotMeasuredPosition = None
        self.robotMeasuredOrientation = None
        self.obstacles = None
        self.optimalPath = None
        self.robotEstimatedPositionHistory = []
        self.robotMeasuredPositionHistory = []
        self.save_states = save_states

    def fieldAreaEstimation(self):
        """ Method that estimate based on the corner markers the region of the fieldArea """
        # Create the marker object
        marker = Marker()

        ## Markers detection ##
        # Define the region where the markers are
        self.markersRegion = marker.detect(cam=self,n_iterations=ITERATIONS_MAP_CREATION if self.camera is not None else 1)
        
        # Remove the markers that are not used for corner detection
        self.markersRegion = {key: self.markersRegion[key] for key in self.markersRegion.keys() if key in IDS_CORNER_MARKERS}

        # Check that all the markers are detected
        if len(self.markersRegion) != 4:
            raise Exception("Not all the markers have been detected. Please check that all the markers are visible and that the camera is not too close to the fieldArea.")
        
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

    # Display method
    def display(self):
        """ Display the frame in real time. Run this method in a while loop to display the frame in real time. If the user press q (and this method returns True), the loop should be stopped.

        Returns:
            bool: True if the user pressed q, False otherwise
        """
        # Get the frame 
        _, frameCut = self.get_frame()

        # Display the obstacles
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                for p in obstacle:
                    pp = tuple(np.array([p[0], self.h_px - p[1]]).astype(int))
                    cv2.circle(frameCut,pp,5,(0,0,0),-1)

        # Display the start position
        if self.startPosition is not None:
            p = tuple(np.array([self.startPosition[0], self.h_px - self.startPosition[1]]).astype(int))
            cv2.circle(frameCut,p,5,(0,255,0),-1)
            cv2.putText(frameCut,"Start",p,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # Display the goal position
        if self.goalPosition is not None:
            p = tuple(np.array([self.goalPosition[0], self.h_px - self.goalPosition[1]]).astype(int))
            cv2.circle(frameCut,p,5,(0,255,0),-1)
            cv2.putText(frameCut,"Goal",p,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # Display the optimal path
        if self.optimalPath is not None:
            for i in range(len(self.optimalPath)-1):
                p1 = tuple(np.array([self.optimalPath[i][0], self.h_px - self.optimalPath[i][1]]).astype(int))
                p2 = tuple(np.array([self.optimalPath[i+1][0], self.h_px - self.optimalPath[i+1][1]]).astype(int))
                cv2.line(frameCut,p1,p2,(0,255,0),1)

            for p in self.optimalPath:
                p = tuple(np.array([p[0], self.h_px - p[1]]).astype(int))
                cv2.circle(frameCut,p,5,(0,255,0),-1)

        # Display the estimated position
        if self.robotEstimatedPosition is not None and self.robotEstimatedOrientation is not None:
            # Display the position
            if not self.save_states:
                p = tuple(np.array([self.robotEstimatedPosition[0], self.h_px - self.robotEstimatedPosition[1]]).astype(int))
                cv2.circle(frameCut,p,5,(255,0,0),-1)
            else:
                # Save the position
                self.robotEstimatedPositionHistory.append(self.robotEstimatedPosition)

                # Display all the positions
                for p in self.robotEstimatedPositionHistory:
                    p = tuple(np.array([p[0], self.h_px - p[1]]).astype(int))
                    cv2.circle(frameCut,p,5,(255,0,0),-1)

            # Display the orientation
            o = self.robotEstimatedOrientation
            o = tuple(np.array([p[0] + 50*np.cos(o), p[1] - 50*np.sin(o)]).astype(int))
            cv2.arrowedLine(frameCut,p,o,(255,0,0),2)

        # Display the measured position and orientation
        if self.robotMeasuredPosition is not None and self.robotMeasuredOrientation is not None:
            # Display the position
            if not self.save_states:
                p = tuple(np.array([self.robotMeasuredPosition[0], self.h_px - self.robotMeasuredPosition[1]]).astype(int))
                cv2.circle(frameCut,p,5,(0,0,255),-1)
            else:
                # Save the position
                self.robotMeasuredPositionHistory.append(self.robotMeasuredPosition)

                # Display all the positions
                for p in self.robotMeasuredPositionHistory:
                    p = tuple(np.array([p[0], self.h_px - p[1]]).astype(int))
                    cv2.circle(frameCut,p,5,(0,0,255),-1)

            # Display the orientation
            o = self.robotMeasuredOrientation
            o = tuple(np.array([p[0] + 50*np.cos(o), p[1] - 50*np.sin(o)]).astype(int))
            cv2.arrowedLine(frameCut,p,o,(0,0,255),2)

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
    def originToFieldReference(self,points):
        """ Convert the points into fieldArea from the origin reference system to the destination reference system

        Args:
            points (np.array((2,n))): Points to convert

        Returns:
            np.array((2,n)): Converted points
        """
        # Convert the points
        points = cv2.perspectiveTransform(np.float32([points]),self.matrix)[0]

        # Invert the y axis to have the origin in the bottom left corner
        points[:,1] = self.h_px - points[:,1]

        # Return the points
        return points