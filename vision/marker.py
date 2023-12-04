# Append the path of the parent directory
import sys, os, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import generic modules
import numpy as np
from math import ceil

# Import vision modules
import cv2

# Import the ids of the markers from the setting file
from utils.settings import *

# Class definition
class Marker:
    TYPE = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    def __init__(self):
        # Define the labels
        self.labels = []
        self.labels += IDS_CORNER_MARKERS
        self.labels.append(ID_GOAL_MARKER)
        self.labels.append(ID_ROBOT_MARKER)       

    def generate(self):
        """ Generate the markers based on the LABELS constant and save them in A4 paper format so that they can be printed """
        
        # Parameters
        a4_width = int(8.27 * 300)
        a4_height = int(11.69 * 300)
        marker_size = 750 # Size of the marker
        margin = 100 # The margin between the markers and the borders of the A4 paper

        # Compute the padding between the images
        min_horizontal_padding = 25
        img_per_row = (a4_width-2*margin-min_horizontal_padding) // marker_size
        horizontal_padding = (a4_width - 2*margin - img_per_row*marker_size) // (img_per_row-1)

        # Create an empty white image with the size of an A4 paper (assume that all the markers can fit in an A4 paper)
        vertical_padding = 50
        max_rows = 0
        while (a4_height-2*margin-(max_rows+1)*marker_size-max_rows*vertical_padding) > 0: max_rows += 1
        number_of_rows = ceil(len(self.labels) / img_per_row)
        imgs = []
        for i in range(ceil(number_of_rows / max_rows)):
            imgs.append(np.ones((a4_height,a4_width,3),dtype=np.uint8) * 255)

        # Iterate through the markers and add them to the images
        page = 0
        rows_counter = 0
        for i in self.labels:
            # Generate the marker
            marker = cv2.aruco.generateImageMarker(self.TYPE, i, marker_size)
            # Convert the marker to a numpy array
            marker = np.array(marker)
            # Convert the marker to an image
            marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
            
            # Compute the position of the marker in the image
            row = rows_counter
            col = i % img_per_row
            # Compute the position of the marker in the image
            x = margin + col*(marker_size+horizontal_padding)
            y = margin + row*(marker_size+vertical_padding)
            # Add the marker to the image
            imgs[page][y:y+marker_size,x:x+marker_size] = marker
            
            # Update the page and the rows_counter
            if col == img_per_row-1:
                if rows_counter == max_rows:
                    page += 1
                    rows_counter = 0
                else:
                    rows_counter += 1
        
        # If the folder vision/markers does not exist, create it
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(curr_dir+"\\markers"):
            shutil.rmtree(curr_dir+"\\markers")
        os.makedirs(curr_dir+"\\markers")

        # Save the image
        for i in range(len(imgs)):
            file_name = curr_dir+f'\\markers\\markers_{i}.png'
            cv2.imwrite(file_name,imgs[i])


    def detect(self,cam,n_iterations=10):  
        """ Detect the markers in the image and return the corners of the markers found

        Args:
            cam (Camera): The camera object
            n_iterations (int): The number of iterations to average the corners found
        
        Returns:
            avg_corners (dict): A dictionary containing the corners of the markers found
        """
        corners = []
        ids = []
        unique_ids = []
        
        # Iterate through the frames and average the corners found
        for i in range(n_iterations):
            # Get the image
            frame,_ = cam.get_frame()

            # Detect the markers
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(self.TYPE, parameters)

            # Detect the markers
            detected_corners, detected_ids, _ = detector.detectMarkers(frame)

            # Filter the case when no markers are detected
            if detected_corners is None or detected_ids is None:
                return None

            # Unpack the corners
            if detected_corners is not None:
                detected_corners = [corner[0] for corner in detected_corners]
            # Unpack the ids
            if detected_ids is not None:
                detected_ids = [id[0] for id in detected_ids]

            # Append the corners found
            corners.append(detected_corners)
            ids.append(detected_ids)
            
            for id in detected_ids:
                if id not in unique_ids:
                    unique_ids.append(id)
        
        # Create the variable to contains the information
        avg_corners = {}
        # Find the unique indeces in ids
        for id in unique_ids:
            avg_corners[id] = {
                "points": np.zeros((4,2)),
                "num_samples": 0
            }

        # Iterate through the corners and average the values
        for i in range(len(corners)):
            if corners[i] is not None:
                for j in range(len(corners[i])):
                    avg_corners[ids[i][j]]["points"] += corners[i][j]
                    avg_corners[ids[i][j]]["num_samples"] += 1

        # Reorder avg_corners in ascending order based on the key
        avg_corners = {k: v for k, v in sorted(avg_corners.items(), key=lambda item: item[0])}

        # Average all the elements in avg_corners
        for id in avg_corners.keys():
            avg_corners[id]["points"] //= avg_corners[id]["num_samples"]
        
        return avg_corners
    
if __name__ == "__main__":
    marker = Marker()
    marker.generate()