import cv2
import numpy as np
from math import ceil
# Import the logger file in ../utils
import sys
import shutil
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import logger

class Marker:
    TYPE = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    LABELS = {"LT": 0, "RT": 1, "LB": 2, "RB": 3,}

    def __init__(self):
        pass



    def generate(self):
        """ Generate the markers based on the LABELS constant and save them in A4 paper format so that they can be printed """
        
        # Parameters
        a4_width = int(8.27 * 300)
        a4_height = int(11.69 * 300)
        img_size = 500 # The markers are 600x600 pixels
        margin = 100 # The margin between the markers and the borders of the A4 paper

        # Compute the padding between the images
        img_per_row = (a4_width-2*margin) // img_size
        horizontal_padding = (a4_width - 2*margin - img_per_row*img_size) // (img_per_row-1)

        # Create an empty white image with the size of an A4 paper (assume that all the markers can fit in an A4 paper)
        vertical_padding = 50
        max_rows = 0
        while (a4_height-2*margin-(max_rows+1)*img_size-max_rows*vertical_padding) > 0: max_rows += 1
        number_of_rows = ceil(len(Marker.LABELS) / img_per_row)
        imgs = []
        for i in range(ceil(number_of_rows / max_rows)):
            imgs.append(np.ones((a4_height,a4_width,3),dtype=np.uint8) * 255)

        # Iterate through the markers and add them to the images
        page = 0
        rows_counter = 0
        for i in range(len(self.LABELS)):
            # Generate the marker
            marker = cv2.aruco.generateImageMarker(self.TYPE, i, img_size)
            # Convert the marker to a numpy array
            marker = np.array(marker)
            # Convert the marker to an image
            marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
            
            # Compute the position of the marker in the image
            row = rows_counter
            col = i % img_per_row
            # Compute the position of the marker in the image
            x = margin + col*(img_size+horizontal_padding)
            y = margin + row*(img_size+vertical_padding)
            # Add the marker to the image
            imgs[page][y:y+img_size,x:x+img_size] = marker
            
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

        # Display the log
        logger.debug(f"Markers generated successfully and saved in the file at {file_name}")



    def detect(self,stream = None):
        N_ITERATIONS = 250

        # Create a VideoCapture object if no stream is given
        if stream == None:
            cam = cv2.VideoCapture(0)
        else:
            cam = stream
        
        corners = []
        ids = []

        # Iterate through the frames and average the corners found
        for i in range(N_ITERATIONS):
            # Get the image
            _, frame = cam.read()

            # Detect the markers
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)

            # Detect the markers
            detected_corners, detected_ids, _ = detector.detectMarkers(frame)

            # Unpack the corners
            if detected_corners is not None:
                detected_corners = [corner[0] for corner in detected_corners]
            # Unpack the ids
            if detected_ids is not None:
                detected_ids = [id[0] for id in detected_ids]

            # Append the corners found
            corners.append(detected_corners)
            ids.append(detected_ids)


        # Create the variable to contains the information
        ID_FILTERED = [0,1,2,3]
        avg_corners = {}
        for id_filtered in ID_FILTERED:
            avg_corners[id_filtered] = {
                "x": 0,
                "y": 0,
                "num_samples": 0
            }

        # Iterate through the corners and average the values
        for i in range(len(corners)):
            if corners[i] is not None:
                for j in range(len(corners[i])):
                    if ids[i][j] in ID_FILTERED:
                        avg_corners[ids[i][j]]["x"] += corners[i][j][0][0]
                        avg_corners[ids[i][j]]["y"] += corners[i][j][0][1]
                        avg_corners[ids[i][j]]["num_samples"] += 1

        # Reorder avg_corners based on the ids
        avg_corners = {k: v for k, v in sorted(avg_corners.items(), key=lambda item: item[0])}

        if stream == None:
            frame_copy = frame.copy()
            # Display on the last frame a point where the marker is
            for id_filtered in ID_FILTERED:
                if avg_corners[id_filtered]["num_samples"] > 0:
                    avg_corners[id_filtered]["x"] /= avg_corners[id_filtered]["num_samples"]
                    avg_corners[id_filtered]["y"] /= avg_corners[id_filtered]["num_samples"]
                    cv2.circle(frame, (int(avg_corners[id_filtered]["x"]), int(avg_corners[id_filtered]["y"])), 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(id_filtered), (int(avg_corners[id_filtered]["x"]), int(avg_corners[id_filtered]["y"])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            h = 210*4
            w = 295*4
            origin_points = [[corner["x"],corner["y"]] for corner in avg_corners.values() if corner["num_samples"] > 0]
            print(origin_points)
            target_points = [[w,h],[0,h],[0,0],[w,0]]
            matrix = cv2.getPerspectiveTransform(np.float32(origin_points),np.float32(target_points))
            transformed = cv2.warpPerspective(frame_copy,matrix,(w,h))
            op = np.float32([300,175])
            tp = matrix @ np.float32([300,175,1]).reshape(3,1)
            # Display tp
            # Flip the x axis of the transformed image
            transformed = cv2.flip(transformed,1)
            cv2.circle(transformed, (int(tp[0]), int(tp[1])), 5, (0, 0, 255), -1)
            cv2.imshow("Transformed",transformed)

            # Display the frame
            # Add the point [300,175]
            cv2.circle(frame, (300, 175), 5, (0, 0, 255), -1)
            cv2.imshow("Frame",frame)
            cv2.waitKey(0)
            
            # Release the camera            
            cam.release()
            cv2.destroyAllWindows()
        
        

if __name__ == "__main__":
    marker = Marker()
    marker.detect()