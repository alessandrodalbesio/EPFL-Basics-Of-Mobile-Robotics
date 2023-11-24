import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from dijkstra import DijkstraSPF, Graph
from utils.settings import w_px, h_px
import cv2
import math

class Global:
    
    def __init__(self, obstacles):
        # all attribute of class
        self.initialPoint = None
        self.finalPoint = None
        self.obstacles = obstacles
        self.nb_pts = None
        self.all_points = None

        self.lines = None

        self.optimal_path = None
        self.idx_goal_pt = 1
        self.goal_pt = None

    ## 1) Build visibility graph

    def find_visible_lines(self):
        """Connect each node together, and remove lines which cross obstacles.

        """

        # Create Shapely polygons
        polygons = [ Polygon(obstacle) for obstacle in self.obstacles]
    
        # Create an array of all points (initial, final points, and vertices)
        goal_pts =  np.concatenate(([self.initialPoint],[self.finalPoint]))
        obstacles_pts = np.concatenate(self.obstacles)
        self.all_points = np.concatenate((goal_pts, obstacles_pts))

        # Connecting all points 
        all_lines = []

        for i in range(len(self.all_points)):
        
                # Define current point
                p1 = Point(self.all_points[i])

                # Iterate through other points 
                for j in range(len(self.all_points)):

                    # Skip if the same point
                    if i==j:
                        continue

                    # Define the other point
                    p2 = Point(self.all_points[j])

                    # Create a line segment between the two points 
                    segment = LineString([p1, p2])
                    all_lines.extend([segment])

        # Filter to remove lines which intersect the polygons
        self.lines = [line for line in all_lines if not any(line.crosses(polygon) or (line.within(polygon)) for polygon in polygons )]


    def plot_visibility(self):
        """Plots the map with obstacles, initial and final points, and all visible lines.
        """

        # Call 'find_visible_lines' to compute all lines which don't cross any obstacle
        self.find_visible_lines()

        # Create Shapely polygons
        polygons = [ Polygon(obstacle) for obstacle in self.obstacles]

        # Extract x and y coordinates from LineString objects
        x_coords_lines = [list(line.xy[0]) for line in self.lines]
        y_coords_lines = [list(line.xy[1]) for line in self.lines]

        # Extract x and y coordinates from Polygon objects
        x_coords_polygons = [list(polygon.exterior.xy[0]) + [polygon.exterior.xy[0][0]] for polygon in polygons]
        y_coords_polygons = [list(polygon.exterior.xy[1]) + [polygon.exterior.xy[1][0]] for polygon in polygons]

        # Plot the LineString objects
        for x, y in zip(x_coords_lines, y_coords_lines):
            plt.plot(x, y, marker='o', color = 'grey',mfc = 'black')

        # Plot the Polygon objects with filling
        for x, y in zip(x_coords_polygons, y_coords_polygons):
            plt.fill(x, y, alpha=0.3, color='green')

        # Plot initial an final points
        plt.plot(self.initialPoint[0],self.initialPoint[1],marker='8',color='green', markersize=10)
        plt.plot(self.finalPoint[0],self.finalPoint[1],marker='X',color='red', markersize=20)
        plt.show()

    ## 2) Find optimal path

    def create_weight_matrix(self):
        """Builds the weight matrix by computing the distance between each connected points.

        Returns:
            np.array((n, n)): weight matrix
        """

        # Create an empty numpy array of size len(all_points)
        self.nb_pts = len(self.all_points)
        weight_matrix = np.zeros((self.nb_pts,self.nb_pts))

        for i in range (len(self.lines)):

            line_dist = self.lines[i].length

            line_coord = np.array(self.lines[i].coords)

            x_index = ((line_coord[0,0] == self.all_points[:,0]) & (line_coord[0,1] == self.all_points[:,1]))
            y_index = ((line_coord[1,0] == self.all_points[:,0]) & (line_coord[1,1] == self.all_points[:,1]))

            weight_matrix[x_index,y_index] = line_dist

        return weight_matrix


    def find_optimal_path(self, initialPoint, finalPoint):
        """Finds optimal path using a weight matrix and the Dijkstra library
        
        Args:
            np.array([x,y]): Initial point 
            np.array([x,y]): Final point 
        """

        self.initialPoint = initialPoint
        self.finalPoint = finalPoint

        weight_matrix = self.create_weight_matrix()

        nodes = np.arange(0, self.nb_pts, 1)
        graph = Graph()

        for i in range (self.nb_pts):
            for j in range (self.nb_pts):
                if (weight_matrix[i,j] !=0):
                    graph.add_edge(nodes[i], nodes[j], weight_matrix[i,j])

        dijkstra = DijkstraSPF(graph, nodes[0])

        optimal_nodes = (dijkstra.get_path(nodes[1]))

        self.optimal_path = [ self.all_points[node] for node in optimal_nodes]

    
    ## 3) Global navigation controller 

    def compute_angle_traj(self, estimated_pt):
        """Computes the angle of the vector formed by 2 points.
        
        Args:
            np.array([x,y]): Current goal point 
            np.array([x,y]): Current estimated point 

        Returns:
            float: angle of the vector, in range [0,2pi]
        """
    
        deltaX = self.goal_pt[0] - estimated_pt[0]
        deltaY = self.goal_pt[1] - estimated_pt[1]
        
        # All the angles are expressed in the range (O,2pi) for consistency
        traj_angle = (np.arctan2(deltaY, deltaX) + 2 * np.pi ) % (2 * np.pi)
        
        return traj_angle
    
    def global_controller(self, estimated_pt,estimated_angle):
        """Controller which makes the robot follow the optimal global path.
        
        Args:
            np.array([x,y]): Current estimated point 
            float: Current estimated angle
            np.array([x,y]): Current goal point 
            int: Index of current goal point 
        
        Returns:
            int: Left motor speed
            int: Right motor speed
        """
        #print("ENTER LOOP")
        goal_achieved = False

        # Defining thresholds, scaling coefficients, and nominal speed
        dist_from_goal_thresh = 50 # TO BE CHANGED
        angle_thresh = 0.2 # in rad, TO BE CHANGED
        nominal_speed = 100 # TO BE CHANGED
        k_angle = 100 # TO BE CHANGED
        k_traj = 300 # TO BE CHANGED

        # Initial step
        if self.goal_pt is None :
            #print('INI')
            self.goal_pt = self.optimal_path[1]
            #idx_goal_pt = 1

        # Change intermediate goal point 
        if np.linalg.norm(estimated_pt - self.goal_pt)< dist_from_goal_thresh:
            #print(np.linalg.norm(estimated_pt - self.goal_pt))
            # Final step: put motor speed to 0 if the final goal is reached
            #print(self.goal_pt,self.optimal_path[-1])
            if ((self.goal_pt == self.optimal_path[-1]).all()):
                #print('arrived to goal')
                motorLeft = 0
                motorRight = 0
                goal_achieved = True
                return motorLeft,motorRight, goal_achieved
            #print('idx goal:',self.idx_goal_pt)
            self.idx_goal_pt = self.idx_goal_pt + 1
            self.goal_pt = self.optimal_path[self.idx_goal_pt]

        # Compute the difference between the trajectory angle and the estimated angle
        traj_angle = self.compute_angle_traj(estimated_pt) 
        angle_diff = (traj_angle - estimated_angle)
        #print('traj_angle:',traj_angle/math.pi,'pi')
        #print('estimated angle:',estimated_angle/math.pi,'pi')
        #print('angle diff:',angle_diff/math.pi,'pi')


        # Update direction of robot
        if (abs(angle_diff)> angle_thresh):  
            #print("ANGLE")
            # Hypothesis: when angle_diff < 0: right motor < 0 and left motor > 0 NOT SURE, TO BE CHANGED
            motorLeft = - k_angle * angle_diff
            motorRight = + k_angle * angle_diff

        # Update trajectory of robot
        else:
            #print("TRAJ")
            motorLeft = nominal_speed - k_traj * angle_diff
            motorRight = nominal_speed + k_traj * angle_diff


        return motorLeft,motorRight, goal_achieved