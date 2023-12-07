# Append the path of the parent directory
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import generic modules
import numpy as np
import matplotlib.pyplot as plt
from utils.settings import *

# Import path planning modules
from shapely.geometry import Point, LineString, Polygon
from dijkstra import DijkstraSPF, Graph

class Global:
    """ This class implements the global navigation part of the project. The structure is as follows:
    1) Build visibility graph
    2) Find optimal path: weight matrix and Dijkstra algorithm
    3) Global controller
    """
    
    def __init__(self, obstacles):

        # Map attributes
        self.obstacles = obstacles
        
        # Reset all the attributes of the class
        self.reset()

    def reset(self):
        """Resets the attributes of the class. This function is called at the beginning of each new map."""
        self.nb_pts = None
        self.all_points = None
        self.lines = None
        self.optimal_path = None
        self.idx_goal_pt = 1
        self.goal_pt = None
        self.goal_reached = False

    ## 1) Build visibility graph

    def find_visible_lines(self, initialPoint, finalPoint):
        """ Find the visible lines between all points, and remove lines which cross obstacles.
        Args:
            np.array([x,y]): Initial point 
            np.array([x,y]): Final point        
        """

        # Create Shapely polygons
        polygons = [ Polygon(obstacle) for obstacle in self.obstacles]
    
        # Create an array of all points (initial, final points, and vertices)
        self.all_points =  np.concatenate(([initialPoint],[finalPoint]))
        for obstacle in self.obstacles:
            obstacle_points = np.array(obstacle)[:-1]
            self.all_points = np.concatenate((self.all_points, obstacle_points))

        # Connecting all points 
        all_lines = []
        lines_indecies = []
        for i in range(len(self.all_points)):
        
                # Define current point
                p1 = Point(self.all_points[i])

                # Iterate through other points 
                for j in range(i+1,len(self.all_points)):

                    # Define the other point
                    p2 = Point(self.all_points[j])

                    # Create a line segment between the two points 
                    segment = LineString([p1, p2])
                    all_lines.extend([segment])
                    lines_indecies.extend([(i,j)])

        # Filter to remove lines which intersect the polygons
        self.lines = all_lines.copy()
        self.lines_indecies = lines_indecies.copy()
        for i in range(len(all_lines)):
            line = all_lines[i]
            for pol in polygons:
                if line.crosses(pol) or line.within(pol):
                    self.lines.remove(line)
                    self.lines_indecies.remove(lines_indecies[i])
                    break

    def plot_visibility(self,initialPoint, finalPoint):
        """Plots the map with obstacles, initial and final points, and all visible lines.
        """

        # Call 'find_visible_lines' to compute all lines which don't cross any obstacle
        self.find_visible_lines(initialPoint, finalPoint)

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
            plt.plot(x, y, marker='o', color = 'grey',mfc = 'black',linestyle=":")

        # Plot the Polygon objects with filling
        for x, y in zip(x_coords_polygons, y_coords_polygons):
            plt.fill(x, y, alpha=0.3, color='green')

        # Plot initial an final points
        plt.plot(initialPoint[0],initialPoint[1],marker='8',color='green', markersize=10, label = 'Initial point')
        plt.plot(finalPoint[0],finalPoint[1],marker='X',color='red', markersize=20, label = 'Final point')
        plt.title('Visibility graph, with obstacles, initial, and final points')
        plt.axis('off')
        plt.legend()
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

        # Iterate over all lines to associate a distance weight
        for i in range (len(self.lines)):

            line_dist = self.lines[i].length
            line_coord = np.array(self.lines[i].coords)

            x_index = self.lines_indecies[i][0]
            y_index = self.lines_indecies[i][1]

            weight_matrix[x_index,y_index] = line_dist
            weight_matrix[y_index,x_index] = line_dist

        return weight_matrix

    def find_optimal_path(self, initialPoint, finalPoint):
        """Finds optimal path using a weight matrix and the Dijkstra library
        
        Args:
            np.array([x,y]): Initial point 
            np.array([x,y]): Final point 
        """

        # Reset the attributes of the class
        self.reset()

        # Create visibility graph
        self.find_visible_lines(initialPoint=initialPoint, finalPoint=finalPoint)
        
        # Create weight matrix
        weight_matrix = self.create_weight_matrix()

        # Initialize nodes and the graph which are input of the Dijkstra algorithm
        nodes = np.arange(0, self.nb_pts, 1)
        graph = Graph()

        # Iterate over all point to create edges of the Dijkstra graph
        for i in range (self.nb_pts):
            for j in range (self.nb_pts):
                if (weight_matrix[i,j] != 0):
                    graph.add_edge(nodes[i], nodes[j], weight_matrix[i,j])
        
        # Call the Dijkstra function
        dijkstra = DijkstraSPF(graph, nodes[0]) # nodes[0] is the Initial Point
        # Get the optimal path from the Dijkstra structure created
        optimal_nodes = (dijkstra.get_path(nodes[1])) # nodes[1] is the final point 

        # Go back to point coordinates to store the path points
        self.optimal_path = [ self.all_points[node] for node in optimal_nodes] 

        # Goal point is initialized, for the followed control part
        if self.goal_pt is None :
            self.goal_pt = self.optimal_path[1]

    
    ## 3) Global navigation controller 

    def compute_angle_traj(self, estimated_pt):
        """Computes the angle of the vector formed by 2 points.
        
        Args:
            np.array([x,y]): Current goal point 
            np.array([x,y]): Current estimated point 

        Returns:
            float: angle of the vector, in range [0,2pi]
        """
        # Compute the trajectory angle
        deltaX = self.goal_pt[0] - estimated_pt[0]
        deltaY = self.goal_pt[1] - estimated_pt[1]
        
        # All the angles are expressed in the range (O,2pi) for consistency
        traj_angle = (np.arctan2(deltaY, deltaX) + 2 * np.pi ) % (2 * np.pi)
        
        # Return the trajectory angle
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

        # Change intermediate goal point
        if np.linalg.norm(estimated_pt - self.goal_pt) < DIST_FROM_GOAL_THRESH:

            # Arrived to goal point
            if ((self.goal_pt == self.optimal_path[-1]).all()):
                motorLeft = 0
                motorRight = 0
                self.goal_reached = True
                return motorLeft,motorRight
            
            # Update goal point
            self.idx_goal_pt = self.idx_goal_pt + 1
            self.goal_pt = self.optimal_path[self.idx_goal_pt]
        
        
        # Compute the trajectory angle
        traj_angle = self.compute_angle_traj(estimated_pt) 

        # Convert the measured angle in a range (-pi,pi)
        if 0 < traj_angle <= np.pi:
            if traj_angle <= estimated_angle <= traj_angle + np.pi:
                angle_diff = estimated_angle - traj_angle
            else:
                if traj_angle + np.pi <= estimated_angle <= 2 * np.pi:
                    angle_diff = estimated_angle - traj_angle - 2 * np.pi
                else:
                    angle_diff = estimated_angle - traj_angle
        else:
            if traj_angle - np.pi <= estimated_angle <= traj_angle:
                angle_diff = estimated_angle - traj_angle
            else:
                if 0 <= estimated_angle <= traj_angle- np.pi:
                    angle_diff = estimated_angle - traj_angle + 2 * np.pi
                else:
                    angle_diff = estimated_angle - traj_angle

        # Update direction of robot
        if (abs(angle_diff) > ANGLE_THRESH):
            motorLeft = K_ANGLE * angle_diff
            motorRight = -K_ANGLE * angle_diff
        else:
            motorLeft = NOMINAL_SPEED + (K_TRAJ * abs(angle_diff) if angle_diff > 0 else 0)
            motorRight = NOMINAL_SPEED + (K_TRAJ * abs(angle_diff) if angle_diff < 0 else 0)

        # Return motor speeds
        return motorLeft,motorRight
    
    def local_goal_point_update(self, estimated_pt):
        """If a local obstacle is too close to the current goal point, the current goal point is changed to the next one, so that the robot doesn't get stuck.
        
        Args:
            np.array([x,y]): Current estimated point 
        """
        if (np.linalg.norm(estimated_pt - self.goal_pt) < DIST_FROM_GOAL_THRESH_LOCAL):
            # Update goal point
            self.idx_goal_pt = self.idx_goal_pt + 1
            self.goal_pt = self.optimal_path[self.idx_goal_pt]