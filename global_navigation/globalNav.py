import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from dijkstra import DijkstraSPF, Graph
from utils.settings import w_px, h_px
import cv2

class Global:
    
    def __init__(self, obstacles):
        # all attribute of class
        self.initialPoint = None
        self.finalPoint = None
        self.obstacles = obstacles

    ## 1) Build visibility graph

    def find_visible_lines(self):

        # Create Shapely polygons
        polygons = [ Polygon(obstacle) for obstacle in self.obstacles]
    
        # Create an array of all points (initial, final points, and vertices)
        goal_pts =  np.concatenate(([self.initialPoint],[self.finalPoint]))
        obstacles_pts = np.concatenate(self.obstacles)
        all_points = np.concatenate((goal_pts, obstacles_pts))

        # Connecting all points 
        all_lines = []

        for i in range(len(all_points)):
        
                # Define current point
                p1 = Point(all_points[i])

                # Iterate through other points 
                for j in range(len(all_points)):

                    # Skip if the same point
                    if i==j:
                        continue

                    # Define the other point
                    p2 = Point(all_points[j])

                    # Create a line segment between the two points 
                    segment = LineString([p1, p2])
                    all_lines.extend([segment])

        # Filter to remove lines which intersect the polygons
        filtered_lines = [line for line in all_lines if not any(line.crosses(polygon) or (line.within(polygon)) for polygon in polygons )]

        return filtered_lines, all_points

    def plot_visibility(self, lines):

        # Create Shapely polygons
        polygons = [ Polygon(obstacle) for obstacle in self.obstacles]

        # Extract x and y coordinates from LineString objects
        x_coords_lines = [list(line.xy[0]) for line in lines]
        y_coords_lines = [list(line.xy[1]) for line in lines]

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

        lines , all_points = self.find_visible_lines()

        # Create an empty numpy array of size len(all_points)
        nb_pts = len(all_points)
        weight_matrix = np.zeros((nb_pts,nb_pts))

        for i in range (len(lines)):

            line_dist = lines[i].length

            line_coord = np.array(lines[i].coords)

            x_index = ((line_coord[0,0] == all_points[:,0]) & (line_coord[0,1] == all_points[:,1]))
            y_index = ((line_coord[1,0] == all_points[:,0]) & (line_coord[1,1] == all_points[:,1]))

            weight_matrix[x_index,y_index] = line_dist

        return nb_pts, all_points, weight_matrix


    def find_optimal_path(self, initialPoint, finalPoint):
        self.initialPoint = initialPoint
        self.finalPoint = finalPoint

        nb_pts, all_points, weight_matrix = self.create_weight_matrix()

        nodes = np.arange(0, nb_pts, 1)
        graph = Graph()

        for i in range (nb_pts):
            for j in range (nb_pts):
                if (weight_matrix[i,j] !=0):
                    graph.add_edge(nodes[i], nodes[j], weight_matrix[i,j])

        dijkstra = DijkstraSPF(graph, nodes[0])

        optimal_nodes = (dijkstra.get_path(nodes[1]))

        optimal_path = [ all_points[node] for node in optimal_nodes]

        return optimal_path
    