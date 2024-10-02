import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate  # Import interpolate from scipy
import ompl.base as ob
import ompl.geometric as og
import math
import cv2
from PIL import Image
import time


# Euclidean heuristic for comparison
def euclidean_heuristic(node1, node2):
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

# Euclidean heuristic for 2.5D (including the z-axis)
def euclidean_heuristic_25d(node1, node2):
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2 + (node1[2] - node2[2])**2)

# Manhattan heuristic for 2D
def manhattan_heuristic(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

# Manhattan heuristic for 2.5D (including the z-axis)
def manhattan_heuristic_25d(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1]) + abs(node1[2] - node2[2])

# Geodesic heuristic for 2.5D (takes into account elevation changes)
def geodesic_heuristic_25d(node1, node2, heightmap):
    z1 = heightmap[int(node1[1] * heightmap.shape[0]), int(node1[0] * heightmap.shape[1])]
    z2 = heightmap[int(node2[1] * heightmap.shape[0]), int(node2[0] * heightmap.shape[1])]
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2 + (z1 - z2)**2)

# Load 2D map image for 2D planning
def load_map(filepath):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = img.resize((100, 100))  # Resize the map for faster processing
    map_array = np.array(img)
    map_array = np.where(map_array < 128, 0, 1)  # Set a threshold for obstacles
    return map_array

# State validity check for 2D planning
def isStateValid2D(state, map_array):
    x = int(state[0] * (map_array.shape[1] - 1))  # Scale to map size
    y = int(state[1] * (map_array.shape[0] - 1))
    return map_array[y, x] == 1

# Load heightmap for 2.5D planning
def load_heightmap(filepath):
    heightmap = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if heightmap is None:
        raise FileNotFoundError(f"Could not find the heightmap at {filepath}")
    heightmap = heightmap.astype(np.float32) / 255.0
    return heightmap

# State validity check for 2.5D planning
def isStateValid25D(state, heightmap):
    x_idx = int(state[0] * heightmap.shape[1])
    y_idx = int(state[1] * heightmap.shape[0])
    z_value = heightmap[y_idx, x_idx]
    return np.isclose(state[2], z_value, atol=0.05)

# Calculate the total Euclidean distance for 2D paths
def calculate_total_distance_2d(path):
    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += euclidean_heuristic(path[i-1], path[i])
    return total_distance

# Calculate the total Euclidean distance for 2.5D paths
def calculate_total_distance_25d(path):
    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += euclidean_heuristic_25d(path[i-1], path[i])
    return total_distance

# Calculate the total Manhattan distance for 2D paths
def calculate_total_manhattan_2d(path):
    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += manhattan_heuristic(path[i-1], path[i])
    return total_distance

# Calculate the total Manhattan distance for 2.5D paths
def calculate_total_manhattan_25d(path):
    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += manhattan_heuristic_25d(path[i-1], path[i])
    return total_distance

# Calculate the total Geodesic distance for 2.5D paths
def calculate_total_geodesic_25d(path, heightmap):
    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += geodesic_heuristic_25d(path[i-1], path[i], heightmap)
    return total_distance

# Planning function for 2D
def plan_2d_with_map(map_filepath):
    map_array = load_map(map_filepath)
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0)
    bounds.setHigh(1)
    space.setBounds(bounds)

    start = ob.State(space)
    goal = ob.State(space)
    start[0], start[1] = 0.1, 0.1
    goal[0], goal[1] = 0.9, 0.9

    space_information = ob.SpaceInformation(space)
    space_information.setStateValidityChecker(ob.StateValidityCheckerFn(lambda s: isStateValid2D(s, map_array)))

    pdef = ob.ProblemDefinition(space_information)
    pdef.setStartAndGoalStates(start, goal)

    planner = og.BITstar(space_information)
    planner.setProblemDefinition(pdef)
    planner.setup()

    start_time = time.time()
    solved = planner.solve(5.0)
    elapsed_time = time.time() - start_time

    if solved:
        path = pdef.getSolutionPath()
        path_states = [(state[0], state[1]) for state in path.getStates()]
        return path_states, elapsed_time
    return None, None

# Planning function for 2.5D
def plan_25d_with_map(heightmap_filepath):
    heightmap = load_heightmap(heightmap_filepath)
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, 0)
    bounds.setHigh(0, 1)
    bounds.setLow(1, 0)
    bounds.setHigh(1, 1)
    bounds.setLow(2, 0)
    bounds.setHigh(2, 1)
    space.setBounds(bounds)

    start_x, start_y, goal_x, goal_y = 0.1, 0.1, 0.9, 0.9
    start_z = float(heightmap[int(start_y * heightmap.shape[0]), int(start_x * heightmap.shape[1])])
    goal_z = float(heightmap[int(goal_y * heightmap.shape[0]), int(goal_x * heightmap.shape[1])])

    start = ob.State(space)
    goal = ob.State(space)
    start[0], start[1], start[2] = float(start_x), float(start_y), float(start_z)
    goal[0], goal[1], goal[2] = float(goal_x), float(goal_y), float(goal_z)

    space_information = ob.SpaceInformation(space)
    space_information.setStateValidityChecker(ob.StateValidityCheckerFn(lambda s: isStateValid25D(s, heightmap)))

    pdef = ob.ProblemDefinition(space_information)
    pdef.setStartAndGoalStates(start, goal)

    planner = og.BITstar(space_information)
    planner.setProblemDefinition(pdef)
    planner.setup()

    start_time = time.time()
    solved = planner.solve(5.0)
    elapsed_time = time.time() - start_time

    if solved:
        path = pdef.getSolutionPath()
        path_states = [(state[0], state[1], state[2]) for state in path.getStates()]
        return path_states, elapsed_time
    return None, None

# Plot the 2D and 2.5D paths on their respective maps
# def plot_paths(map_array, path_2d, path_25d, heightmap):
#     plt.subplot(1, 2, 1)
#     plt.imshow(map_array, cmap='gray', origin='lower')
#     if path_2d:
#         x_vals, y_vals = zip(*path_2d)
#         plt.plot(np.array(x_vals) * map_array.shape[1], np.array(y_vals) * map_array.shape[0], 'r-')
#     plt.title("2D Path")

#     plt.subplot(1, 2, 2)
#     plt.imshow(heightmap, cmap='gray', origin='lower')
#     if path_25d:
#         x_vals, y_vals = zip(*[(p[0], p[1]) for p in path_25d])
#         plt.plot(np.array(x_vals) * heightmap.shape[1], np.array(y_vals) * heightmap.shape[0], 'b-')
#     plt.title("2.5D Path")
#     plt.show()

# Function to plot the original and spline paths on respective maps
def plot_paths(map_array, path_2d, path_25d, heightmap):
    # Plot 2D Path with Spline
    plt.subplot(1, 2, 1)
    plt.imshow(map_array, cmap='gray', origin='lower')
    
    if path_2d:
        # Plot original 2D path
        x_vals, y_vals = zip(*path_2d)
        plt.plot(np.array(x_vals) * map_array.shape[1], np.array(y_vals) * map_array.shape[0], 'r-', label="Original Path")

        # Calculate and plot 2D spline
        x_spline, y_spline = calculate_spline(path_2d)
        plt.plot(x_spline * map_array.shape[1], y_spline * map_array.shape[0], 'g-', label="Spline Path")

    plt.title("2D Path")
    plt.legend()

    # Plot 2.5D Path with Spline
    plt.subplot(1, 2, 2)
    plt.imshow(heightmap, cmap='gray', origin='lower')
    
    if path_25d:
        # Extract x, y coordinates from 2.5D path
        x_vals, y_vals = zip(*[(p[0], p[1]) for p in path_25d])
        
        # Plot original 2.5D path
        plt.plot(np.array(x_vals) * heightmap.shape[1], np.array(y_vals) * heightmap.shape[0], 'b-', label="Original Path")
        
        # Calculate and plot 2.5D spline
        x_spline, y_spline = calculate_spline([(p[0], p[1]) for p in path_25d])
        plt.plot(x_spline * heightmap.shape[1], y_spline * heightmap.shape[0], 'c-', label="Spline Path")

    plt.title("2.5D Path")
    plt.legend()
    
    plt.show()
# Function to generate and interpolate a smooth path from waypoints
def calculate_spline(path, num_points=100, degree=3):
    if len(path) < 2:
        return path  # Return original path if not enough points for interpolation
    
    x_vals, y_vals = zip(*path)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # Calculate the spline representation
    tck, u = interpolate.splprep([x_vals, y_vals], s=0, k=degree)
    
    # Generate new interpolated points
    u_fine = np.linspace(0, 1, num=num_points)
    x_spline, y_spline = interpolate.splev(u_fine, tck)
    
    return np.array(x_spline), np.array(y_spline)

def interpolate_path(waypoints):
    if waypoints is None or len(waypoints) < 2:
        return None, None
    
    # Split waypoints into x and y components
    waypoints = np.array(waypoints)
    x = waypoints[:, 0]
    y = waypoints[:, 1]

    # Generate spline representation of the path
    tck, u = interpolate.splprep([x, y], s=0, k=3)  # k=3 for cubic spline, s=0 means no smoothing

    # Generate new interpolated points
    u_fine = np.linspace(0, 1, num=100)  # Adjust 'num' for smoother path
    x_fine, y_fine = interpolate.splev(u_fine, tck)

    return x_fine, y_fine

# Function to plot the waypoints and interpolated path
def plot_interpolated_path(waypoints, x_fine, y_fine):
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro', label="Waypoints")  # Waypoints
    plt.plot(x_fine, y_fine, 'b-', label="Spline Path")  # Spline interpolation
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Spline Interpolation of BIT* Path')
    plt.grid(True)
    plt.show()

# Main function to run both 2D and 2.5D plans and compare results
if __name__ == "__main__":
    map_filepath = "/home/salmaaldhanhani/ompl/map_test.png"
    heightmap_filepath = "/home/salmaaldhanhani/ompl/grayscale_elevation_map.png"

    print("Running BIT* in 2D with Map...")
    path_2d, time_2d = plan_2d_with_map(map_filepath)

    print("Running BIT* in 2.5D with Map...")
    path_25d, time_25d = plan_25d_with_map(heightmap_filepath)

    if path_2d and path_25d:
        # Calculate the total Euclidean, Manhattan, and Geodesic distances for 2D and 2.5D paths
        total_distance_2d_euclidean = calculate_total_distance_2d(path_2d)
        total_distance_2d_manhattan = calculate_total_manhattan_2d(path_2d)
        
        total_distance_25d_euclidean = calculate_total_distance_25d(path_25d)
        total_distance_25d_manhattan = calculate_total_manhattan_25d(path_25d)
        total_distance_25d_geodesic = calculate_total_geodesic_25d(path_25d, load_heightmap(heightmap_filepath))
        
        # Plot the paths
        map_array = load_map(map_filepath)
        heightmap = load_heightmap(heightmap_filepath)
        plot_paths(map_array, path_2d, path_25d, heightmap)
        
       
        
        # Print the distances and times
        print(f"2D Path Time: {time_2d:.4f} seconds")
        print(f"2D Path Euclidean Distance: {total_distance_2d_euclidean:.4f}, Manhattan Distance: {total_distance_2d_manhattan:.4f}")
        
        print(f"2.5D Path Time: {time_25d:.4f} seconds")
        print(f"2.5D Path Euclidean Distance: {total_distance_25d_euclidean:.4f}, Manhattan Distance: {total_distance_25d_manhattan:.4f}, Geodesic Distance: {total_distance_25d_geodesic:.4f}")
    else:
        print("No solution found for one or both planners.")
