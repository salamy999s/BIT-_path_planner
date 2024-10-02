
import ompl.base as ob
import ompl.geometric as og
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Step 1: Define the Euclidean heuristic function
def euclidean_heuristic(node1, node2):
    """Compute Euclidean distance between two nodes."""
    return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

# Step 2: Load the map image and prepare for state validity checking
def load_map(filepath):
    """Load a map image and convert it into an occupancy grid."""
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = img.resize((100, 100))  # Resize the map for faster processing
    map_array = np.array(img)
    # Normalize values: 0 (black) as obstacles, 255 (white) as free space
    map_array = np.where(map_array < 128, 0, 1)  # Set a threshold for obstacles
    return map_array

def isStateValid(state, map_array):
    """Check if the given state is in a free space based on the map."""
    x = int(state[0] * (map_array.shape[1] - 1))  # Scale state to image size
    y = int(state[1] * (map_array.shape[0] - 1))
    
    # Check if the state is within a valid area (free space)
    return map_array[y, x] == 1

# Step 3: Setting up the 2D environment with BIT*
def plan_2d_with_map():
    # Load the map for obstacle checking
    map_filepath = '/home/salmaaldhanhani/ompl/map_test.png'
    map_array = load_map(map_filepath)

    # Define a 2D space
    space = ob.RealVectorStateSpace(2)

    # Set bounds for the space
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # Define a simple start and goal state
    start = ob.State(space)
    goal = ob.State(space)
    start[0] = 0.1
    start[1] = 0.1
    goal[0] = 0.9
    goal[1] = 0.9

    # Set up the space information
    space_information = ob.SpaceInformation(space)

    # Define the state validity function using the map
    def state_checker(state):
        return isStateValid(state, map_array)

    space_information.setStateValidityChecker(ob.StateValidityCheckerFn(state_checker))

    # Define the problem with start and goal
    pdef = ob.ProblemDefinition(space_information)
    pdef.setStartAndGoalStates(start, goal)

    # Set up the BIT* planner
    planner = og.BITstar(space_information)
    planner.setProblemDefinition(pdef)
    planner.setup()

    # Try to solve the problem within 1 second
    solved = planner.solve(1.0)

    # Output the result and visualize the path
    if solved:
        print("Solution found in 2D space with map")
        path = pdef.getSolutionPath()
        print(path)
        
        # Extract the states (nodes) from the path for plotting
        path_states = []
        for state in path.getStates():
            x = state[0]
            y = state[1]
            path_states.append((x, y))
        
        # Plot the map and the path
        plot_map_with_path(map_array, path_states)
    else:
        print("No solution found in 2D space with map")

# Step 4: Visualization function to plot map and path
def plot_map_with_path(map_array, path_states):
    """Plot the map and overlay the found path."""
    plt.imshow(map_array, cmap='gray', origin='lower')
    
    # Unzip the path states into X and Y for plotting
    x_vals, y_vals = zip(*path_states)
    
    # Scale the values to fit the map size
    x_vals = [x * map_array.shape[1] for x in x_vals]
    y_vals = [y * map_array.shape[0] for y in y_vals]

    # Plot the path
    plt.plot(x_vals, y_vals, marker='o', color='red', linewidth=2, markersize=5)
    plt.title("BIT* Path on Map")
    plt.show()

# Step 5: Main function to run the planner
if __name__ == "__main__":
    print("Running BIT* in 2D with Map...")
    plan_2d_with_map()
