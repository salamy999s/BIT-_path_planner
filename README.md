# BIT-_path_planner
This Python script implements a path planning algorithm using the BIT* (Batch Informed Trees Star) planner from the OMPL (Open Motion Planning Library). The script allows for planning in both 2D and 2.5D environments.

    2D Environment: Path planning on a 2D map where obstacles are identified.
    2.5D Environment: Path planning on a heightmap that includes elevation data.

The script compares different heuristic methods such as Euclidean, Manhattan, and Geodesic distances for both environments. It also provides visualization of the planned paths and generates spline-interpolated paths for smoother trajectories.
Requirements

The following Python libraries are required:

    numpy
    matplotlib
    scipy
    cv2 (OpenCV)
    PIL (Python Imaging Library)
    ompl (Open Motion Planning Library)

You can install these libraries via pip:
pip install numpy matplotlib scipy opencv-python pillow
You also need to install OMPL. Refer to the OMPL installation guide for detailed instructions.
Features

    Path Planning:
        2D Map: The map is loaded from an image file and converted to a binary array where obstacles are identified.
        2.5D Heightmap: A grayscale heightmap image is used to model elevation for 2.5D path planning.
    Heuristics:
        Euclidean Heuristic: Computes straight-line distances between nodes.
        Manhattan Heuristic: Computes the Manhattan distance between nodes.
        Geodesic Heuristic (2.5D only): Accounts for elevation changes while computing the distance between nodes.
    Spline Interpolation:
        After planning, the path is smoothed using cubic spline interpolation for both 2D and 2.5D environments.

Usage

    Plan a Path in 2D:
        The script will load a 2D map image and run the BIT* planner to generate a path from the start to the goal.
        The solution path will be plotted and saved as an image.

    Plan a Path in 2.5D:
        The script will load a grayscale heightmap and run the BIT* planner with elevation constraints.
        The 2.5D path will be computed, including elevation changes, and plotted.

Running the Script

To run the script, execute the following command in your terminal:
python test.py
The script will load a 2D map (map_test.png) and a heightmap (grayscale_elevation_map.png) and perform path planning in both environments. It will also print out the path distances (Euclidean, Manhattan, and Geodesic for 2.5D) and the time taken for each solution.
Output

    2D Path:
        A visual representation of the 2D path on the map, including the original and interpolated (spline) paths.

    2.5D Path:
        A visual representation of the 2.5D path on the heightmap, with the elevation included in the path calculation.

    Path Metrics:
        Euclidean, Manhattan, and Geodesic distances for both 2D and 2.5D paths.
        Time taken for each planning session.

File Structure

    test.py: The main script that contains all the functions for planning and path interpolation.
    map_test.png: A grayscale image used for the 2D path planning.
    grayscale_elevation_map.png: A grayscale heightmap used for 2.5D path planning.
