Overview
This Python script implements a path planning algorithm using the BIT* (Batch Informed Trees Star)
planner from the OMPL (Open Motion Planning Library). The script allows for planning in both 2D
and 2.5D environments, with comparisons of different heuristics such as Euclidean, Manhattan, and
Geodesic distances.
It also includes features like spline interpolation for smooth paths and visualization of planned paths.
Requirements
Required Python libraries:
- numpy
- matplotlib
- scipy
- opencv-python
- pillow
- ompl
Install using:
pip install numpy matplotlib scipy opencv-python pillow
Refer to the OMPL documentation for installing OMPL.
Features
- 2D Path Planning: Uses a map image for planning paths in 2D.
- 2.5D Path Planning: Uses a heightmap image for terrain-based planning with elevation.
- Heuristic Comparison: Euclidean, Manhattan, and Geodesic distances are calculated.
- Spline Interpolation: Smooths out the paths using cubic splines.
- Path Visualization: Plots the paths on their respective maps.
Usage
Run the script by executing:
python test.py
The script will load a 2D map and a heightmap to run BIT* for both 2D and 2.5D path planning.
Results including distances and times will be displayed, and path visualizations will be generated.
Output
- 2D Path: Visualized on a binary map image.
- 2.5D Path: Visualized on a heightmap with elevation.
- Metrics: Euclidean, Manhattan, and Geodesic distances, and time taken for path planning.
