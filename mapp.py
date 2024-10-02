import csv
import numpy as np

# Filepath for the output CSV file
csv_file_path = "/home/salmaaldhanhani/ompl/generated_elevation_data.csv"

# Parameters for generating the elevation data
lat_min, lat_max = 50.0, 51.0  # Latitude range
lon_min, lon_max = 24.0, 25.0  # Longitude range
grid_size = 100  # Resolution of the grid

# Create a grid of latitude and longitude values
lats = np.linspace(lat_min, lat_max, grid_size)
lons = np.linspace(lon_min, lon_max, grid_size)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Generate some synthetic elevation data (e.g., Gaussian hills and valleys)
elevations = np.sin(lon_grid) * np.cos(lat_grid) * 100  # Just an example of elevation variation

# Write the data to CSV
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Longitude', 'Latitude', 'Elevation'])  # Write header
    for i in range(grid_size):
        for j in range(grid_size):
            writer.writerow([lons[j], lats[i], elevations[i, j]])

print(f"CSV file '{csv_filename}' generated successfully!")
