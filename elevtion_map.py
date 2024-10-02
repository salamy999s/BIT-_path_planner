import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Create a grid of points
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)

# Generate synthetic elevation data (e.g., a Gaussian function)
Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * (X**2 + Y**2))

# Add different heights at specific locations
# Hill 1: Add a hill to the top-left
hill_1 = np.exp(-0.5 * ((X + 3)**2 + (Y + 3)**2)) * 3  # Peak height of 3 units
Z += hill_1

# Hill 2: Add another hill to the bottom-right
hill_2 = np.exp(-0.5 * ((X - 3)**2 + (Y - 3)**2)) * 5  # Peak height of 5 units
Z += hill_2

# Valley: Add a valley (negative height) at the center
valley = -np.exp(-0.5 * (X**2 + Y**2)) * 4  # Depth of the valley is -4 units
Z += valley

# --- Part 1: Grayscale Map for Visualization ---

# Normalize the elevation data to fit into grayscale (0-255)
Z_normalized = (Z - Z.min()) / (Z.max() - Z.min()) * 255

# Convert the normalized elevation data to an 8-bit unsigned integer
Z_grayscale = Z_normalized.astype(np.uint8)

# Save the grayscale elevation map as an image
img = Image.fromarray(Z_grayscale)
img.save('grayscale_elevation_map.png')

# Show the elevation map as an image
plt.imshow(Z_grayscale, cmap='gray', extent=[-5, 5, -5, 5])
plt.colorbar(label='Grayscale Elevation (0-255)')
plt.title('Grayscale Elevation Map')
plt.savefig('grayscale_elevation_map_plot.png')
plt.show()

# --- Part 2: Heightmap for Path Planning ---

# Save the actual height values to a file for path planning
# You can use numpy to save it as a matrix in a .txt or .npy format
np.savetxt('heightmap.txt', Z, fmt='%.5f')  # Save as text file with 5 decimal precision
np.save('heightmap.npy', Z)  # Save as binary numpy format

# Visualize the elevation as a 3D heightmap
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain')
ax.set_title('3D Heightmap')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Elevation')
plt.show()
