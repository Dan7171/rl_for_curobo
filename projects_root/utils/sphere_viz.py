import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Assuming your tensor is called 'spheres' with shape (65, 4) where columns are [x, y, z, r]
# One-liner to plot 3D spheres:
fig = plt.figure(); ax = fig.add_subplot(111, projection='3d'); [ax.scatter(spheres[i, 0], spheres[i, 1], spheres[i, 2], s=spheres[i, 3]*1000, alpha=0.6) for i in range(spheres.shape[0])]; ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); plt.show()

# Alternative one-liner using all points at once (more efficient):
# fig = plt.figure(); ax = fig.add_subplot(111, projection='3d'); ax.scatter(spheres[:, 0], spheres[:, 1], spheres[:, 2], s=spheres[:, 3]*1000, alpha=0.6, c=range(len(spheres)), cmap='viridis'); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); plt.show() 