#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = pd.read_csv('../data/nuage_de_points.txt', sep='\t', decimal='.')
data.head()


#Compute the least squares plane

# Prepare matrices
X_mat = np.column_stack((data['X'], data['Y'], np.ones(len(data))))
Z = data['Z'].values

# Solve least squares
coeffs, residuals, rank, s = np.linalg.lstsq(X_mat, Z, rcond=None)
a, b, c = coeffs
print(f"Plane equation: Z = {a:.3f}*X + {b:.3f}*Y + {c:.3f}")



#Visualize the points and plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter points
ax.scatter(data['X'], data['Y'], data['Z'], color='r', label='Points')

# Create plane
X_range = np.linspace(data['X'].min(), data['X'].max(), 10)
Y_range = np.linspace(data['Y'].min(), data['Y'].max(), 10)
X_grid, Y_grid = np.meshgrid(X_range, Y_range)
Z_grid = a*X_grid + b*Y_grid + c

ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Point cloud and least squares plane')
plt.show()

