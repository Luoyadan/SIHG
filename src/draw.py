from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, 0.05)
Y = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(X, Y)

Z = np.sqrt((X**2 + Y**2)/10. + 1)

xcolors = X - min(X.flat)
xcolors = xcolors/max(xcolors.flat)

surf = ax.plot_surface(X, Y, Z, cmap=cm.PuBu
    )
plt.axis('off')
plt.show()