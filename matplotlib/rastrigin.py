import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

""" Rastrigen function, classic example of pathological gradients
"""

def rastrigin(x,y):

    n = len(x)
    def rs(x):
        from math import pi
        a = 10
        return a*n + x ** 2 - a * torch.cos(2 * pi * x)

    return rs(x) + rs(y)

n = 30
x = torch.linspace(-5.12, 5.12, n)
y = torch.linspace(-5.12, 5.12, n)
xv, yv = torch.meshgrid([x, y])
zv = rastrigin(xv, yv)


xv = xv.cpu().numpy()
yv = yv.cpu().numpy()
zv = zv.cpu().numpy()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line1 = ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, vmin=zv.min(), vmax=zv.max(), cmap=plt.get_cmap('Spectral'))
fig.canvas.draw_idle()
plt.show()