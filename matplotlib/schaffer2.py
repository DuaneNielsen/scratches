import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def f(x, y):
    return torch.sin(torch.sqrt(x ** 2 + y ** 2))

def shaffer2(x, y):
    return torch.sin(torch.sin((x**2 - y**2 - 0.5))) / ((1 + 0.001 * (x**2 + y**2))**2) + 0.5


x = torch.linspace(-100, 100, 200)
y = torch.linspace(-100, 100, 200)
xv, yv = torch.meshgrid([x,y])
zv = shaffer2(xv, yv)

xv = xv.cpu().numpy()
yv = yv.cpu().numpy()
zv = zv.cpu().numpy()

#plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line1 = ax.plot_surface(xv, yv, zv, rstride=20, cstride=20, vmin=zv.min(), vmax=zv.max(), cmap=plt.get_cmap('viridis'))
fig.canvas.draw_idle()
plt.show()
