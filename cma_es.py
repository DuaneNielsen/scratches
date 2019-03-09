import torch
import torch.distributions as dist
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def rastrigin(x,y):

    n = 10
    def rs(x):
        from math import pi
        a = 10
        return a*n + x ** 2 - a * torch.cos(2 * pi * x)

    return rs(x) + rs(x)


def top_25_percent(scores, higher_is_better=True):
    """
    Calculates the top 25 best scores
    :param scores: a list of the scores
    :return: a longtensor with indices of the top 25 scores
    """
    indexed = [(i, s) for i, s in enumerate(scores)]
    indexed = sorted(indexed, key=lambda score: score[1], reverse=higher_is_better)
    indices = [indexed[i][0] for i in range(len(indexed)//4)]
    return torch.tensor(indices)


n = 100
x = torch.linspace(-5.12, 5.12, n)
y = torch.linspace(-5.12, 5.12, n)
xv, yv = torch.meshgrid([x, y])
zv = rastrigin(xv, yv)

xv = xv.cpu().numpy()
yv = yv.cpu().numpy()
zv = zv.cpu().numpy()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line1 = ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, vmin=zv.min(), vmax=zv.max(), cmap=plt.get_cmap('viridis'), alpha=0.6)
line2 = None

mu = torch.Tensor(np.random.uniform(size=2) + 2.0)
sigma = torch.Tensor(np.random.uniform(size=2) * 5.0)

for _ in range(100):
    space = dist.Normal(mu, sigma)
    parameters = space.sample((100,))
    scores = rastrigin(parameters[:, 1], parameters[:, 1])
    scores_l = scores.split(1, dim=0)

    xv = parameters[:, 0].cpu().numpy()
    yv = parameters[:, 0].cpu().numpy()
    zv = scores.cpu().numpy()

    if line2 is None:
        line2 = ax.scatter(xv, yv, zv, vmin=0, vmax=200, c=zv, cmap=plt.get_cmap('Reds'))
    else:
        line2._offsets3d = (xv, yv, zv)
    fig.canvas.draw_idle()
    plt.pause(0.5)

    best = top_25_percent(scores_l, higher_is_better=False)
    best_individual = parameters[best[0]]
    best_score = rastrigin(best_individual[0].unsqueeze(0), best_individual[1].unsqueeze(0))
    print(best_score)

    mu = torch.index_select(parameters, 0, best).mean(0)
    sigma = torch.sqrt(parameters.var(0))



