import math
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.tri as tri


class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])



# Mid-points of triangle sides opposite of each corner

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
             for i in range(3)]
def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)

def draw_pdf_contours(dists, nlevels=200, subdiv=8, **kwargs):

    f,axarr = plt.subplots(1,3)

    for idx,dist in enumerate(dists):
        ax = axarr[idx]
        refiner = tri.UniformTriRefiner(triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)
        pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

        ax.tricontourf(trimesh, pvals, nlevels, **kwargs)
        ax.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75**0.5)
        ax.axis('off')
        ax.text(0.35,-0.1,"({},{},{})".format(dist._alpha[0],dist._alpha[1],dist._alpha[2]))

    if "savepath" in kwargs.keys():
        f.savefig(kwargs["savepath"])

    plt.show()

draw_pdf_contours([Dirichlet([2, 2, 2]),Dirichlet([5, 5, 5]),Dirichlet([2, 5, 15])],
                  savepath="/Users/mh/workspace/Blog/BlogWithAngular/img/Dirichlet.png")
