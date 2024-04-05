"""
Maps: Mesh2Mesh
===============

This mapping allows you to go from one mesh to another.
"""

import discretize
from SimPEG import maps, utils
import matplotlib.pyplot as plt


def run(plotIt=True):
    M = discretize.TensorMesh([100, 100])
    h1 = utils.unpack_widths([(6, 7, -1.5), (6, 10), (6, 7, 1.5)])
    h1 = h1 / h1.sum()
    M2 = discretize.TensorMesh([h1, h1])
    V = utils.model_builder.create_random_model(M.vnC, seed=79, its=50)
    v = utils.mkvc(V)
    modh = maps.Mesh2Mesh([M, M2])
    modH = maps.Mesh2Mesh([M2, M])
    H = modH * v
    h = modh * H

    if not plotIt:
        return

    ax = plt.subplot(131)
    M.plot_image(v, ax=ax)
    ax.set_title("Fine Mesh (Original)")
    ax = plt.subplot(132)
    M2.plot_image(H, clim=[0, 1], ax=ax)
    ax.set_title("Course Mesh")
    ax = plt.subplot(133)
    M.plot_image(h, clim=[0, 1], ax=ax)
    ax.set_title("Fine Mesh (Interpolated)")


if __name__ == "__main__":
    run()
    plt.show()
