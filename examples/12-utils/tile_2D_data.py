"""
Tiling 2D data
==============

We may want to subdivide a survey into
smaller sub-problems.
This example shows how to quickly get
the limits of smaller subsets of data
"""
from SimPEG import EM, Utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def run(plotIt=True):

    # Create a random survey
    nLocs = np.random.randint(200)+1
    xyLocs = np.random.randn(nLocs, 2)

    # let's allow a maximum of 10% of the original survey
    maxNpoints = int(nLocs/10)

    tiles = Utils.modelutils.tileSurveyPoints(xyLocs, maxNpoints)

    X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
    X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

    surveyMask = np.ones(nLocs, dtype='bool')
    ppt = np.zeros(xyLocs.shape[0])
    for tt in range(X1.shape[0]):

        # Grab the data for current tile
        ind_t = np.all([xyLocs[:, 0] >= X1[tt], xyLocs[:, 0] <= X2[tt],
                        xyLocs[:, 1] >= Y1[tt], xyLocs[:, 1] <= Y2[tt],
                        surveyMask], axis=0)

        ppt[ind_t] = ind_t.sum()
        # Remember selected data in case of tile overlap
        surveyMask[ind_t] = False

    if plotIt:
        fig = plt.figure()
        ax1 = plt.subplot()
        im = plt.scatter(xyLocs[:, 0], xyLocs[:, 1], c=ppt)
        for ii in range(X1.shape[0]):
            ax1.add_patch(Rectangle((X1[ii], Y1[ii]),
                                    X2[ii]-X1[ii],
                                    Y2[ii]-Y1[ii],
                                    facecolor='none', edgecolor='k'))
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Number of tiles: " + str(X1.shape[0]) +
                      "/" + str(xyLocs.shape[0]))
        ax1.set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    run()
    plt.show()
