from discretize.utils import (
    exampleLrmGrid, meshTensor, closestPoints, ExtractCoreMesh
)
import numpy as np
import discretize as Mesh


def meshBuilder(xyz, h, padDist, nCmin=3, meshGlobal=None, expFact=1.3):
    """
        Function to quickly generate a Tensor or Tree mesh
        given a cloud of xyz points, finest core cell size
        and padding distance.

    """

    # Get center of the mesh
    midX = np.mean([xyz[:, 0].max(), xyz[:, 0].min()])
    midY = np.mean([xyz[:, 1].max(), xyz[:, 1].min()])
    midZ = xyz[:, 2].max()

    # Make sure the core has odd number of cells for centereing
    # + 3 buffer cells
    nCx = int((xyz[:, 0].max() - xyz[:, 0].min()) / h[0])
    nCx += int(nCmin*2 + 1 - nCx % 2)
    nCy = int((xyz[:, 1].max() - xyz[:, 1].min()) / h[1])
    nCy += int(nCmin*2 + 1 - nCy % 2)
    nCz = int((xyz[:, 2].max() - xyz[:, 2].min()) / h[2])
    nCz += int(nCmin*2 + 1 - nCy % 2)

    # Figure out paddings
    def expand(dx, pad):
        L = 0
        nC = 0
        while L < pad:
            nC += 1
            L = np.sum(dx * expFact**(np.asarray(range(nC))+1))

        return nC

    npadEast = expand(h[0], padDist[0, 0])
    npadWest = expand(h[0], padDist[0, 1])
    npadSouth = expand(h[1], padDist[1, 0])
    npadNorth = expand(h[1], padDist[1, 1])
    npadDown = expand(h[2], padDist[2, 0])
    npadUp = expand(h[2], padDist[2, 1])

    # Add paddings
    hx = [(h[0], npadWest, -expFact), (h[0], nCx), (h[0], npadEast, expFact)]
    hy = [(h[1], npadSouth, -expFact), (h[1], nCy), (h[1], npadNorth, expFact)]
    hz = [(h[2], npadDown, -expFact), (h[2], nCz), (h[2], npadUp, expFact)]

    # Create mesh
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CC0')

    mesh._x0 = np.r_[mesh.x0[0] + midX,
                     mesh.x0[1] + midY,
                     mesh.x0[2] - mesh.vectorNz[-1] + midZ]

    if meshGlobal is not None:
        # Shift tile center to closest cell in base grid
        ind = closestPoints(meshGlobal, np.r_[midX, midY, midZ], gridLoc='CC')
        shift = np.squeeze(meshGlobal.gridCC[ind,:]) - np.r_[midX, midY, midZ]
        mesh._x0 = mesh.x0 + shift

    return mesh
