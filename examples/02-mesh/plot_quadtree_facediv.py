"""
Mesh: QuadTree: FaceDiv
=======================

Showing the face divergence on the quadtree with numbering.
"""
import numpy as np
import matplotlib.pyplot as plt
from SimPEG import Mesh


def run(plotIt=True, n=60):

    M = Mesh.TreeMesh([[(1, 16)], [(1, 16)]], levels=4)

    M.insert_cells(
            np.c_[5, 5], np.r_[3],
            finalize=True
        )

    if plotIt:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        M.plotGrid(cells=True, nodes=False, ax=axes[0])
        axes[0].axis('off')
        axes[0].set_title('Simple QuadTree Mesh')
        axes[0].set_xlim([-1, 17])
        axes[0].set_ylim([-1, 17])

        for ii, loc in zip(range(M.nC), M.gridCC):
            axes[0].text(loc[0]+0.2, loc[1], '{0:d}'.format(ii), color='r')

        axes[0].plot(M.gridFx[:, 0], M.gridFx[:, 1], 'g>')
        for ii, loc in zip(range(M.nFx), M.gridFx):
            axes[0].text(loc[0]+0.2, loc[1], '{0:d}'.format(ii), color='g')

        axes[0].plot(M.gridFy[:, 0], M.gridFy[:, 1], 'm^')
        for ii, loc in zip(range(M.nFy), M.gridFy):
            axes[0].text(
                loc[0]+0.2, loc[1]+0.2, '{0:d}'.format(
                    (ii+M.nFx)
                ),
                color='m'
            )

        axes[1].spy(M.faceDiv)
        axes[1].set_title('Face Divergence')
        axes[1].set_ylabel('Cell Number')
        axes[1].set_xlabel('Face Number')

if __name__ == '__main__':
    run()
    plt.show()
