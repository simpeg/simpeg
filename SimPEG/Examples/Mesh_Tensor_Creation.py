from SimPEG import *

def run(plotIt=True):
    """

        Mesh: Tensor: Creation
        ======================

        For tensor meshes, there are some functions that can come
        in handy. For example, creating mesh tensors can be a bit time
        consuming, these can be created speedily by just giving numbers
        and sizes of padding. See the example below, that follows this
        notation::

            h1 = (
                   (cellSize, numPad, [, increaseFactor]),
                   (cellSize, numCore),
                   (cellSize, numPad, [, increaseFactor])
                 )

        .. note::

            You can center your mesh by passing a 'C' for the x0[i] position.
            A 'N' will make the entire mesh negative, and a '0' (or a 0) will
            make the mesh start at zero.

    """
    h1 = [(10, 5, -1.3), (5, 20), (10, 3, 1.3)]
    M = Mesh.TensorMesh([h1, h1], x0='CN')
    if plotIt:
        M.plotGrid(showIt=True)

if __name__ == '__main__':
    run()

