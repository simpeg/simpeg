"""
Maps: Parametric Block in a Layer
===================================

Parametric description of a block confined to a layer in a
wholespace. The mapping can be applied in 2D or 3D. Here we show a 2D
example.

The model is given by

.. code::

    m = np.r_[
       'value of the background',
       'value in the layer',
       'value in the block',
       'center of the layer (depth)',
       'thickness of the layer',
       'x-center of block',
       'width of the block'
    ]

"""
from SimPEG import Mesh, Maps
import numpy as np
import matplotlib.pyplot as plt


def run(plotIt=True):

    mesh = Mesh.TensorMesh([50, 50], x0='CC')  # 2D Tensor Mesh
    mapping = Maps.ParametricBlockInLayer(mesh)  # mapping

    m = np.hstack(
        np.r_[
            1.,  # value of the background
            2.,  # value in the layer
            3.,  # value in the block
            -0.1,  # center of the layer (depth)
            0.2,  # thickness of the layer
            0.3,  # x-center of block
            0.2  # width of the block
        ]
    )

    # apply the mapping to define the physical property on the mesh
    rho = mapping * m

    if plotIt is True:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        mesh.plotImage(rho, ax=ax)

if __name__ == '__main__':
    run()
    plt.show()
