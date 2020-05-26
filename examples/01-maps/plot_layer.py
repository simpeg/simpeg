"""
Maps: Parametrized Layer
========================

Build a model of a parametrized layer in a wholespace. If you want to
build a model of a parametrized layer in a halfspace, also use
maps.InjectActiveCell.

The model is

.. code::

    m = [
        'background physical property value',
        'layer physical property value',
        'layer center',
        'layer thickness'
    ]
"""
import discretize
from SimPEG import maps
import numpy as np
import matplotlib.pyplot as plt


def run(plotIt=True):

    mesh = discretize.TensorMesh([50, 50], x0="CC")  # 2D tensor mesh
    mapping = maps.ParametricLayer(mesh)  # parametric layer in wholespace

    # model
    m = np.hstack(
        np.r_[
            1.0,  # background value
            2.0,  # layer value
            -0.1,  # layer center
            0.2,  # layer thickness
        ]
    )
    rho = mapping * m  # apply the mapping

    if plotIt is True:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        mesh.plotImage(rho, ax=ax)


if __name__ == "__main__":
    run()
    plt.show()
