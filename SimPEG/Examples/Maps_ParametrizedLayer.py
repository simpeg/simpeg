from SimPEG import Mesh, Maps, np
import matplotlib.pyplot as plt


def run(plotIt=True):
    """
        Maps: Parametrized Layer
        ========================

        Build a model of a parametrized layer in a wholespace. If you want to
        build a model of a parametrized layer in a halfspace, also use
        Maps.InjectActiveCell.

        The model is

        .. code::

            m = [
                'background physical property value',
                'layer physical property value',
                'layer center',
                'layer thickness'
            ]

    """

    mesh = Mesh.TensorMesh([50, 50], x0='CC')  # 2D tensor mesh
    mapping = Maps.ParametrizedLayer(mesh)  # parametrized layer in wholespace

    # model
    m = np.hstack(
        np.r_[
            1., # background value
            2., # layer value
            -0.1, # layer center
            0.2 # layer thickness
        ]
    )
    rho = mapping * m # apply the mapping

    if plotIt is True:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        mesh.plotImage(rho, ax=ax)

if __name__ == '__main__':
    run()
    plt.show()
