from SimPEG import Mesh, Utils, np
import matplotlib.pyplot as plt


def run(plotIt=True):
    """
        Mesh: Basic: Types
        ==================

        Here we show SimPEG used to create three different types of meshes.

    """
    sz = [16, 16]
    tM = Mesh.TensorMesh(sz)
    qM = Mesh.TreeMesh(sz)

    def refine(cell):
        if np.sqrt(((np.r_[cell.center]-0.5)**2).sum()) < 0.4:
            return 4
        return 3

    qM.refine(refine)
    rM = Mesh.CurvilinearMesh(Utils.meshutils.exampleLrmGrid(sz, 'rotate'))

    if not plotIt:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    opts = {}
    tM.plotGrid(ax=axes[0], **opts)
    axes[0].set_title('TensorMesh')
    qM.plotGrid(ax=axes[1], **opts)
    axes[1].set_title('TreeMesh')
    rM.plotGrid(ax=axes[2], **opts)
    axes[2].set_title('CurvilinearMesh')

if __name__ == '__main__':
    run()
    plt.show()
