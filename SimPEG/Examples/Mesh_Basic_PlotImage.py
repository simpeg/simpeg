from SimPEG import *

def run(plotIt=True):
    """
        Mesh: Basic: PlotImage
        ======================

        You can use M.PlotImage to plot images on all of the Meshes.


    """
    M = Mesh.TensorMesh([32,32])
    v = Utils.ModelBuilder.randomModel(M.vnC, seed=789)
    v = Utils.mkvc(v)

    O = Mesh.TreeMesh([32,32])
    O.refine(1)
    def function(cell):
        if (cell.center[0] < 0.75 and cell.center[0] > 0.25 and
            cell.center[1] < 0.75 and cell.center[1] > 0.25):return 5
        if (cell.center[0] < 0.9 and cell.center[0] > 0.1 and
            cell.center[1] < 0.9 and cell.center[1] > 0.1):return 4
        return 3
    O.refine(function)

    P = M.getInterpolationMat(O.gridCC, 'CC')

    ov = P * v

    if plotIt:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1,2,figsize=(10,5))

        out = M.plotImage(v, grid=True, ax=axes[0])
        cb = plt.colorbar(out[0], ax=axes[0]); cb.set_label("Random Field")
        axes[0].set_title('TensorMesh')

        out = O.plotImage(ov, grid=True, ax=axes[1], clim=[0,1])
        cb = plt.colorbar(out[0], ax=axes[1]); cb.set_label("Random Field")
        axes[1].set_title('TreeMesh')

        plt.show()

if __name__ == '__main__':
    run()
