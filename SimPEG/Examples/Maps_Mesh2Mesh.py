from SimPEG import Mesh, Maps, Utils

def run(plotIt=True):
    """

        Maps: Mesh2Mesh
        ===============

        This mapping allows you to go from one mesh to another.

    """

    M = Mesh.TensorMesh([100,100])
    h1 = Utils.meshTensor([(6,7,-1.5),(6,10),(6,7,1.5)])
    h1 = h1/h1.sum()
    M2 = Mesh.TensorMesh([h1,h1])
    V = Utils.ModelBuilder.randomModel(M.vnC, seed=79, its=50)
    v = Utils.mkvc(V)
    modh = Maps.Mesh2Mesh([M,M2])
    modH = Maps.Mesh2Mesh([M2,M])
    H = modH * v
    h = modh * H

    if not plotIt: return

    import matplotlib.pyplot as plt
    ax = plt.subplot(131)
    M.plotImage(v, ax=ax)
    ax.set_title('Fine Mesh (Original)')
    ax = plt.subplot(132)
    M2.plotImage(H,clim=[0,1],ax=ax)
    ax.set_title('Course Mesh')
    ax = plt.subplot(133)
    M.plotImage(h,clim=[0,1],ax=ax)
    ax.set_title('Fine Mesh (Interpolated)')
    plt.show()


if __name__ == '__main__':
    run()

