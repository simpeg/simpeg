from SimPEG import *

def run(plotIt=True, n=60):
    """
        Mesh: QuadTree: FaceDiv
        =======================



    """


    M = Mesh.TreeMesh([[(1,16)],[(1,16)]], levels=4)
    M._refineCell([0,0,0])
    M._refineCell([0,0,1])
    M._refineCell([4,4,2])
    M.__dirty__ = True
    M.number()


    if plotIt:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,1,figsize=(10,10))

        M.plotGrid(cells=True, nodes=False, ax=axes[0])
        axes[0].axis('off')
        axes[0].set_title('Simple QuadTree Mesh')
        axes[0].set_xlim([-1,17])
        axes[0].set_ylim([-1,17])

        for ii, loc in zip(range(M.nC),M.gridCC):
            axes[0].text(loc[0]+0.2,loc[1],'{0:d}'.format(ii), color='r')

        axes[0].plot(M.gridFx[:,0],M.gridFx[:,1], 'g>')
        for ii, loc in zip(range(M.nFx),M.gridFx):
            axes[0].text(loc[0]+0.2,loc[1],'{0:d}'.format(ii), color='g')

        axes[0].plot(M.gridFy[:,0],M.gridFy[:,1], 'm^')
        for ii, loc in zip(range(M.nFy),M.gridFy):
            axes[0].text(loc[0]+0.2,loc[1]+0.2,'{0:d}'.format((ii+M.nFx)), color='m')

        axes[1].spy(M.faceDiv)
        axes[1].set_title('Face Divergence')
        axes[1].set_ylabel('Cell Number')
        axes[1].set_xlabel('Face Number')
        plt.show()

if __name__ == '__main__':
    run()
