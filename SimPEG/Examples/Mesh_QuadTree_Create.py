from SimPEG import *

def run(plotIt=True):
    from SimPEG import Mesh, np
    M = Mesh.TreeMesh([32,32])
    M.refine(3)
    def function(cell):
        xyz = cell.center
        for i in range(3):
            if np.abs(np.sin(xyz[0]*np.pi*2)*0.5 + 0.5 - xyz[1]) < 0.2*i:
                return 6-i
        return 0
    M.refine(function);
    if plotIt: M.plotGrid(showIt=True)

if __name__ == '__main__':
    Utils._makeExample(__file__)
    run()
