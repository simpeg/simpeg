from SimPEG import *

def run(plotIt=True):
    """
        Mesh: QuadTree: Creation
        ========================

        You can give the refine method a function, which is evaluated on every cell
        of the TreeMesh.

        Occasionally it is useful to initially refine to a constant level
        (e.g. 3 in this 32x32 mesh). This means the function is first evaluated
        on an 8x8 mesh (2^3).

    """
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
    run()
