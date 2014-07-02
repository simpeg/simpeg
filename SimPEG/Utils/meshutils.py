import numpy as np
from scipy import sparse as sp
from matutils import mkvc, ndgrid, sub2ind, sdiag
from codeutils import asArray_N_x_Dim
from codeutils import isScalar

def exampleLrmGrid(nC, exType):
    assert type(nC) == list, "nC must be a list containing the number of nodes"
    assert len(nC) == 2 or len(nC) == 3, "nC must either two or three dimensions"
    exType = exType.lower()

    possibleTypes = ['rect', 'rotate']
    assert exType in possibleTypes, "Not a possible example type."

    if exType == 'rect':
        return list(ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False))
    elif exType == 'rotate':
        if len(nC) == 2:
            X, Y = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (+(X - 0.5))*amt]
        elif len(nC) == 3:
            X, Y, Z = ndgrid([np.cumsum(np.r_[0, np.ones(nx)/nx]) for nx in nC], vector=False)
            amt = 0.5-np.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
            amt[amt < 0] = 0
            return [X + (-(Y - 0.5))*amt, Y + (-(Z - 0.5))*amt, Z + (-(X - 0.5))*amt]

def meshTensor(value):
    """
        **meshTensor** takes a list of numbers and tuples that have the form::

            mT = [ float, (cellSize, numCell), (cellSize, numCell, factor) ]

        For example, a time domain mesh code needs
        many time steps at one time::

            [(1e-5, 30), (1e-4, 30), 1e-3]

        Means take 30 steps at 1e-5 and then 30 more at 1e-4,
        and then one step of 1e-3.

        Tensor meshes can also be created by increase factors::

            [(10.0, 5, -1.3), (10.0, 50), (10.0, 5, 1.3)]

        When there is a third number in the tuple, it
        refers to the increase factor, if this number
        is negative this section of the tensor is flipped right-to-left.

        .. plot::

            from SimPEG import Mesh
            tx = [(10.0,10,-1.3),(10.0,40),(10.0,10,1.3)]
            ty = [(10.0,10,-1.3),(10.0,40)]
            M = Mesh.TensorMesh([tx, ty])
            M.plotGrid(showIt=True)

    """
    if type(value) is not list:
        raise Exception('meshTensor must be a list of scalars and tuples.')

    proposed = []
    for v in value:
        if isScalar(v):
            proposed += [float(v)]
        elif type(v) is tuple and len(v) == 2:
            proposed += [float(v[0])]*int(v[1])
        elif type(v) is tuple and len(v) == 3:
            start = float(v[0])
            num = int(v[1])
            factor = float(v[2])
            pad = ((np.ones(num)*np.abs(factor))**(np.arange(num)+1))*start
            if factor < 0: pad = pad[::-1]
            proposed += pad.tolist()
        else:
            raise Exception('meshTensor must contain only scalars and len(2) or len(3) tuples.')

    return np.array(proposed)

def closestPoints(mesh, pts, gridLoc='CC'):
    """
        Move a list of points to the closest points on a grid.

        :param simpeg.Mesh.BaseMesh mesh: The mesh
        :param numpy.ndarray pts: Points to move
        :param string gridLoc: ['CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez']
        :rtype: numpy.ndarray
        :return: nodeInds
    """

    pts = asArray_N_x_Dim(pts, mesh.dim)
    grid = getattr(mesh, 'grid' + gridLoc)
    nodeInds = np.empty(pts.shape[0], dtype=int)

    for i, pt in enumerate(pts):
        if mesh.dim == 1:
            nodeInds[i] = ((pt - grid)**2).argmin()
        else:
            nodeInds[i] = ((np.tile(pt, (grid.shape[0],1)) - grid)**2).sum(axis=1).argmin()

    return nodeInds

def readUBCTensorMesh(fileName):
    """
        Read UBC GIF 3DTensor mesh and generate 3D Tensor mesh in simpegTD

        Input:
        :param fileName, path to the UBC GIF mesh file

        Output:
        :param SimPEG TensorMesh object
        :return
    """

    # Interal function to read cell size lines for the UBC mesh files.
    def readCellLine(line):
        for seg in line.split():
            if '*' in seg:
                st = seg
                sp = seg.split('*')
                re = np.array(sp[0],dtype=int)*(' ' + sp[1])
                line = line.replace(st,re.strip())
        return np.array(line.split(),dtype=float)

    # Read the file as line strings, remove lines with comment = !
    msh = np.genfromtxt(fileName,delimiter='\n',dtype=np.str,comments='!')

    # Fist line is the size of the model
    sizeM = np.array(msh[0].split(),dtype=float)
    # Second line is the South-West-Top corner coordinates.
    x0 = np.array(msh[1].split(),dtype=float)
    # Read the cell sizes
    h1 = readCellLine(msh[2])
    h2 = readCellLine(msh[3])
    h3temp = readCellLine(msh[4])
    h3 = h3temp[::-1] # Invert the indexing of the vector to start from the bottom.
    # Adjust the reference point to the bottom south west corner
    x0[2] = x0[2] - np.sum(h3)
    # Make the mesh
    from SimPEG import Mesh
    tensMsh = Mesh.TensorMesh([h1,h2,h3],x0)
    return tensMsh


def readUBCTensorModel(fileName, mesh):
    """
        ReadUBC 3DTensor mesh model and generate 3D Tensor mesh model in simpegTD

    """
    f = open(fileName, 'r')
    model = np.array(map(float, f.readlines()))
    f.close()
    model = np.reshape(model, (mesh.nCz, mesh.nCx, mesh.nCy), order = 'F')
    model = model[::-1,:,:]
    model = np.transpose(model, (1, 2, 0))
    model = mkvc(model)

    return model

def writeUBCTensorMesh(mesh, fileName):
    """
        Writes a SimPEG TensorMesh to a UBC-GIF format mesh file.

        :param simpeg.Mesh.TensorMesh mesh: The mesh
        :param str fileName: File to write to
    """
    assert mesh.dim == 3
    s = ''
    s += '%i %i %i\n' %tuple(mesh.vnC)
    origin = mesh.x0
    origin.dtype = float
    origin[2] = origin[2]+mesh.hz.sum()
    s += '%.2f %.2f %.2f\n' %tuple(origin)
    s += ('%.2f '*mesh.nCx+'\n')%tuple(mesh.hx)
    s += ('%.2f '*mesh.nCy+'\n')%tuple(mesh.hy)
    s += ('%.2f '*mesh.nCz+'\n')%tuple(mesh.hz[::-1])
    f = open(fileName, 'w')
    f.write(s)
    f.close()

def writeUBCTensorModel(mesh, model, fileName):
    """
        Writes a model associated with a SimPEG TensorMesh
        to a UBC-GIF format model file.

        :param simpeg.Mesh.TensorMesh mesh: The mesh
        :param numpy.ndarray model: The model
        :param str fileName: File to write to
    """

    # Reshape model to a matrix
    modelMat = mesh.r(model,'CC','CC','M')
    # Transpose the axes
    modelMatT = modelMat.transpose((2,0,1))
    # Flip z to positive down
    modelMatTR = mkvc(modelMatT[::-1,:,:])

    np.savetxt(fileName, modelMatTR.ravel())


if __name__ == '__main__':
    from SimPEG import Mesh
    import matplotlib.pyplot as plt
    tx = [(10.0,10,-1.3),(10.0,40),(10.0,10,1.3)]
    ty = [(10.0,10,-1.3),(10.0,40)]
    M = Mesh.TensorMesh([tx, ty])
    M.plotGrid()
    plt.gca().axis('tight')
    plt.show()
