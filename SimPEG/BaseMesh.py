import numpy as np
from utils import mkvc


class BaseMesh(object):
    """BaseMesh does all the counting you don't want to do.

    x0          origin                    ndarray   (dim, )
    n           number of cells           ndarray   (dim, )
    dim         dimension of mesh         int       1, 2, or 3

    nCx         num cells in x dir        int
    nCy         num cells in y dir        int
    nCz         num cells in z dir        int
    nC          total number of cells     int

    nNx         num nodes in x dir        int
    nNy         num nodes in y dir        int
    nNz         num nodes in z dir        int
    nN          total number of nodes     int

    nEx         num edges in x dir        ndarray   [nEx_x, nEx_y, nEx_z]
    nEy         num edges in y dir        ndarray   [nEy_x, nEy_y, nEy_z]
    nEz         num edges in z dir        ndarray   [nEz_x, nEz_y, nEz_z]
    nE          total number of edges     ndarray   (dim, )

    nFx         num faces in x dir        ndarray   [nFx_x, nFx_y, nFx_z]
    nFy         num faces in y dir        ndarray   [nFy_x, nFy_y, nFy_z]
    nFz         num faces in z dir        ndarray   [nFz_x, nFz_y, nFz_z]
    nF          total number of faces     ndarray   (dim, )

    """
    def __init__(self, n, x0=None):

        # Check inputs
        if x0 is None:
            x0 = np.zeros(len(n))

        if not len(n) == len(x0):
            raise Exception("Dimension mismatch. x0 != len(n)")

        if len(n) > 3:
            raise Exception("Dimensions higher than 3 are not supported.")

        # Ensure x0 & n are 1D vectors
        self._n = np.array(n, dtype=int).ravel()
        self._x0 = np.array(x0).ravel()
        self._dim = len(n)

    def x0():
        doc = "Origin of the mesh"
        fget = lambda self: self._x0
        return locals()
    x0 = property(**x0())

    def r(self, x, xType='CC', outType='CC', format='V'):
        """
        Mesh.r is a quick reshape command that will do the best it can at giving you what you want.

        For example, you have a face variable, and you want the x component of it reshaped to a 3D matrix.

        Mesh.r can fulfil your dreams...

        mesh.r(V, 'F', 'Fx', 'M')
               |   |     |    { How: 'M' or ['V'] for a matrix (ndgrid style) or a vector (n x dim) }
               |   |     { What you want: ['CC'], 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez' }
               |   { What is it: ['CC'], 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez' }
               { The input: as a list or ndarray }


        For example:

                Xex, Yex, Zex = r(mesh.gridEx, 'Ex', 'Ex', 'M')  # Separates each component of the Ex grid into 3 matrices

                XedgeVector = r(edgeVector, 'E', 'Ex', 'V')  # Given an edge vector, this will return just the part on the x edges as a vector

                eX, eY, eZ = r(edgeVector, 'E', 'E', 'V')  # Separates each component of the edgeVector into 3 vectors
        """

        assert (type(x) == list or type(x) == np.ndarray), "x must be either a list or a ndarray"
        assert xType in ['CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'], "xType must be either 'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
        assert outType in ['CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'], "outType must be either 'CC', 'N', 'F', Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
        assert format in ['M', 'V'], "format must be either 'M' or 'V'"
        assert outType[:len(xType)] == xType, "You cannot change types when reshaping."
        assert xType in outType, 'You cannot change type of components.'
        if type(x) == list:
            for i, xi in enumerate(x):
                assert type(x) == np.ndarray, "x[%i] must be a numpy array" % i
                assert xi.size == x[0].size, "Number of elements in list must not change."

            x_array = np.ones((x.size, len(x)))
            # Unwrap it and put it in a np array
            for i, xi in enumerate(x):
                x_array[:, i] = mkvc(xi)
            x = x_array

        assert type(x) == np.ndarray, "x must be a numpy array"

        x = x[:]  # make a copy.
        xTypeIsFExyz = len(xType) > 1 and xType[0] in ['F', 'E'] and xType[1] in ['x', 'y', 'z']

        def outKernal(xx, nn):
            """Returns xx as either a matrix (shape == nn) or a vector."""
            if format == 'M':
                return xx.reshape(nn, order='F')
            elif format == 'V':
                return mkvc(xx)

        def switchKernal(xx):
            """Switches over the different options."""
            if xType in ['CC', 'N']:
                nn = (self.n) if xType == 'CC' else (self.n+1)
                assert xx.size == np.prod(nn), "Number of elements must not change."
                return outKernal(xx, nn)
            elif xType in ['F', 'E']:
                # This will only deal with components of fields, not full 'F' or 'E'
                xx = mkvc(xx)  # unwrap it in case it is a matrix
                nn = self.nF if xType == 'F' else self.nE
                nn = np.r_[0, nn]

                nx = [0, 0, 0]
                nx[0] = self.nFx if xType == 'F' else self.nEx
                nx[1] = self.nFy if xType == 'F' else self.nEy
                nx[2] = self.nFz if xType == 'F' else self.nEz

                for dim, dimName in enumerate(['x', 'y', 'z']):
                    if dimName in outType:
                        assert self.dim > dim, ("Dimensions of mesh not great enough for %s%s", (xType, dimName))
                        assert xx.size == np.sum(nn), 'Vector is not the right size.'
                        start = np.sum(nn[:dim+1])
                        end = np.sum(nn[:dim+2])
                        return outKernal(xx[start:end], nx[dim])
            elif xTypeIsFExyz:
                # This will deal with partial components (x, y or z) lying on edges or faces
                if 'x' in xType:
                    nn = self.nFx if 'F' in xType else self.nEx
                elif 'y' in xType:
                    nn = self.nFy if 'F' in xType else self.nEy
                elif 'z' in xType:
                    nn = self.nFz if 'F' in xType else self.nEz
                assert xx.size == np.prod(nn), 'Vector is not the right size.'
                return outKernal(xx, nn)

        # Check if we are dealing with a vector quantity
        isVectorQuantity = len(x.shape) == 2 and x.shape[1] == self.dim

        if outType in ['F', 'E']:
            assert ~isVectorQuantity, 'Not sure what to do with a vector vector quantity..'
            outTypeCopy = outType
            out = ()
            for ii, dirName in enumerate(['x', 'y', 'z'][:self.dim]):
                outType = outTypeCopy + dirName
                out += (switchKernal(x),)
            return out
        elif isVectorQuantity:
            out = ()
            for ii in range(x.shape[1]):
                out += (switchKernal(x[:, ii]),)
            return out
        else:
            return switchKernal(x)

    def n():
        doc = "Number of Cells in each dimension (array of integers)"
        fget = lambda self: self._n
        return locals()
    n = property(**n())

    def dim():
        doc = "The dimension of the mesh (1, 2, or 3)."
        fget = lambda self: self._dim
        return locals()
    dim = property(**dim())

    def nCx():
        doc = "Number oc cells in the x direction"
        fget = lambda self: self.n[0]
        return locals()
    nCx = property(**nCx())

    def nCy():
        doc = "Number of cells in the y direction"

        def fget(self):
            if self.dim > 1:
                return self.n[1]
            else:
                return None
        return locals()
    nCy = property(**nCy())

    def nCz():
        doc = "Number of cells in the z direction"

        def fget(self):
            if self.dim > 2:
                return self.n[2]
            else:
                return None
        return locals()
    nCz = property(**nCz())

    def nC():
        doc = "Total number of cells"
        fget = lambda self: np.prod(self.n)
        return locals()
    nC = property(**nC())

    def nNx():
        doc = "Number of nodes in the x-direction"
        fget = lambda self: self.nCx + 1
        return locals()
    nNx = property(**nNx())

    def nNy():
        doc = "Number of noes in the y-direction"

        def fget(self):
            if self.dim > 1:
                return self.n[1] + 1
            else:
                return None
        return locals()
    nNy = property(**nNy())

    def nNz():
        doc = "Number of nodes in the z-direction"

        def fget(self):
            if self.dim > 2:
                return self.n[2] + 1
            else:
                return None
        return locals()
    nNz = property(**nNz())

    def nN():
        doc = "Total number of nodes"
        fget = lambda self: np.prod(self.n + 1)
        return locals()
    nN = property(**nN())

    def nEx():
        doc = "Number of x-edges"
        fget = lambda self: np.array([x for x in [self.nCx, self.nNy, self.nNz] if not x is None])
        return locals()
    nEx = property(**nEx())

    def nEy():
        doc = "Number of y-edges"

        def fget(self):
            if self.dim > 1:
                return np.array([x for x in [self.nNx, self.nCy, self.nNz] if not x is None])
            else:
                return None
        return locals()
    nEy = property(**nEy())

    def nEz():
        doc = "Number of z-edges"

        def fget(self):
            if self.dim > 2:
                return np.array([x for x in [self.nNx, self.nNy, self.nCz] if not x is None])
            else:
                return None
        return locals()
    nEz = property(**nEz())

    def nE():
        doc = "Total number of edges"
        fget = lambda self: np.array([np.prod(x) for x in [self.nEx, self.nEy, self.nEz] if not x is None])
        return locals()
    nE = property(**nE())

    def nFx():
        doc = "Number of x-faces"
        fget = lambda self: np.array([x for x in [self.nNx, self.nCy, self.nCz] if not x is None])
        return locals()
    nFx = property(**nFx())

    def nFy():
        doc = "Number of y-faces"

        def fget(self):
            if self.dim > 1:
                return np.array([x for x in [self.nCx, self.nNy, self.nCz] if not x is None])
            else:
                return None
        return locals()
    nFy = property(**nFy())

    def nFz():
        doc = "Number of z-faces"

        def fget(self):
            if self.dim > 2:
                return np.array([x for x in [self.nCx, self.nCy, self.nNz] if not x is None])
            else:
                return None
        return locals()
    nFz = property(**nFz())

    def nF():
        doc = "Total number of faces in each dimension"
        fget = lambda self: np.array([np.prod(x) for x in [self.nFx, self.nFy, self.nFz] if not x is None])
        return locals()
    nF = property(**nF())

    def normals():
        doc = "Face Normals"

        def fget(self):
            if self.dim == 2:
                nX = np.c_[np.ones(self.nF[0]), np.zeros(self.nF[0])]
                nY = np.c_[np.zeros(self.nF[1]), np.ones(self.nF[1])]
                return np.r_[nX, nY]
            elif self.dim == 3:
                nX = np.c_[np.ones(self.nF[0]), np.zeros(self.nF[0]), np.zeros(self.nF[0])]
                nY = np.c_[np.zeros(self.nF[1]), np.ones(self.nF[1]), np.zeros(self.nF[1])]
                nZ = np.c_[np.zeros(self.nF[2]), np.zeros(self.nF[2]), np.ones(self.nF[2])]
                return np.r_[nX, nY, nZ]
        return locals()
    normals = property(**normals())

    def tangents():
        doc = "Edge Tangents"

        def fget(self):
            if self.dim == 2:
                tX = np.c_[np.ones(self.nE[0]), np.zeros(self.nE[0])]
                tY = np.c_[np.zeros(self.nE[1]), np.ones(self.nE[1])]
                return np.r_[tX, tY]
            elif self.dim == 3:
                tX = np.c_[np.ones(self.nE[0]), np.zeros(self.nE[0]), np.zeros(self.nE[0])]
                tY = np.c_[np.zeros(self.nE[1]), np.ones(self.nE[1]), np.zeros(self.nE[1])]
                tZ = np.c_[np.zeros(self.nE[2]), np.zeros(self.nE[2]), np.ones(self.nE[2])]
                return np.r_[tX, tY, tZ]
        return locals()
    tangents = property(**tangents())

    def projectFaceVector(self, fV):
        """Given a vector, fV, in cartesian coordinates, this will project it onto the mesh using the normals"""
        assert type(fV) == np.ndarray, 'fV must be an ndarray'
        assert len(fV.shape) == 2 and fV.shape[0] == np.sum(self.nF) and fV.shape[1] == self.dim, 'fV must be an ndarray of shape (nF x dim)'
        return np.sum(fV*self.normals, 1)

    def projectEdgeVector(self, eV):
        """Given a vector, eV, in cartesian coordinates, this will project it onto the mesh using the tangents"""
        assert type(eV) == np.ndarray, 'eV must be an ndarray'
        assert len(eV.shape) == 2 and eV.shape[0] == np.sum(self.nE) and eV.shape[1] == self.dim, 'eV must be an ndarray of shape (nE x dim)'
        return np.sum(eV*self.tangents, 1)


if __name__ == '__main__':
    m = BaseMesh([3, 2, 4])
    print m.n
