import numpy as np
from SimPEG import Utils


class BaseMesh(object):
    """
    BaseMesh does all the counting you don't want to do.
    BaseMesh should be inherited by meshes with a regular structure.

    :param numpy.array n: (or list) number of cells in each direction (dim, )
    :param numpy.array x0: (or list) Origin of the mesh (dim, )

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
        self._x0 = np.array(x0, dtype=float).ravel()
        self._dim = len(self._x0)

    @property
    def x0(self):
        """
        Origin of the mesh

        :rtype: numpy.array
        :return: x0, (dim, )
        """
        return self._x0

    @property
    def dim(self):
        """
        The dimension of the mesh (1, 2, or 3).

        :rtype: int
        :return: dim
        """
        return self._dim

    @property
    def nC(self):
        """
        Total number of cells in the mesh.

        :rtype: int
        :return: nC

        .. plot::
            :include-source:

            from SimPEG import Mesh, np
            Mesh.TensorMesh([np.ones(n) for n in [2,3]]).plotGrid(centers=True,showIt=True)
        """
        return self._n.prod()

    @property
    def nN(self):
        """
        Total number of nodes

        :rtype: int
        :return: nN

        .. plot::
            :include-source:

            from SimPEG import Mesh, np
            Mesh.TensorMesh([np.ones(n) for n in [2,3]]).plotGrid(nodes=True,showIt=True)
        """
        return (self._n+1).prod()

    @property
    def nEx(self):
        """
        Number of x-edges

        :rtype: int
        :return: nEx
        """
        return (self._n + np.r_[0,1,1][:self.dim]).prod()

    @property
    def nEy(self):
        """
        Number of y-edges

        :rtype: int
        :return: nEy
        """
        return None if self.dim < 2 else (self._n + np.r_[1,0,1][:self.dim]).prod()

    @property
    def nEz(self):
        """
        Number of z-edges

        :rtype: int
        :return: nEz
        """
        return None if self.dim < 3 else (self._n + np.r_[1,1,0][:self.dim]).prod()

    @property
    def vnE(self):
        """
        Total number of edges in each direction

        :rtype: numpy.array
        :return: [nEx, nEy, nEz], (dim, )

        .. plot::
            :include-source:

            from SimPEG import Mesh, np
            Mesh.TensorMesh([np.ones(n) for n in [2,3]]).plotGrid(edges=True,showIt=True)
        """
        return np.array([x for x in [self.nEx, self.nEy, self.nEz] if not x is None])

    @property
    def nE(self):
        """
        Total number of edges.

        :rtype: int
        :return: sum([nEx, nEy, nEz])

        """
        return self.vnE.sum()

    @property
    def nFx(self):
        """
        Number of x-faces

        :rtype: int
        :return: nFx
        """
        return (self._n + np.r_[1,0,0][:self.dim]).prod()

    @property
    def nFy(self):
        """
        Number of y-faces

        :rtype: int
        :return: nFy
        """
        return None if self.dim < 2 else (self._n + np.r_[0,1,0][:self.dim]).prod()

    @property
    def nFz(self):
        """
        Number of z-faces

        :rtype: int
        :return: nFz
        """
        return None if self.dim < 3 else (self._n + np.r_[0,0,1][:self.dim]).prod()

    @property
    def vnF(self):
        """
        Total number of faces in each direction

        :rtype: numpy.array
        :return: [nFx, nFy, nFz], (dim, )

        .. plot::
            :include-source:

            from SimPEG import Mesh, np
            Mesh.TensorMesh([np.ones(n) for n in [2,3]]).plotGrid(faces=True,showIt=True)
        """
        return np.array([x for x in [self.nFx, self.nFy, self.nFz] if not x is None])

    @property
    def nF(self):
        """
        Total number of faces.

        :rtype: int
        :return: sum([nFx, nFy, nFz])

        """
        return self.vnF.sum()

    @property
    def normals(self):
        """
        Face Normals

        :rtype: numpy.array
        :return: normals, (sum(nF), dim)
        """
        if self.dim == 2:
            nX = np.c_[np.ones(self.nFx), np.zeros(self.nFx)]
            nY = np.c_[np.zeros(self.nFy), np.ones(self.nFy)]
            return np.r_[nX, nY]
        elif self.dim == 3:
            nX = np.c_[np.ones(self.nFx), np.zeros(self.nFx), np.zeros(self.nFx)]
            nY = np.c_[np.zeros(self.nFy), np.ones(self.nFy), np.zeros(self.nFy)]
            nZ = np.c_[np.zeros(self.nFz), np.zeros(self.nFz), np.ones(self.nFz)]
            return np.r_[nX, nY, nZ]

    @property
    def tangents(self):
        """
        Edge Tangents

        :rtype: numpy.array
        :return: normals, (sum(nE), dim)
        """
        if self.dim == 2:
            tX = np.c_[np.ones(self.nEx), np.zeros(self.nEx)]
            tY = np.c_[np.zeros(self.nEy), np.ones(self.nEy)]
            return np.r_[tX, tY]
        elif self.dim == 3:
            tX = np.c_[np.ones(self.nEx), np.zeros(self.nEx), np.zeros(self.nEx)]
            tY = np.c_[np.zeros(self.nEy), np.ones(self.nEy), np.zeros(self.nEy)]
            tZ = np.c_[np.zeros(self.nEz), np.zeros(self.nEz), np.ones(self.nEz)]
            return np.r_[tX, tY, tZ]

    def projectFaceVector(self, fV):
        """
        Given a vector, fV, in cartesian coordinates, this will project it onto the mesh using the normals

        :param numpy.array fV: face vector with shape (nF, dim)
        :rtype: numpy.array
        :return: projected face vector, (nF, )

        """
        assert isinstance(fV, np.ndarray), 'fV must be an ndarray'
        assert len(fV.shape) == 2 and fV.shape[0] == self.nF and fV.shape[1] == self.dim, 'fV must be an ndarray of shape (nF x dim)'
        return np.sum(fV*self.normals, 1)

    def projectEdgeVector(self, eV):
        """
        Given a vector, eV, in cartesian coordinates, this will project it onto the mesh using the tangents

        :param numpy.array eV: edge vector with shape (nE, dim)
        :rtype: numpy.array
        :return: projected edge vector, (nE, )

        """
        assert isinstance(eV, np.ndarray), 'eV must be an ndarray'
        assert len(eV.shape) == 2 and eV.shape[0] == self.nE and eV.shape[1] == self.dim, 'eV must be an ndarray of shape (nE x dim)'
        return np.sum(eV*self.tangents, 1)


class BaseRectangularMesh(BaseMesh):
    """BaseRectangularMesh"""
    def __init__(self, n, x0=None):
        BaseMesh.__init__(self, n, x0)

    @property
    def nCx(self):
        """
        Number of cells in the x direction

        :rtype: int
        :return: nCx
        """
        return self._n[0]

    @property
    def nCy(self):
        """
        Number of cells in the y direction

        :rtype: int
        :return: nCy or None if dim < 2
        """
        return None if self.dim < 2 else self._n[1]

    @property
    def nCz(self):
        """Number of cells in the z direction

        :rtype: int
        :return: nCz or None if dim < 3
        """
        return None if self.dim < 3 else self._n[2]

    @property
    def vnC(self):
        """
        Total number of cells in each direction

        :rtype: numpy.array
        :return: [nCx, nCy, nCz]
        """
        return np.array([x for x in [self.nCx, self.nCy, self.nCz] if not x is None])

    @property
    def nNx(self):
        """
        Number of nodes in the x-direction

        :rtype: int
        :return: nNx
        """
        return self.nCx + 1

    @property
    def nNy(self):
        """
        Number of nodes in the y-direction

        :rtype: int
        :return: nNy or None if dim < 2
        """
        return None if self.dim < 2 else self.nCy + 1

    @property
    def nNz(self):
        """
        Number of nodes in the z-direction

        :rtype: int
        :return: nNz or None if dim < 3
        """
        return None if self.dim < 3 else self.nCz + 1

    @property
    def vnN(self):
        """
        Total number of nodes in each direction

        :rtype: numpy.array
        :return: [nNx, nNy, nNz]
        """
        return np.array([x for x in [self.nNx, self.nNy, self.nNz] if not x is None])

    @property
    def vnEx(self):
        """
        Number of x-edges in each direction

        :rtype: numpy.array
        :return: vnEx
        """
        return np.array([x for x in [self.nCx, self.nNy, self.nNz] if not x is None])

    @property
    def vnEy(self):
        """
        Number of y-edges in each direction

        :rtype: numpy.array
        :return: vnEy or None if dim < 2
        """
        return None if self.dim < 2 else np.array([x for x in [self.nNx, self.nCy, self.nNz] if not x is None])

    @property
    def vnEz(self):
        """
        Number of z-edges in each direction

        :rtype: numpy.array
        :return: vnEz or None if dim < 3
        """
        return None if self.dim < 3 else np.array([x for x in [self.nNx, self.nNy, self.nCz] if not x is None])

    @property
    def vnFx(self):
        """
        Number of x-faces in each direction

        :rtype: numpy.array
        :return: vnFx
        """
        return np.array([x for x in [self.nNx, self.nCy, self.nCz] if not x is None])

    @property
    def vnFy(self):
        """
        Number of y-faces in each direction

        :rtype: numpy.array
        :return: vnFy or None if dim < 2
        """
        return None if self.dim < 2 else np.array([x for x in [self.nCx, self.nNy, self.nCz] if not x is None])

    @property
    def vnFz(self):
        """
        Number of z-faces in each direction

        :rtype: numpy.array
        :return: vnFz or None if dim < 3
        """
        return None if self.dim < 3 else np.array([x for x in [self.nCx, self.nCy, self.nNz] if not x is None])

    ##################################
    # Redo the numbering so they are dependent of the vector numbers
    ##################################

    @property
    def nC(self):
        """
        Total number of cells

        :rtype: int
        :return: nC
        """
        return self.vnC.prod()

    @property
    def nN(self):
        """
        Total number of nodes

        :rtype: int
        :return: nN
        """
        return self.vnN.prod()

    @property
    def nEx(self):
        """
        Number of x-edges

        :rtype: int
        :return: nEx
        """
        return self.vnEx.prod()

    @property
    def nEy(self):
        """
        Number of y-edges

        :rtype: int
        :return: nEy
        """
        if self.dim < 2: return
        return self.vnEy.prod()

    @property
    def nEz(self):
        """
        Number of z-edges

        :rtype: int
        :return: nEz
        """
        if self.dim < 3: return
        return self.vnEz.prod()

    @property
    def nFx(self):
        """
        Number of x-faces

        :rtype: int
        :return: nFx
        """
        return self.vnFx.prod()

    @property
    def nFy(self):
        """
        Number of y-faces

        :rtype: int
        :return: nFy
        """
        if self.dim < 2: return
        return self.vnFy.prod()

    @property
    def nFz(self):
        """
        Number of z-faces

        :rtype: int
        :return: nFz
        """
        if self.dim < 3: return
        return self.vnFz.prod()

    def r(self, x, xType='CC', outType='CC', format='V'):
        """
        Mesh.r is a quick reshape command that will do the best it can at giving you what you want.

        For example, you have a face variable, and you want the x component of it reshaped to a 3D matrix.

        Mesh.r can fulfil your dreams::

            mesh.r(V, 'F', 'Fx', 'M')
                   |   |     |    { How: 'M' or ['V'] for a matrix (ndgrid style) or a vector (n x dim) }
                   |   |     { What you want: ['CC'], 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez' }
                   |   { What is it: ['CC'], 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez' }
                   { The input: as a list or ndarray }


        For example::

            Xex, Yex, Zex = r(mesh.gridEx, 'Ex', 'Ex', 'M')  # Separates each component of the Ex grid into 3 matrices

            XedgeVector = r(edgeVector, 'E', 'Ex', 'V')  # Given an edge vector, this will return just the part on the x edges as a vector

            eX, eY, eZ = r(edgeVector, 'E', 'E', 'V')  # Separates each component of the edgeVector into 3 vectors
        """

        assert (type(x) == list or isinstance(x, np.ndarray)), "x must be either a list or a ndarray"
        assert xType in ['CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'], "xType must be either 'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
        assert outType in ['CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'], "outType must be either 'CC', 'N', 'F', Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', or 'Ez'"
        assert format in ['M', 'V'], "format must be either 'M' or 'V'"
        assert outType[:len(xType)] == xType, "You cannot change types when reshaping."
        assert xType in outType, 'You cannot change type of components.'
        if type(x) == list:
            for i, xi in enumerate(x):
                assert isinstance(x, np.ndarray), "x[{0:d}] must be a numpy array".format(i)
                assert xi.size == x[0].size, "Number of elements in list must not change."

            x_array = np.ones((x.size, len(x)))
            # Unwrap it and put it in a np array
            for i, xi in enumerate(x):
                x_array[:, i] = Utils.mkvc(xi)
            x = x_array

        assert isinstance(x, np.ndarray), "x must be a numpy array"

        x = x[:]  # make a copy.
        xTypeIsFExyz = len(xType) > 1 and xType[0] in ['F', 'E'] and xType[1] in ['x', 'y', 'z']

        def outKernal(xx, nn):
            """Returns xx as either a matrix (shape == nn) or a vector."""
            if format == 'M':
                return xx.reshape(nn, order='F')
            elif format == 'V':
                return Utils.mkvc(xx)

        def switchKernal(xx):
            """Switches over the different options."""
            if xType in ['CC', 'N']:
                nn = (self._n) if xType == 'CC' else (self._n+1)
                assert xx.size == np.prod(nn), "Number of elements must not change."
                return outKernal(xx, nn)
            elif xType in ['F', 'E']:
                # This will only deal with components of fields, not full 'F' or 'E'
                xx = Utils.mkvc(xx)  # unwrap it in case it is a matrix
                nn = self.vnF if xType == 'F' else self.vnE
                nn = np.r_[0, nn]

                nx = [0, 0, 0]
                nx[0] = self.vnFx if xType == 'F' else self.vnEx
                nx[1] = self.vnFy if xType == 'F' else self.vnEy
                nx[2] = self.vnFz if xType == 'F' else self.vnEz

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
                    nn = self.vnFx if 'F' in xType else self.vnEx
                elif 'y' in xType:
                    nn = self.vnFy if 'F' in xType else self.vnEy
                elif 'z' in xType:
                    nn = self.vnFz if 'F' in xType else self.vnEz
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
