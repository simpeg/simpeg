from SimPEG import Utils, np, sp
from BaseMesh import BaseRectangularMesh
from View import TensorView
from DiffOperators import DiffOperators
from InnerProducts import InnerProducts

def _getTensorGrid(self, key):
    if getattr(self, '_grid' + key, None) is None:
        setattr(self, '_grid' + key, Utils.ndgrid(self.getTensor(key)))
    return getattr(self, '_grid' + key)


class BaseTensorMesh(BaseRectangularMesh):

    __metaclass__ = Utils.SimPEGMetaClass

    _meshType = 'BASETENSOR'

    _unitDimensions = [1, 1, 1]

    def __init__(self, h_in, x0=None):
        assert type(h_in) is list, 'h_in must be a list'
        h = range(len(h_in))
        for i, h_i in enumerate(h_in):
            if type(h_i) in [int, long, float]:
                # This gives you something over the unit cube.
                h_i = self._unitDimensions[i] * np.ones(int(h_i))/int(h_i)
            assert type(h_i) == np.ndarray, ("h[%i] is not a numpy array." % i)
            assert len(h_i.shape) == 1, ("h[%i] must be a 1D numpy array." % i)
            h[i] = h_i[:] # make a copy.

        BaseRectangularMesh.__init__(self, np.array([x.size for x in h]), x0)
        assert len(h) == len(self.x0), "Dimension mismatch. x0 != len(h)"

        # Ensure h contains 1D vectors
        self._h = [Utils.mkvc(x.astype(float)) for x in h]

    @property
    def h(self):
        """h is a list containing the cell widths of the tensor mesh in each dimension."""
        return self._h

    @property
    def hx(self):
        "Width of cells in the x direction"
        return self._h[0]

    @property
    def hy(self):
        "Width of cells in the y direction"
        return None if self.dim < 2 else self._h[1]

    @property
    def hz(self):
        "Width of cells in the z direction"
        return None if self.dim < 3 else self._h[2]

    @property
    def vectorNx(self):
        """Nodal grid vector (1D) in the x direction."""
        return np.r_[0., self.hx.cumsum()] + self.x0[0]

    @property
    def vectorNy(self):
        """Nodal grid vector (1D) in the y direction."""
        return None if self.dim < 2 else np.r_[0., self.hy.cumsum()] + self.x0[1]

    @property
    def vectorNz(self):
        """Nodal grid vector (1D) in the z direction."""
        return None if self.dim < 3 else np.r_[0., self.hz.cumsum()] + self.x0[2]

    @property
    def vectorCCx(self):
        """Cell-centered grid vector (1D) in the x direction."""
        return np.r_[0, self.hx[:-1].cumsum()] + self.hx*0.5 + self.x0[0]

    @property
    def vectorCCy(self):
        """Cell-centered grid vector (1D) in the y direction."""
        return None if self.dim < 2 else np.r_[0, self.hy[:-1].cumsum()] + self.hy*0.5 + self.x0[1]

    @property
    def vectorCCz(self):
        """Cell-centered grid vector (1D) in the z direction."""
        return None if self.dim < 3 else np.r_[0, self.hz[:-1].cumsum()] + self.hz*0.5 + self.x0[2]

    @property
    def gridCC(self):
        """Cell-centered grid."""
        return _getTensorGrid(self, 'CC')

    @property
    def gridN(self):
        """Nodal grid."""
        return _getTensorGrid(self, 'N')

    @property
    def gridFx(self):
        """Face staggered grid in the x direction."""
        if self.nFx == 0: return
        return _getTensorGrid(self, 'Fx')

    @property
    def gridFy(self):
        """Face staggered grid in the y direction."""
        if self.nFy == 0 or self.dim < 2: return
        return _getTensorGrid(self, 'Fy')

    @property
    def gridFz(self):
        """Face staggered grid in the z direction."""
        if self.nFz == 0 or self.dim < 3: return
        return _getTensorGrid(self, 'Fz')

    @property
    def gridEx(self):
        """Edge staggered grid in the x direction."""
        if self.nEx == 0: return
        return _getTensorGrid(self, 'Ex')

    @property
    def gridEy(self):
        """Edge staggered grid in the y direction."""
        if self.nEy == 0 or self.dim < 2: return
        return _getTensorGrid(self, 'Ey')

    @property
    def gridEz(self):
        """Edge staggered grid in the z direction."""
        if self.nEz == 0 or self.dim < 3: return
        return _getTensorGrid(self, 'Ez')

    def getTensor(self, locType):
        """ Returns a tensor list.

        :param str locType: What tensor (see below)
        :rtype: list
        :return: list of the tensors that make up the mesh.

        locType can be::

            'Ex'    -> x-component of field defined on edges
            'Ey'    -> y-component of field defined on edges
            'Ez'    -> z-component of field defined on edges
            'Fx'    -> x-component of field defined on faces
            'Fy'    -> y-component of field defined on faces
            'Fz'    -> z-component of field defined on faces
            'N'     -> scalar field defined on nodes
            'CC'    -> scalar field defined on cell centers
        """

        if   locType is 'Fx':
            ten = [self.vectorNx , self.vectorCCy, self.vectorCCz]
        elif locType is 'Fy':
            ten = [self.vectorCCx, self.vectorNy , self.vectorCCz]
        elif locType is 'Fz':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorNz ]
        elif locType is 'Ex':
            ten = [self.vectorCCx, self.vectorNy , self.vectorNz ]
        elif locType is 'Ey':
            ten = [self.vectorNx , self.vectorCCy, self.vectorNz ]
        elif locType is 'Ez':
            ten = [self.vectorNx , self.vectorNy , self.vectorCCz]
        elif locType is 'CC':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorCCz]
        elif locType is 'N':
            ten = [self.vectorNx , self.vectorNy , self.vectorNz ]

        return [t for t in ten if t is not None]


class TensorMesh(BaseTensorMesh, TensorView, DiffOperators, InnerProducts):
    """
    TensorMesh is a mesh class that deals with tensor product meshes.

    Any Mesh that has a constant width along the entire axis
    such that it can defined by a single width vector, called 'h'.

    ::

        hx = np.array([1,1,1])
        hy = np.array([1,2])
        hz = np.array([1,1,1,1])

        mesh = Mesh.TensorMesh([hx, hy, hz])

    Example of a padded tensor mesh:

    .. plot::

        from SimPEG import Mesh, Utils
        M = Mesh.TensorMesh(Utils.meshTensors(((10,10),(40,10),(10,10)), ((10,10),(20,10),(0,0))))
        M.plotGrid()

    For a quick tensor mesh on a (10x12x15) unit cube::

        mesh = Mesh.TensorMesh([10, 12, 15])

    """

    __metaclass__ = Utils.SimPEGMetaClass

    _meshType = 'TENSOR'

    def __init__(self, h_in, x0=None):
        BaseTensorMesh.__init__(self, h_in, x0)

    def __str__(self):
        outStr = '  ---- {0:d}-D TensorMesh ----  '.format(self.dim)
        def printH(hx, outStr=''):
            i = -1
            while True:
                i = i + 1
                if i > hx.size:
                    break
                elif i == hx.size:
                    break
                h = hx[i]
                n = 1
                for j in range(i+1, hx.size):
                    if hx[j] == h:
                        n = n + 1
                        i = i + 1
                    else:
                        break

                if n == 1:
                    outStr = outStr + ' {0:.2f},'.format(h)
                else:
                    outStr = outStr + ' {0:d}*{1:.2f},'.format(n,h)

            return outStr[:-1]

        if self.dim == 1:
            outStr = outStr + '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr = outStr + '\n  nCx: {0:d}'.format(self.nCx)
            outStr = outStr + printH(self.hx, outStr='\n   hx:')
            pass
        elif self.dim == 2:
            outStr = outStr + '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr = outStr + '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr = outStr + '\n  nCx: {0:d}'.format(self.nCx)
            outStr = outStr + '\n  nCy: {0:d}'.format(self.nCy)
            outStr = outStr + printH(self.hx, outStr='\n   hx:')
            outStr = outStr + printH(self.hy, outStr='\n   hy:')
        elif self.dim == 3:
            outStr = outStr + '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr = outStr + '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr = outStr + '\n   z0: {0:.2f}'.format(self.x0[2])
            outStr = outStr + '\n  nCx: {0:d}'.format(self.nCx)
            outStr = outStr + '\n  nCy: {0:d}'.format(self.nCy)
            outStr = outStr + '\n  nCz: {0:d}'.format(self.nCz)
            outStr = outStr + printH(self.hx, outStr='\n   hx:')
            outStr = outStr + printH(self.hy, outStr='\n   hy:')
            outStr = outStr + printH(self.hz, outStr='\n   hz:')

        return outStr


    # --------------- Geometries ---------------------
    @property
    def vol(self):
        """Construct cell volumes of the 3D model as 1d array."""
        if getattr(self, '_vol', None) is None:
            vh = self.h
            # Compute cell volumes
            if self.dim == 1:
                self._vol = Utils.mkvc(vh[0])
            elif self.dim == 2:
                # Cell sizes in each direction
                self._vol = Utils.mkvc(np.outer(vh[0], vh[1]))
            elif self.dim == 3:
                # Cell sizes in each direction
                self._vol = Utils.mkvc(np.outer(Utils.mkvc(np.outer(vh[0], vh[1])), vh[2]))
        return self._vol

    @property
    def area(self):
        """Construct face areas of the 3D model as 1d array."""
        if getattr(self, '_area', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if(self.dim == 1):
                self._area = np.ones(n[0]+1)
            elif(self.dim == 2):
                area1 = np.outer(np.ones(n[0]+1), vh[1])
                area2 = np.outer(vh[0], np.ones(n[1]+1))
                self._area = np.r_[Utils.mkvc(area1), Utils.mkvc(area2)]
            elif(self.dim == 3):
                area1 = np.outer(np.ones(n[0]+1), Utils.mkvc(np.outer(vh[1], vh[2])))
                area2 = np.outer(vh[0], Utils.mkvc(np.outer(np.ones(n[1]+1), vh[2])))
                area3 = np.outer(vh[0], Utils.mkvc(np.outer(vh[1], np.ones(n[2]+1))))
                self._area = np.r_[Utils.mkvc(area1), Utils.mkvc(area2), Utils.mkvc(area3)]
        return self._area

    @property
    def edge(self):
        """Construct edge legnths of the 3D model as 1d array."""
        if getattr(self, '_edge', None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if(self.dim == 1):
                self._edge = Utils.mkvc(vh[0])
            elif(self.dim == 2):
                l1 = np.outer(vh[0], np.ones(n[1]+1))
                l2 = np.outer(np.ones(n[0]+1), vh[1])
                self._edge = np.r_[Utils.mkvc(l1), Utils.mkvc(l2)]
            elif(self.dim == 3):
                l1 = np.outer(vh[0], Utils.mkvc(np.outer(np.ones(n[1]+1), np.ones(n[2]+1))))
                l2 = np.outer(np.ones(n[0]+1), Utils.mkvc(np.outer(vh[1], np.ones(n[2]+1))))
                l3 = np.outer(np.ones(n[0]+1), Utils.mkvc(np.outer(np.ones(n[1]+1), vh[2])))
                self._edge = np.r_[Utils.mkvc(l1), Utils.mkvc(l2), Utils.mkvc(l3)]
        return self._edge

    # --------------- Methods ---------------------

    def isInside(self, pts, locType='N'):
        """
        Determines if a set of points are inside a mesh.

        :param numpy.ndarray pts: Location of points to test
        :rtype numpy.ndarray
        :return inside, numpy array of booleans
        """

        tensors = self.getTensor(locType)
        if type(pts) == list:
            pts = np.array(pts)
        assert type(pts) == np.ndarray, "must be a numpy array"
        if self.dim > 1:
            assert pts.shape[1] == self.dim, "must be a column vector of shape (nPts, mesh.dim)"
        elif len(pts.shape) == 1:
            pts = pts[:,np.newaxis]
        else:
            assert pts.shape[1] == self.dim, "must be a column vector of shape (nPts, mesh.dim)"

        inside = np.ones(pts.shape[0],dtype=bool)
        for i, tensor in enumerate(tensors):
            inside = inside & (pts[:,i] >= tensor.min()) & (pts[:,i] <= tensor.max())
        return inside

    def getInterpolationMat(self, loc, locType, zerosOutside=False):
        """ Produces interpolation matrix

        :param numpy.ndarray loc: Location of points to interpolate to
        :param str locType: What to interpolate (see below)
        :rtype: scipy.sparse.csr.csr_matrix
        :return: M, the interpolation matrix

        locType can be::

            'Ex'    -> x-component of field defined on edges
            'Ey'    -> y-component of field defined on edges
            'Ez'    -> z-component of field defined on edges
            'Fx'    -> x-component of field defined on faces
            'Fy'    -> y-component of field defined on faces
            'Fz'    -> z-component of field defined on faces
            'N'     -> scalar field defined on nodes
            'CC'    -> scalar field defined on cell centers
        """

        if type(loc) == list:
            loc = np.array(loc)
        assert type(loc) == np.ndarray, "must be a numpy array"
        if self.dim > 1:
            assert loc.shape[1] == self.dim, "must be a column vector of shape (nPts, mesh.dim)"
        elif len(loc.shape) == 1:
            loc = loc[:,np.newaxis]
        else:
            assert loc.shape[1] == self.dim, "must be a column vector of shape (nPts, mesh.dim)"

        if zerosOutside is False:
            assert np.all(self.isInside(loc)), "Points outside of mesh"
        else:
            indZeros = np.logical_not(self.isInside(loc))
            loc[indZeros, :] = np.array([v.mean() for v in self.getTensor('CC')])

        ind = 0 if 'x' in locType else 1 if 'y' in locType else 2 if 'z' in locType else -1
        if locType in ['Fx','Fy','Fz','Ex','Ey','Ez'] and self.dim >= ind:
            nF_nE = self.vnF if 'F' in locType else self.vnE
            components = [Utils.spzeros(loc.shape[0], n) for n in nF_nE]
            components[ind] = Utils.interpmat(loc, *self.getTensor(locType))
            Q = sp.hstack(components)
        elif locType in ['CC', 'N']:
            Q = Utils.interpmat(loc, *self.getTensor(locType))
        else:
            raise NotImplementedError('getInterpolationMat: locType=='+locType+' and mesh.dim=='+str(self.dim))
        if zerosOutside:
            Q[indZeros, :] = 0
        return Q.tocsr()

if __name__ == '__main__':
    print('Welcome to tensor mesh!')

    testDim = 1
    h1 = 0.3*np.ones(7)
    h1[0] = 0.5
    h1[-1] = 0.6
    h2 = .5 * np.ones(4)
    h3 = .4 * np.ones(6)

    h = [h1, h2, h3]
    h = h[:testDim]

    M = TensorMesh(h)
    print M

    xn = M.plotGrid()
