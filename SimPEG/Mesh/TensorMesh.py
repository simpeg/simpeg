from __future__ import print_function
from SimPEG import Utils, np, sp
from .BaseMesh import BaseMesh, BaseRectangularMesh
from .View import TensorView
from .DiffOperators import DiffOperators
from .InnerProducts import InnerProducts
from .MeshIO import TensorMeshIO
import warnings


class BaseTensorMesh(BaseMesh):

    _meshType = 'BASETENSOR'

    _unitDimensions = [1, 1, 1]

    def __init__(self, h_in, x0_in=None):
        assert type(h_in) in [list, tuple], 'h_in must be a list'
        assert len(h_in) in [1,2,3], 'h_in must be of dimension 1, 2, or 3'
        h = list(range(len(h_in)))
        for i, h_i in enumerate(h_in):
            if Utils.isScalar(h_i) and type(h_i) is not np.ndarray:
                # This gives you something over the unit cube.
                h_i = self._unitDimensions[i] * np.ones(int(h_i))/int(h_i)
            elif type(h_i) is list:
                h_i = Utils.meshTensor(h_i)
            assert isinstance(h_i, np.ndarray), ("h[{0:d}] is not a numpy array.".format(i))
            assert len(h_i.shape) == 1, ("h[{0:d}] must be a 1D numpy array.".format(i))
            h[i] = h_i[:] # make a copy.

        x0 = np.zeros(len(h))
        if x0_in is not None:
            assert len(h) == len(x0_in), "Dimension mismatch. x0 != len(h)"
            for i in range(len(h)):
                x_i, h_i = x0_in[i], h[i]
                if Utils.isScalar(x_i):
                    x0[i] = x_i
                elif x_i == '0':
                    x0[i] = 0.0
                elif x_i == 'C':
                    x0[i] = -h_i.sum()*0.5
                elif x_i == 'N':
                    x0[i] = -h_i.sum()
                else:
                    raise Exception("x0[{0:d}] must be a scalar or '0' to be zero, 'C' to center, or 'N' to be negative.".format(i))

        if isinstance(self, BaseRectangularMesh):
            BaseRectangularMesh.__init__(self, np.array([x.size for x in h]), x0)
        else:
            BaseMesh.__init__(self, np.array([x.size for x in h]), x0)

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
        return self._getTensorGrid('CC')

    @property
    def gridN(self):
        """Nodal grid."""
        return self._getTensorGrid('N')

    @property
    def gridFx(self):
        """Face staggered grid in the x direction."""
        if self.nFx == 0: return
        return self._getTensorGrid('Fx')

    @property
    def gridFy(self):
        """Face staggered grid in the y direction."""
        if self.nFy == 0 or self.dim < 2: return
        return self._getTensorGrid('Fy')

    @property
    def gridFz(self):
        """Face staggered grid in the z direction."""
        if self.nFz == 0 or self.dim < 3: return
        return self._getTensorGrid('Fz')

    @property
    def gridEx(self):
        """Edge staggered grid in the x direction."""
        if self.nEx == 0: return
        return self._getTensorGrid('Ex')

    @property
    def gridEy(self):
        """Edge staggered grid in the y direction."""
        if self.nEy == 0 or self.dim < 2: return
        return self._getTensorGrid('Ey')

    @property
    def gridEz(self):
        """Edge staggered grid in the z direction."""
        if self.nEz == 0 or self.dim < 3: return
        return self._getTensorGrid('Ez')

    def _getTensorGrid(self, key):
        if getattr(self, '_grid' + key, None) is None:
            setattr(self, '_grid' + key, Utils.ndgrid(self.getTensor(key)))
        return getattr(self, '_grid' + key)

    def getTensor(self, key):
        """ Returns a tensor list.

        :param str key: What tensor (see below)
        :rtype: list
        :return: list of the tensors that make up the mesh.

        key can be::

            'CC'    -> scalar field defined on cell centers
            'N'     -> scalar field defined on nodes
            'Fx'    -> x-component of field defined on faces
            'Fy'    -> y-component of field defined on faces
            'Fz'    -> z-component of field defined on faces
            'Ex'    -> x-component of field defined on edges
            'Ey'    -> y-component of field defined on edges
            'Ez'    -> z-component of field defined on edges

        """

        if   key == 'Fx':
            ten = [self.vectorNx , self.vectorCCy, self.vectorCCz]
        elif key == 'Fy':
            ten = [self.vectorCCx, self.vectorNy , self.vectorCCz]
        elif key == 'Fz':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorNz ]
        elif key == 'Ex':
            ten = [self.vectorCCx, self.vectorNy , self.vectorNz ]
        elif key == 'Ey':
            ten = [self.vectorNx , self.vectorCCy, self.vectorNz ]
        elif key == 'Ez':
            ten = [self.vectorNx , self.vectorNy , self.vectorCCz]
        elif key == 'CC':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorCCz]
        elif key == 'N':
            ten = [self.vectorNx , self.vectorNy , self.vectorNz ]

        return [t for t in ten if t is not None]

    # --------------- Methods ---------------------

    def isInside(self, pts, locType='N'):
        """
        Determines if a set of points are inside a mesh.

        :param numpy.ndarray pts: Location of points to test
        :rtype: numpy.ndarray
        :return: inside, numpy array of booleans
        """
        pts = Utils.asArray_N_x_Dim(pts, self.dim)

        tensors = self.getTensor(locType)

        if locType == 'N' and self._meshType == 'CYL':
            #NOTE: for a CYL mesh we add a node to check if we are inside in the radial direction!
            tensors[0] = np.r_[0.,tensors[0]]
            tensors[1] = np.r_[tensors[1], 2.0*np.pi]

        inside = np.ones(pts.shape[0],dtype=bool)
        for i, tensor in enumerate(tensors):
            TOL = np.diff(tensor).min() * 1.0e-10
            inside = inside & (pts[:,i] >= tensor.min()-TOL) & (pts[:,i] <= tensor.max()+TOL)
        return inside

    def getInterpolationMat(self, loc, locType='CC', zerosOutside=False):
        """ Produces interpolation matrix

        :param numpy.ndarray loc: Location of points to interpolate to
        :param str locType: What to interpolate (see below)
        :rtype: scipy.sparse.csr_matrix
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
            'CCVx'  -> x-component of vector field defined on cell centers
            'CCVy'  -> y-component of vector field defined on cell centers
            'CCVz'  -> z-component of vector field defined on cell centers
        """
        if self._meshType == 'CYL' and self.isSymmetric and locType in ['Ex','Ez','Fy']:
            raise Exception('Symmetric CylMesh does not support {0!s} interpolation, as this variable does not exist.'.format(locType))

        loc = Utils.asArray_N_x_Dim(loc, self.dim)

        if zerosOutside is False:
            assert np.all(self.isInside(loc)), "Points outside of mesh"
        else:
            indZeros = np.logical_not(self.isInside(loc))
            loc[indZeros, :] = np.array([v.mean() for v in self.getTensor('CC')])

        if locType in ['Fx','Fy','Fz','Ex','Ey','Ez']:
            ind = {'x':0, 'y':1, 'z':2}[locType[1]]
            assert self.dim >= ind, 'mesh is not high enough dimension.'
            nF_nE = self.vnF if 'F' in locType else self.vnE
            components = [Utils.spzeros(loc.shape[0], n) for n in nF_nE]
            components[ind] = Utils.interpmat(loc, *self.getTensor(locType))
            # remove any zero blocks (hstack complains)
            components = [comp for comp in components if comp.shape[1] > 0]
            Q = sp.hstack(components)
        elif locType in ['CC', 'N']:
            Q = Utils.interpmat(loc, *self.getTensor(locType))
        elif locType in ['CCVx', 'CCVy', 'CCVz']:
            Q = Utils.interpmat(loc, *self.getTensor('CC'))
            Z = Utils.spzeros(loc.shape[0],self.nC)
            if locType == 'CCVx':
                Q = sp.hstack([Q,Z,Z])
            elif locType == 'CCVy':
                Q = sp.hstack([Z,Q,Z])
            elif locType == 'CCVz':
                Q = sp.hstack([Z,Z,Q])

        else:
            raise NotImplementedError('getInterpolationMat: locType=='+locType+' and mesh.dim=='+str(self.dim))

        if zerosOutside:
            Q[indZeros, :] = 0

        return Q.tocsr()


    def _fastInnerProduct(self, projType, prop=None, invProp=False, invMat=False):
        """
            Fast version of getFaceInnerProduct.
            This does not handle the case of a full tensor prop.

            :param numpy.array prop: material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
            :param str projType: 'E' or 'F'
            :param bool returnP: returns the projection matrices
            :param bool invProp: inverts the material property
            :param bool invMat: inverts the matrix
            :rtype: scipy.sparse.csr_matrix
            :return: M, the inner product matrix (nF, nF)
        """
        assert projType in ['F', 'E'], ("projType must be 'F' for faces or 'E'"
                                        " for edges")

        if prop is None:
            prop = np.ones(self.nC)

        if invProp:
            prop = 1./prop

        if Utils.isScalar(prop):
            prop = prop*np.ones(self.nC)

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == 'CYL':
            n_elements = np.sum(getattr(self, 'vn'+projType).nonzero())
        else:
            n_elements = self.dim

        # Isotropic? or anisotropic?
        if prop.size == self.nC:
            Av = getattr(self, 'ave'+projType+'2CC')
            Vprop = self.vol * Utils.mkvc(prop)
            M = n_elements * Utils.sdiag(Av.T * Vprop)

        elif prop.size == self.nC*self.dim:
            Av = getattr(self, 'ave'+projType+'2CCV')

            # if cyl, then only certain components are relevant due to symmetry
            # for faces, x, z matters, for edges, y (which is theta) matters
            if self._meshType == 'CYL':
                if projType == 'E':
                    prop = prop[:, 1] # this is the action of a projection mat
                elif projType == 'F':
                    prop = prop[:, [0, 2]]

            V = sp.kron(sp.identity(n_elements), Utils.sdiag(self.vol))
            M = Utils.sdiag(Av.T * V * Utils.mkvc(prop))
        else:
            return None

        if invMat:
            return Utils.sdInv(M)
        else:
            return M

    def _fastInnerProductDeriv(self, projType, prop, invProp=False,
                               invMat=False):
        """
            :param str projType: 'E' or 'F'
            :param TensorType tensorType: type of the tensor
            :param bool invProp: inverts the material property
            :param bool invMat: inverts the matrix
            :rtype: function
            :return: dMdmu, the derivative of the inner product matrix
        """
        assert projType in ['F', 'E'], ("projType must be 'F' for faces or 'E'"
                                        " for edges")

        tensorType = Utils.TensorType(self, prop)

        dMdprop = None

        if invMat or invProp:
            MI = self._fastInnerProduct(projType, prop, invProp=invProp,
                                        invMat=invMat)

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == 'CYL':
            n_elements = np.sum(getattr(self, 'vn'+projType).nonzero())
        else:
            n_elements = self.dim


        if tensorType == 0:  # isotropic, constant
            Av = getattr(self, 'ave'+projType+'2CC')
            V = Utils.sdiag(self.vol)
            ones = sp.csr_matrix((np.ones(self.nC), (range(self.nC),
                                                     np.zeros(self.nC))),
                                 shape=(self.nC, 1))
            if not invMat and not invProp:
                dMdprop = n_elements * Av.T * V * ones
            elif invMat and invProp:
                dMdprop =  n_elements * (Utils.sdiag(MI.diagonal()**2) * Av.T *
                                         V * ones * Utils.sdiag(1./prop**2))
            elif invProp:
                dMdprop = n_elements * Av.T * V * Utils.sdiag(- 1./prop**2)
            elif invMat:
                dMdprop = n_elements * (Utils.sdiag(- MI.diagonal()**2) * Av.T
                                        * V)

        elif tensorType == 1:  # isotropic, variable in space
            Av = getattr(self, 'ave'+projType+'2CC')
            V = Utils.sdiag(self.vol)
            if not invMat and not invProp:
                dMdprop = n_elements * Av.T * V
            elif invMat and invProp:
                dMdprop =  n_elements * (Utils.sdiag(MI.diagonal()**2) * Av.T *
                                         V * Utils.sdiag(1./prop**2))
            elif invProp:
                dMdprop = n_elements * Av.T * V * Utils.sdiag(-1./prop**2)
            elif invMat:
                dMdprop = n_elements * (Utils.sdiag(- MI.diagonal()**2) * Av.T
                                        * V)

        elif tensorType == 2: # anisotropic
            Av = getattr(self, 'ave'+projType+'2CCV')
            V = sp.kron(sp.identity(self.dim), Utils.sdiag(self.vol))

            if self._meshType == 'CYL':
                Zero = sp.csr_matrix((self.nC, self.nC))
                Eye = sp.eye(self.nC)
                if projType == 'E':
                    P = sp.hstack([Zero, Eye, Zero])
                    # print(P.todense())
                elif projType == 'F':
                    P = sp.vstack([sp.hstack([Eye, Zero, Zero]),
                                   sp.hstack([Zero, Zero, Eye])])
                    # print(P.todense())
            else:
                P = sp.eye(self.nC*self.dim)

            if not invMat and not invProp:
                dMdprop = Av.T * P * V
            elif invMat and invProp:
                dMdprop = (Utils.sdiag(MI.diagonal()**2) * Av.T * P * V *
                           Utils.sdiag(1./prop**2))
            elif invProp:
                dMdprop = Av.T * P * V * Utils.sdiag(-1./prop**2)
            elif invMat:
                dMdprop = Utils.sdiag(- MI.diagonal()**2) * Av.T * P * V

        if dMdprop is not None:
            def innerProductDeriv(v=None):
                if v is None:
                    warnings.warn("Depreciation Warning: "
                                  "TensorMesh.innerProductDeriv."
                                  " You should be supplying a vector. "
                                  "Use: sdiag(u)*dMdprop", FutureWarning)
                    return dMdprop
                return Utils.sdiag(v) * dMdprop
            return innerProductDeriv
        else:
            return None


class TensorMesh(BaseTensorMesh, BaseRectangularMesh, TensorView, DiffOperators,
      InnerProducts, TensorMeshIO):
    """
    TensorMesh is a mesh class that deals with tensor product meshes.

    Any Mesh that has a constant width along the entire axis
    such that it can defined by a single width vector, called 'h'.

    ::

        hx = np.array([1,1,1])
        hy = np.array([1,2])
        hz = np.array([1,1,1,1])

        mesh = Mesh.TensorMesh([hx, hy, hz])

    Example of a padded tensor mesh using :func:`SimPEG.Utils.meshutils.meshTensor`:

    .. plot::
        :include-source:

        from SimPEG import Mesh, Utils
        M = Mesh.TensorMesh([[(10,10,-1.3),(10,40),(10,10,1.3)], [(10,10,-1.3),(10,20)]])
        M.plotGrid()

    For a quick tensor mesh on a (10x12x15) unit cube::

        mesh = Mesh.TensorMesh([10, 12, 15])

    """

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
                    outStr += ' {0:.2f},'.format(h)
                else:
                    outStr += ' {0:d}*{1:.2f},'.format(n,h)

            return outStr[:-1]

        if self.dim == 1:
            outStr += '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr += '\n  nCx: {0:d}'.format(self.nCx)
            outStr += printH(self.hx, outStr='\n   hx:')
            pass
        elif self.dim == 2:
            outStr += '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr += '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr += '\n  nCx: {0:d}'.format(self.nCx)
            outStr += '\n  nCy: {0:d}'.format(self.nCy)
            outStr += printH(self.hx, outStr='\n   hx:')
            outStr += printH(self.hy, outStr='\n   hy:')
        elif self.dim == 3:
            outStr += '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr += '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr += '\n   z0: {0:.2f}'.format(self.x0[2])
            outStr += '\n  nCx: {0:d}'.format(self.nCx)
            outStr += '\n  nCy: {0:d}'.format(self.nCy)
            outStr += '\n  nCz: {0:d}'.format(self.nCz)
            outStr += printH(self.hx, outStr='\n   hx:')
            outStr += printH(self.hy, outStr='\n   hy:')
            outStr += printH(self.hz, outStr='\n   hz:')

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

    @property
    def faceBoundaryInd(self):
        """
            Find indices of boundary faces in each direction
        """
        if self.dim==1:
            indxd = (self.gridFx==min(self.gridFx))
            indxu = (self.gridFx==max(self.gridFx))
            return indxd, indxu
        elif self.dim==2:
            indxd = (self.gridFx[:,0]==min(self.gridFx[:,0]))
            indxu = (self.gridFx[:,0]==max(self.gridFx[:,0]))
            indyd = (self.gridFy[:,1]==min(self.gridFy[:,1]))
            indyu = (self.gridFy[:,1]==max(self.gridFy[:,1]))
            return indxd, indxu, indyd, indyu
        elif self.dim==3:
            indxd = (self.gridFx[:,0]==min(self.gridFx[:,0]))
            indxu = (self.gridFx[:,0]==max(self.gridFx[:,0]))
            indyd = (self.gridFy[:,1]==min(self.gridFy[:,1]))
            indyu = (self.gridFy[:,1]==max(self.gridFy[:,1]))
            indzd = (self.gridFz[:,2]==min(self.gridFz[:,2]))
            indzu = (self.gridFz[:,2]==max(self.gridFz[:,2]))
            return indxd, indxu, indyd, indyu, indzd, indzu

    @property
    def cellBoundaryInd(self):
        """
            Find indices of boundary faces in each direction
        """
        if self.dim==1:
            indxd = (self.gridCC==min(self.gridCC))
            indxu = (self.gridCC==max(self.gridCC))
            return indxd, indxu
        elif self.dim==2:
            indxd = (self.gridCC[:,0]==min(self.gridCC[:,0]))
            indxu = (self.gridCC[:,0]==max(self.gridCC[:,0]))
            indyd = (self.gridCC[:,1]==min(self.gridCC[:,1]))
            indyu = (self.gridCC[:,1]==max(self.gridCC[:,1]))
            return indxd, indxu, indyd, indyu
        elif self.dim==3:
            indxd = (self.gridCC[:,0]==min(self.gridCC[:,0]))
            indxu = (self.gridCC[:,0]==max(self.gridCC[:,0]))
            indyd = (self.gridCC[:,1]==min(self.gridCC[:,1]))
            indyu = (self.gridCC[:,1]==max(self.gridCC[:,1]))
            indzd = (self.gridCC[:,2]==min(self.gridCC[:,2]))
            indzu = (self.gridCC[:,2]==max(self.gridCC[:,2]))
            return indxd, indxu, indyd, indyu, indzd, indzu
