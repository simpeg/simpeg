import Utils, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from Tests import checkDerivative
from PropMaps import PropMap, Property
from numpy.polynomial import polynomial
from scipy.interpolate import UnivariateSpline

class IdentityMap(object):
    """
    SimPEG Map

    """
    __metaclass__ = Utils.SimPEGMetaClass

    def __init__(self, mesh=None, nP=None, **kwargs):
        Utils.setKwargs(self, **kwargs)

        if nP is not None:
            assert type(nP) in [int, long], ' Number of parameters must be an integer.'

        self.mesh = mesh
        self._nP  = nP

    @property
    def nP(self):
        """
            :rtype: int
            :return: number of parameters that the mapping accepts
        """
        if self._nP is not None:
            return self._nP
        if self.mesh is None:
            return '*'
        return self.mesh.nC

    @property
    def shape(self):
        """
            The default shape is (mesh.nC, nP) if the mesh is defined.
            If this is a meshless mapping (i.e. nP is defined independently)
            the shape will be the the shape (nP,nP).

            :rtype: (int,int)
            :return: shape of the operator as a tuple
        """
        if self._nP is not None:
            return (self.nP, self.nP)
        if self.mesh is None:
            return ('*', self.nP)
        return (self.mesh.nC, self.nP)


    def _transform(self, m):
        """
            Changes the model into the physical property.

            .. note::

                This can be called by the __mul__ property against a numpy.ndarray.

            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

        """
        return m

    def inverse(self, D):
        """
            Changes the physical property into the model.

            .. note::

                The *transformInverse* may not be easy to create in general.

            :param numpy.array D: physical property
            :rtype: numpy.array
            :return: model

        """
        raise NotImplementedError('The transformInverse is not implemented.')

    def deriv(self, m):
        """
            The derivative of the transformation.

            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

        """
        return sp.identity(self.nP)

    def test(self, m=None, **kwargs):
        """Test the derivative of the mapping.

            :param numpy.array m: model
            :param kwargs: key word arguments of :meth:`SimPEG.Tests.checkDerivative`
            :rtype: bool
            :return: passed the test?

        """
        print 'Testing %s' % str(self)
        if m is None:
            m = abs(np.random.rand(self.nP))
        if 'plotIt' not in kwargs:
            kwargs['plotIt'] = False
        return checkDerivative(lambda m : [self * m, self.deriv(m)], m, num=4, **kwargs)

    def _assertMatchesPair(self, pair):
        assert (isinstance(self, pair) or
            isinstance(self, ComboMap) and isinstance(self.maps[0], pair)
            ), "Mapping object must be an instance of a %s class."%(pair.__name__)

    def __mul__(self, val):
        if isinstance(val, IdentityMap):
            if not (self.shape[1] == '*' or val.shape[0] == '*') and not self.shape[1] == val.shape[0]:
                raise ValueError('Dimension mismatch in %s and %s.' % (str(self), str(val)))
            return ComboMap([self, val])
        elif isinstance(val, np.ndarray):
            if not self.shape[1] == '*' and not self.shape[1] == val.shape[0]:
                raise ValueError('Dimension mismatch in %s and np.ndarray%s.' % (str(self), str(val.shape)))
            return self._transform(val)
        raise Exception('Unrecognized data type to multiply. Try a map or a numpy.ndarray!')

    def __str__(self):
        return "%s(%s,%s)" % (self.__class__.__name__, self.shape[0], self.shape[1])


class ComboMap(IdentityMap):
    """Combination of various maps."""

    def __init__(self, maps, **kwargs):
        IdentityMap.__init__(self, None, **kwargs)

        self.maps = []
        for ii, m in enumerate(maps):
            assert isinstance(m, IdentityMap), 'Unrecognized data type, inherit from an IdentityMap or ComboMap!'
            if ii > 0 and not (self.shape[1] == '*' or m.shape[0] == '*') and not self.shape[1] == m.shape[0]:
                prev = self.maps[-1]
                errArgs = (prev.__class__.__name__, prev.shape[0], prev.shape[1], m.__class__.__name__, m.shape[0], m.shape[1])
                raise ValueError('Dimension mismatch in map[%s] (%s, %s) and map[%s] (%s, %s).' % errArgs)

            if isinstance(m, ComboMap):
                self.maps += m.maps
            elif isinstance(m, IdentityMap):
                self.maps += [m]

    @property
    def shape(self):
        return (self.maps[0].shape[0], self.maps[-1].shape[1])

    @property
    def nP(self):
        """Number of model properties.

           The number of cells in the
           last dimension of the mesh."""
        return self.maps[-1].nP

    def _transform(self, m):
        for map_i in reversed(self.maps):
            m = map_i * m
        return m

    def deriv(self, m):
        deriv = 1
        mi = m
        for map_i in reversed(self.maps):
            deriv = map_i.deriv(mi) * deriv
            mi = map_i * mi
        return deriv

    def __str__(self):
        return 'ComboMap[%s](%s,%s)' % (' * '.join([m.__str__() for m in self.maps]), self.shape[0], self.shape[1])


class ExpMap(IdentityMap):
    """
        Changes the model into the physical property.

        A common example of this is to invert for electrical conductivity
        in log space. In this case, your model will be log(sigma) and to
        get back to sigma, you can take the exponential:

        .. math::

            m = \log{\sigma}

            \exp{m} = \exp{\log{\sigma}} = \sigma
    """

    def __init__(self, mesh, **kwargs):
        IdentityMap.__init__(self, mesh, **kwargs)

    def _transform(self, m):
        return np.exp(Utils.mkvc(m))

    def inverse(self, D):
        """
            :param numpy.array D: physical property
            :rtype: numpy.array
            :return: model

            The *transformInverse* changes the physical property into the model.

            .. math::

                m = \log{\sigma}

        """
        return np.log(Utils.mkvc(D))


    def deriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The *transform* changes the model into the physical property.
            The *transformDeriv* provides the derivative of the *transform*.

            If the model *transform* is:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma

            Then the derivative is:

            .. math::

                \\frac{\partial \exp{m}}{\partial m} = \\text{sdiag}(\exp{m})
        """
        return Utils.sdiag(np.exp(Utils.mkvc(m)))

class ReciprocalMap(IdentityMap):
    """
        Reciprocal mapping. For example, electrical resistivity and conductivity.

        .. math::

            \\rho = \\frac{1}{\sigma}

    """
    def _transform(self, m):
        return 1.0 / Utils.mkvc(m)

    def inverse(self, D):
        return 1.0 / Utils.mkvc(m)

    def deriv(self, m):
        # TODO: if this is a tensor, you might have a problem.
        return Utils.sdiag( - Utils.mkvc(m)**(-2) )



class LogMap(IdentityMap):
    """
        Changes the model into the physical property.

        If \\(p\\) is the physical property and \\(m\\) is the model, then

        ..math::

            p = \\log(m)

        and

        ..math::

            m = \\exp(p)

        NOTE: If you have a model which is log conductivity (ie. \\(m = \\log(\\sigma)\\)),
        you should be using an ExpMap

    """

    def __init__(self, mesh, **kwargs):
        IdentityMap.__init__(self, mesh, **kwargs)

    def _transform(self, m):
        return np.log(Utils.mkvc(m))

    def deriv(self, m):
        mod = Utils.mkvc(m)
        deriv = np.zeros(mod.shape)
        tol = 1e-16 # zero
        ind = np.greater_equal(np.abs(mod),tol)
        deriv[ind] = 1.0/mod[ind]
        return Utils.sdiag(deriv)

    def inverse(self, m):
        return np.exp(Utils.mkvc(m))

class FullMap(IdentityMap):
    """
    FullMap

    Given a scalar, the FullMap maps the value to the
    full model space.
    """

    def __init__(self,mesh,**kwargs):
        IdentityMap.__init__(self, mesh,**kwargs)

    @property
    def nP(self):
        return 1

    def _transform(self, m):
        """
            :param m: model (scalar)
            :rtype: numpy.array
            :return: transformed model
        """
        return np.ones(self.mesh.nC)*m

    def deriv(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: derivative of transformed model
        """
        return np.ones([self.mesh.nC,1])


class Vertical1DMap(IdentityMap):
    """Vertical1DMap

        Given a 1D vector through the last dimension
        of the mesh, this will extend to the full
        model space.
    """

    def __init__(self, mesh, **kwargs):
        IdentityMap.__init__(self, mesh, **kwargs)

    @property
    def nP(self):
        """Number of model properties.

           The number of cells in the
           last dimension of the mesh."""
        return self.mesh.vnC[self.mesh.dim-1]

    def _transform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model
        """
        repNum = self.mesh.vnC[:self.mesh.dim-1].prod()
        return Utils.mkvc(m).repeat(repNum)

    def deriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model
        """
        repNum = self.mesh.vnC[:self.mesh.dim-1].prod()
        repVec = sp.csr_matrix(
                    (np.ones(repNum),
                    (range(repNum), np.zeros(repNum))
                    ), shape=(repNum, 1))
        return sp.kron(sp.identity(self.nP), repVec)


class Map2Dto3D(IdentityMap):
    """Map2Dto3D

        Given a 2D vector, this will extend to the full
        3D model space.
    """

    normal = 'Y' #: The normal

    def __init__(self, mesh, **kwargs):
        assert mesh.dim == 3, 'Only works for a 3D Mesh'
        IdentityMap.__init__(self, mesh, **kwargs)
        assert self.normal in ['X','Y','Z'], 'For now, only "Y" normal is supported'

    @property
    def nP(self):
        """Number of model properties.

           The number of cells in the
           last dimension of the mesh."""
        if self.normal == 'Z':
            return self.mesh.nCx * self.mesh.nCy
        elif self.normal == 'Y':
            return self.mesh.nCx * self.mesh.nCz
        elif self.normal == 'X':
            return self.mesh.nCy * self.mesh.nCz

    def _transform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model
        """
        m = Utils.mkvc(m)
        if self.normal == 'Z':
            return Utils.mkvc(m.reshape(self.mesh.vnC[[0,1]], order='F')[:,:,np.newaxis].repeat(self.mesh.nCz,axis=2))
        elif self.normal == 'Y':
            return Utils.mkvc(m.reshape(self.mesh.vnC[[0,2]], order='F')[:,np.newaxis,:].repeat(self.mesh.nCy,axis=1))
        elif self.normal == 'X':
            return Utils.mkvc(m.reshape(self.mesh.vnC[[1,2]], order='F')[np.newaxis,:,:].repeat(self.mesh.nCx,axis=0))

    def deriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model
        """
        inds = self * np.arange(self.nP)
        nC, nP = self.mesh.nC, self.nP
        P = sp.csr_matrix(
                    (np.ones(nC),
                    (range(nC), inds)
                ), shape=(nC, nP))
        return P

class Mesh2Mesh(IdentityMap):
    """
        Takes a model on one mesh are translates it to another mesh.

    """

    def __init__(self, meshes, **kwargs):
        Utils.setKwargs(self, **kwargs)

        assert type(meshes) is list, "meshes must be a list of two meshes"
        assert len(meshes) == 2, "meshes must be a list of two meshes"
        assert meshes[0].dim == meshes[1].dim, """The two meshes must be the same dimension"""

        self.mesh  = meshes[0]
        self.mesh2 = meshes[1]

        self.P = self.mesh2.getInterpolationMat(self.mesh.gridCC,'CC',zerosOutside=True)

    @property
    def shape(self):
        """Number of parameters in the model."""
        return (self.mesh.nC, self.mesh2.nC)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh2.nC
    def _transform(self, m):
        return self.P*m
    def deriv(self, m):
        return self.P


class ActiveCells(IdentityMap):
    """
        Active model parameters.

    """

    indActive   = None #: Active Cells
    valInactive = None #: Values of inactive Cells
    nC          = None #: Number of cells in the full model

    def __init__(self, mesh, indActive, valInactive, nC=None):
        self.mesh  = mesh

        self.nC = nC or mesh.nC

        if indActive.dtype is not bool:
            z = np.zeros(self.nC,dtype=bool)
            z[indActive] = True
            indActive = z
        self.indActive = indActive
        self.indInactive = np.logical_not(indActive)
        if Utils.isScalar(valInactive):
            self.valInactive = np.ones(self.nC)*float(valInactive)
        else:
            self.valInactive = valInactive.copy()
        self.valInactive[self.indActive] = 0

        inds = np.nonzero(self.indActive)[0]
        self.P = sp.csr_matrix((np.ones(inds.size),(inds, range(inds.size))), shape=(self.nC, self.nP))

    @property
    def shape(self):
        return (self.nC, self.nP)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.indActive.sum()

    def _transform(self, m):
        return self.P*m + self.valInactive

    def inverse(self, D):
        return self.P.T*D

    def deriv(self, m):
        return self.P

class ActiveCellsTopo(IdentityMap):
    """
        Active model parameters. Extend for cells on topography to air cell (only works for tensor mesh)

    """

    indActive   = None #: Active Cells
    valInactive = None #: Values of inactive Cells
    nC          = None #: Number of cells in the full model

    def __init__(self, mesh, indActive, nC=None):
        self.mesh  = mesh

        self.nC = nC or mesh.nC

        if indActive.dtype is not bool:
            z = np.zeros(self.nC,dtype=bool)
            z[indActive] = True
            indActive = z
        self.indActive = indActive

        self.indInactive = np.logical_not(indActive)
        inds = np.nonzero(self.indActive)[0]
        self.P = sp.csr_matrix((np.ones(inds.size),(inds, range(inds.size))), shape=(self.nC, self.nP))

    @property
    def shape(self):
        return (self.nC, self.nP)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.indActive.sum()

    def _transform(self, m):
        val_temp = np.zeros(self.mesh.nC)
        val_temp[self.indActive] = m
        valInactive = np.zeros(self.mesh.nC)
        #1D
        if self.mesh.dim == 1:
            z_temp = self.mesh.gridCC
            val_temp[~self.indActive] = val_temp[np.argmax(z_temp[self.indActive])]
        #2D
        elif self.mesh.dim == 2:
            act_temp = self.indActive.reshape((self.mesh.nCx, self.mesh.nCy), order = 'F')
            val_temp = val_temp.reshape((self.mesh.nCx, self.mesh.nCy), order = 'F')
            y_temp = self.mesh.gridCC[:,1].reshape((self.mesh.nCx, self.mesh.nCy), order = 'F')
            for i in range(self.mesh.nCx):
                act_tempx = act_temp[i,:] == 1
                val_temp[i,~act_tempx] = val_temp[i,np.argmax(y_temp[i,act_tempx])]
            valInactive[~self.indActive] = Utils.mkvc(val_temp)[~self.indActive]
        #3D
        elif self.mesh.dim == 3:
            act_temp = self.indActive.reshape((self.mesh.nCx*self.mesh.nCy, self.mesh.nCz), order = 'F')
            val_temp = val_temp.reshape((self.mesh.nCx*self.mesh.nCy, self.mesh.nCz), order = 'F')
            z_temp = self.mesh.gridCC[:,2].reshape((self.mesh.nCx*self.mesh.nCy, self.mesh.nCz), order = 'F')
            for i in range(self.mesh.nCx*self.mesh.nCy):
                act_tempxy = act_temp[i,:] == 1
                val_temp[i,~act_tempxy] = val_temp[i,np.argmax(z_temp[i,act_tempxy])]
            valInactive[~self.indActive] = Utils.mkvc(val_temp)[~self.indActive]

        self.valInactive = valInactive

        return self.P*m + self.valInactive

    def inverse(self, D):
        return self.P.T*D

    def deriv(self, m):
        return self.P


class Weighting(IdentityMap):
    """
        Model weight parameters.

    """

    weights     = None #: Active Cells
    nC          = None #: Number of cells in the full model

    def __init__(self, mesh, weights=None, nC=None):
        self.mesh  = mesh

        self.nC = nC or mesh.nC

        if weights is None:
            weights = np.ones(self.nC)

        self.weights = np.array(weights, dtype=float)

        self.P = Utils.sdiag(self.weights)

    @property
    def shape(self):
        return (self.nC, self.nP)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.nC

    def _transform(self, m):
        return self.P*m

    def inverse(self, D):
        Pinv = Utils.sdiag(self.weights**(-1.))
        return Pinv*D

    def deriv(self, m):
        return self.P


class ComplexMap(IdentityMap):
    """ComplexMap

        default nP is nC in the mesh times 2 [real, imag]

    """
    def __init__(self, mesh, nP=None):
        IdentityMap.__init__(self, mesh)
        if nP is not None:
            assert nP%2 == 0, 'nP must be even.'
        self._nP = nP or (self.mesh.nC * 2)

    @property
    def nP(self):
        return self._nP

    @property
    def shape(self):
        return (self.nP/2,self.nP)

    def _transform(self, m):
        nC = self.mesh.nC
        return m[:nC] + m[nC:]*1j

    def deriv(self, m):
        nC = self.nP/2
        shp = (nC, nC*2)
        def fwd(v):
            return v[:nC] + v[nC:]*1j
        def adj(v):
            return np.r_[v.real,v.imag]
        return LinearOperator(shp,matvec=fwd,rmatvec=adj)

    inverse = deriv


class CircleMap(IdentityMap):
    """CircleMap

        Parameterize the model space using a circle in a wholespace.

        ..math::

            \sigma(m) = \sigma_1 + (\sigma_2 - \sigma_1)\left(\\arctan\left(100*\sqrt{(\\vec{x}-x_0)^2 + (\\vec{y}-y_0)}-r\\right) \pi^{-1} + 0.5\\right)

        Define the model as:

        ..math::

            m = [\sigma_1, \sigma_2, x_0, y_0, r]

    """
    def __init__(self, mesh, logSigma=True):
        assert mesh.dim == 2, "Working for a 2D mesh only right now. But it isn't that hard to change.. :)"
        IdentityMap.__init__(self, mesh)
        self.logSigma = logSigma

    slope = 1e-1

    @property
    def nP(self):
        return 5

    def _transform(self, m):
        a = self.slope
        sig1,sig2,x,y,r = m[0],m[1],m[2],m[3],m[4]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        X = self.mesh.gridCC[:,0]
        Y = self.mesh.gridCC[:,1]
        return sig1 + (sig2 - sig1)*(np.arctan(a*(np.sqrt((X-x)**2 + (Y-y)**2) - r))/np.pi + 0.5)

    def deriv(self, m):
        a = self.slope
        sig1,sig2,x,y,r = m[0],m[1],m[2],m[3],m[4]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        X = self.mesh.gridCC[:,0]
        Y = self.mesh.gridCC[:,1]
        if self.logSigma:
            g1 = -(np.arctan(a*(-r + np.sqrt((X - x)**2 + (Y - y)**2)))/np.pi + 0.5)*sig1 + sig1
            g2 = (np.arctan(a*(-r + np.sqrt((X - x)**2 + (Y - y)**2)))/np.pi + 0.5)*sig2
        else:
            g1 = -(np.arctan(a*(-r + np.sqrt((X - x)**2 + (Y - y)**2)))/np.pi + 0.5) + 1.0
            g2 = (np.arctan(a*(-r + np.sqrt((X - x)**2 + (Y - y)**2)))/np.pi + 0.5)
        g3 = a*(-X + x)*(-sig1 + sig2)/(np.pi*(a**2*(-r + np.sqrt((X - x)**2 + (Y - y)**2))**2 + 1)*np.sqrt((X - x)**2 + (Y - y)**2))
        g4 = a*(-Y + y)*(-sig1 + sig2)/(np.pi*(a**2*(-r + np.sqrt((X - x)**2 + (Y - y)**2))**2 + 1)*np.sqrt((X - x)**2 + (Y - y)**2))
        g5 = -a*(-sig1 + sig2)/(np.pi*(a**2*(-r + np.sqrt((X - x)**2 + (Y - y)**2))**2 + 1))
        return sp.csr_matrix(np.c_[g1,g2,g3,g4,g5])


class PolyMap(IdentityMap):

    """PolyMap

        Parameterize the model space using a polynomials in a wholespace.

        ..math::

            y = \mathbf{V} c

        Define the model as:

        ..math::

            m = [\sigma_1, \sigma_2, c]

    """
    def __init__(self, mesh, order, logSigma=True, normal='X'):
        IdentityMap.__init__(self, mesh)
        self.logSigma = logSigma
        self.order = order
        self.normal = normal

    slope = 1e4

    @property
    def nP(self):
        if np.isscalar(self.order):
            nP = self.order+3
        else:
            nP =(self.order[0]+1)*(self.order[1]+1)+2
        return nP

    def _transform(self, m):
        # Set model parameters
        alpha = self.slope
        sig1,sig2 = m[0],m[1]
        c = m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        #2D
        if self.mesh.dim == 2:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]
            if self.normal =='X':
                f = polynomial.polyval(Y, c) - X
            elif self.normal =='Y':
                f = polynomial.polyval(X, c) - Y
            else:
                raise(Exception("Input for normal = X or Y or Z"))
        #3D
        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]
            Z = self.mesh.gridCC[:,2]
            if self.normal =='X':
                f = polynomial.polyval2d(Y, Z, c.reshape((self.order[0]+1,self.order[1]+1))) - X
            elif self.normal =='Y':
                f = polynomial.polyval2d(X, Z, c.reshape((self.order[0]+1,self.order[1]+1))) - Y
            elif self.normal =='Z':
                f = polynomial.polyval2d(X, Y, c.reshape((self.order[0]+1,self.order[1]+1))) - Z
            else:
                raise(Exception("Input for normal = X or Y or Z"))
        else:
            raise(Exception("Only supports 2D"))


        return sig1+(sig2-sig1)*(np.arctan(alpha*f)/np.pi+0.5)

    def deriv(self, m):
        alpha = self.slope
        sig1,sig2, c = m[0],m[1],m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        #2D
        if self.mesh.dim == 2:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]

            if self.normal =='X':
                f = polynomial.polyval(Y, c) - X
                V = polynomial.polyvander(Y, len(c)-1)
            elif self.normal =='Y':
                f = polynomial.polyval(X, c) - Y
                V = polynomial.polyvander(X, len(c)-1)
            else:
                raise(Exception("Input for normal = X or Y or Z"))
        #3D
        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]
            Z = self.mesh.gridCC[:,2]

            if self.normal =='X':
                f = polynomial.polyval2d(Y, Z, c.reshape((self.order[0]+1,self.order[1]+1))) - X
                V = polynomial.polyvander2d(Y, Z, self.order)
            elif self.normal =='Y':
                f = polynomial.polyval2d(X, Z, c.reshape((self.order[0]+1,self.order[1]+1))) - Y
                V = polynomial.polyvander2d(X, Z, self.order)
            elif self.normal =='Z':
                f = polynomial.polyval2d(X, Y, c.reshape((self.order[0]+1,self.order[1]+1))) - Z
                V = polynomial.polyvander2d(X, Y, self.order)
            else:
                raise(Exception("Input for normal = X or Y or Z"))

        if self.logSigma:
            g1 = -(np.arctan(alpha*f)/np.pi + 0.5)*sig1 + sig1
            g2 = (np.arctan(alpha*f)/np.pi + 0.5)*sig2
        else:
            g1 = -(np.arctan(alpha*f)/np.pi + 0.5) + 1.0
            g2 = (np.arctan(alpha*f)/np.pi + 0.5)

        g3 = Utils.sdiag(alpha*(sig2-sig1)/(1.+(alpha*f)**2)/np.pi)*V

        return sp.csr_matrix(np.c_[g1,g2,g3])

class SplineMap(IdentityMap):

    """SplineMap

        Parameterize the boundary of two geological units using a spline interpolation

        ..math::

            g = f(x)-y

        Define the model as:

        ..math::

            m = [\sigma_1, \sigma_2, y]

    """
    def __init__(self, mesh, pts, ptsv=None,order=3, logSigma=True, normal='X'):
        IdentityMap.__init__(self, mesh)
        self.logSigma = logSigma
        self.order = order
        self.normal = normal
        self.pts= pts
        self.npts = np.size(pts)
        self.ptsv = ptsv
        self.spl = None

    slope = 1e4
    @property
    def nP(self):
        if self.mesh.dim == 2:
            return np.size(self.pts)+2
        elif self.mesh.dim == 3:
            return np.size(self.pts)*2+2
        else:
            raise(Exception("Only supports 2D and 3D"))

    def _transform(self, m):
        # Set model parameters
        alpha = self.slope
        sig1,sig2 = m[0],m[1]
        c = m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        #2D
        if self.mesh.dim == 2:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]
            self.spl = UnivariateSpline(self.pts, c, k=self.order, s=0)
            if self.normal =='X':
                f = self.spl(Y) - X
            elif self.normal =='Y':
                f = self.spl(X) - Y
            else:
                raise(Exception("Input for normal = X or Y or Z"))

        # 3D:
        # Comments:
        # Make two spline functions and link them using linear interpolation.
        # This is not quite direct extension of 2D to 3D case
        # Using 2D interpolation  is possible

        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]
            Z = self.mesh.gridCC[:,2]

            npts = np.size(self.pts)
            if np.mod(c.size, 2):
                raise(Exception("Put even points!"))

            self.spl = {"splb":UnivariateSpline(self.pts, c[:npts], k=self.order, s=0),
                        "splt":UnivariateSpline(self.pts, c[npts:], k=self.order, s=0)}

            if self.normal =='X':
                zb = self.ptsv[0]
                zt = self.ptsv[1]
                flines = (self.spl["splt"](Y)-self.spl["splb"](Y))*(Z-zb)/(zt-zb) + self.spl["splb"](Y)
                f = flines - X
            # elif self.normal =='Y':
            # elif self.normal =='Z':
            else:
                raise(Exception("Input for normal = X or Y or Z"))
        else:
            raise(Exception("Only supports 2D and 3D"))


        return sig1+(sig2-sig1)*(np.arctan(alpha*f)/np.pi+0.5)

    def deriv(self, m):
        alpha = self.slope
        sig1,sig2, c = m[0],m[1],m[2:]
        if self.logSigma:
            sig1, sig2 = np.exp(sig1), np.exp(sig2)
        #2D
        if self.mesh.dim == 2:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]

            if self.normal =='X':
                f = self.spl(Y) - X
            elif self.normal =='Y':
                f = self.spl(X) - Y
            else:
                raise(Exception("Input for normal = X or Y or Z"))
        #3D
        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[:,0]
            Y = self.mesh.gridCC[:,1]
            Z = self.mesh.gridCC[:,2]
            if self.normal =='X':
                zb = self.ptsv[0]
                zt = self.ptsv[1]
                flines = (self.spl["splt"](Y)-self.spl["splb"](Y))*(Z-zb)/(zt-zb) + self.spl["splb"](Y)
                f = flines - X
            # elif self.normal =='Y':
            # elif self.normal =='Z':
            else:
                raise(Exception("Not Implemented for Y and Z, your turn :)"))

        if self.logSigma:
            g1 = -(np.arctan(alpha*f)/np.pi + 0.5)*sig1 + sig1
            g2 = (np.arctan(alpha*f)/np.pi + 0.5)*sig2
        else:
            g1 = -(np.arctan(alpha*f)/np.pi + 0.5) + 1.0
            g2 = (np.arctan(alpha*f)/np.pi + 0.5)


        if self.mesh.dim ==2:
            g3 = np.zeros((self.mesh.nC, self.npts))
            if self.normal =='Y':
                # Here we use perturbation to compute sensitivity
                # TODO: bit more generalization of this ...
                # Modfications for X and Z directions ...
                for i in range(np.size(self.pts)):
                    ctemp = c[i]
                    ind = np.argmin(abs(self.mesh.vectorCCy-ctemp))
                    ca = c.copy()
                    cb = c.copy()
                    dy = self.mesh.hy[ind]*1.5
                    ca[i] = ctemp+dy
                    cb[i] = ctemp-dy
                    spla = UnivariateSpline(self.pts, ca, k=self.order, s=0)
                    splb = UnivariateSpline(self.pts, cb, k=self.order, s=0)
                    fderiv = (spla(X)-splb(X))/(2*dy)
                    g3[:,i] = Utils.sdiag(alpha*(sig2-sig1)/(1.+(alpha*f)**2)/np.pi)*fderiv

        elif self.mesh.dim==3:
            g3 = np.zeros((self.mesh.nC, self.npts*2))
            if self.normal =='X':
                # Here we use perturbation to compute sensitivity
                for i in range(self.npts*2):
                    ctemp = c[i]
                    ind = np.argmin(abs(self.mesh.vectorCCy-ctemp))
                    ca = c.copy()
                    cb = c.copy()
                    dy = self.mesh.hy[ind]*1.5
                    ca[i] = ctemp+dy
                    cb[i] = ctemp-dy
                    #treat bottom boundary
                    if i< self.npts:
                        splba = UnivariateSpline(self.pts, ca[:self.npts], k=self.order, s=0)
                        splbb = UnivariateSpline(self.pts, cb[:self.npts], k=self.order, s=0)
                        flinesa = (self.spl["splt"](Y)-splba(Y))*(Z-zb)/(zt-zb) + splba(Y) - X
                        flinesb = (self.spl["splt"](Y)-splbb(Y))*(Z-zb)/(zt-zb) + splbb(Y) - X
                    #treat top boundary
                    else:
                        splta = UnivariateSpline(self.pts, ca[self.npts:], k=self.order, s=0)
                        spltb = UnivariateSpline(self.pts, ca[self.npts:], k=self.order, s=0)
                        flinesa = (self.spl["splt"](Y)-splta(Y))*(Z-zb)/(zt-zb) + splta(Y) - X
                        flinesb = (self.spl["splt"](Y)-spltb(Y))*(Z-zb)/(zt-zb) + spltb(Y) - X
                    fderiv = (flinesa-flinesb)/(2*dy)
                    g3[:,i] = Utils.sdiag(alpha*(sig2-sig1)/(1.+(alpha*f)**2)/np.pi)*fderiv
        else :
            raise(Exception("Not Implemented for Y and Z, your turn :)"))
        return sp.csr_matrix(np.c_[g1,g2,g3])






