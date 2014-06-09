import Utils, numpy as np, scipy.sparse as sp
from Tests import checkDerivative


class IdentityMap(object):
    """
    SimPEG Map

    """

    __metaclass__ = Utils.SimPEGMetaClass

    mesh = None      #: A SimPEG Mesh

    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def nP(self):
        """
            :rtype: int
            :return: number of parameters in the model
        """
        return self.mesh.nC

    @property
    def shape(self):
        """
            The default shape is (mesh.nC, nP).

            :rtype: (int,int)
            :return: shape of the operator as a tuple
        """
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
            m = np.random.rand(self.nP)
        if 'plotIt' not in kwargs:
            kwargs['plotIt'] = False
        return checkDerivative(lambda m : [self * m, self.deriv(m)], m, **kwargs)

    def _assertMatchesPair(self, pair):
        assert (isinstance(self, pair) or
            isinstance(self, ComboMap) and isinstance(self.maps[0], pair)
            ), "Mapping object must be an instance of a %s class."%(pair.__name__)

    def __mul__(self, val):
        if isinstance(val, IdentityMap):
            if not self.shape[1] == val.shape[0]:
                raise ValueError('Dimension mismatch in %s and %s.' % (str(self), str(val)))
            return ComboMap([self, val])
        elif isinstance(val, np.ndarray):
            if not self.shape[1] == val.shape[0]:
                raise ValueError('Dimension mismatch in %s and np.ndarray%s.' % (str(self), str(val.shape)))
            return self._transform(val)
        raise Exception('Unrecognized data type to multiply. Try a map or a numpy.ndarray!')

    def __str__(self):
        return "%s(%d,%d)" % (self.__class__.__name__, self.shape[0], self.shape[1])

class ComboMap(IdentityMap):
    """Combination of various maps."""

    def __init__(self, maps, **kwargs):
        IdentityMap.__init__(self, None, **kwargs)

        self.maps = []
        for ii, m in enumerate(maps):
            assert isinstance(m, IdentityMap), 'Unrecognized data type, inherit from an IdentityMap or ComboMap!'
            if ii > 0 and not self.shape[1] == m.shape[0]:
                prev = self.maps[-1]
                errArgs = (prev.__name__, prev.shape[0], prev.shape[1], m.__name__, m.shape[0], m.shape[1])
                raise ValueError('Dimension mismatch in map[%s] (%i, %i) and map[%s] (%i, %i).' % errArgs)

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
        return 'ComboMap[%s]%s' % (' * '.join([m.__str__() for m in self.maps]), str(self.shape))


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
            valInactive = np.ones(self.nC)*float(valInactive)

        valInactive[self.indActive] = 0
        self.valInactive = valInactive

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
        return Utils.SimPEGLinearOperator(shp,fwd,adj)

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
        return np.c_[g1,g2,g3,g4,g5]
