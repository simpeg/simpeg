from __future__ import division
import Utils, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from Tests import checkDerivative
from PropMaps import PropMap, Property
from numpy.polynomial import polynomial
from scipy.interpolate import UnivariateSpline
import warnings
from SimPEG.Utils import Zero

class IdentityMap(object):
    """
    SimPEG Map

    """
    __metaclass__ = Utils.SimPEGMetaClass

    def __init__(self, mesh=None, nP=None, **kwargs):
        Utils.setKwargs(self, **kwargs)

        if nP is not None:
            assert type(nP) in [int, long, np.int64], ' Number of parameters must be an integer.'

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
    """
        Combination of various maps.

        The ComboMap holds the information for multiplying and combining
        maps. It also uses the chain rule to create the derivative.
        Remember, any time that you make your own combination of mappings
        be sure to test that the derivative is correct.

    """

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
        Electrical conductivity varies over many orders of magnitude, so it is a common
        technique when solving the inverse problem to parameterize and optimize in terms
        of log conductivity. This makes sense not only because it ensures all conductivities
        will be positive, but because this is fundamentally the space where conductivity
        lives (i.e. it varies logarithmically).

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

class SurjectFull(IdentityMap):
    """
    SurjectFull

    Given a scalar, the SurjectFull maps the value to the
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

class FullMap(SurjectFull):
    def __init__(self,mesh,**kwargs):
        warnings.warn(
            "`FullMap` is deprecated and will be removed in future versions. Use `SurjectFull` instead",
            FutureWarning)
        SurjectFull.__init__(self,mesh,**kwargs)

class SurjectVertical1D(IdentityMap):
    """SurjectVertical1DMap

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

class Vertical1DMap(SurjectVertical1D):
    def __init__(self,mesh,**kwargs):
        warnings.warn(
            "`Vertical1DMap` is deprecated and will be removed in future versions. Use `SurjectVertical1D` instead",
            FutureWarning)
        SurjectVertical1D.__init__(self,mesh,**kwargs)

class Surject2Dto3D(IdentityMap):
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

class Map2Dto3D(Surject2Dto3D):
    def __init__(self,mesh,**kwargs):
        warnings.warn(
            "`Map2Dto3D` is deprecated and will be removed in future versions. Use `Surject2Dto3D` instead",
            FutureWarning)
        Surject2Dto3D.__init__(self,mesh,**kwargs)

class Mesh2Mesh(IdentityMap):
    """
        Takes a model on one mesh are translates it to another mesh.

        .. plot::

            from SimPEG import *
            import matplotlib.pyplot as plt
            M = Mesh.TensorMesh([100,100])
            h1 = Utils.meshTensor([(6,7,-1.5),(6,10),(6,7,1.5)])
            h1 = h1/h1.sum()
            M2 = Mesh.TensorMesh([h1,h1])
            V = Utils.ModelBuilder.randomModel(M.vnC, seed=79, its=50)
            v = Utils.mkvc(V)
            modh = Maps.Mesh2Mesh([M,M2])
            modH = Maps.Mesh2Mesh([M2,M])
            H = modH * v
            h = modh * H
            ax = plt.subplot(131)
            M.plotImage(v, ax=ax)
            ax.set_title('Fine Mesh (Original)')
            ax = plt.subplot(132)
            M2.plotImage(H,clim=[0,1],ax=ax)
            ax.set_title('Course Mesh')
            ax = plt.subplot(133)
            M.plotImage(h,clim=[0,1],ax=ax)
            ax.set_title('Fine Mesh (Interpolated)')
            plt.show()


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


class InjectActiveCells(IdentityMap):
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

class ActiveCells(InjectActiveCells):
    def __init__(self, mesh, indActive, valInactive, nC=None):
        warnings.warn(
            "`ActiveCells` is deprecated and will be removed in future versions. Use `InjectActiveCells` instead",
            FutureWarning)
        InjectActiveCells.__init__(self, mesh, indActive, valInactive, nC)

class InjectActiveCellsTopo(IdentityMap):
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

class ActiveCellsTopo(InjectActiveCellsTopo):
    def __init__(self, mesh, indActive, valInactive, nC=None):
        warnings.warn(
            "`ActiveCellsTopo` is deprecated and will be removed in future versions. Use `InjectActiveCellsTopo` instead",
            FutureWarning)
        InjectActiveCellsTopo.__init__(self, mesh, indActive, valInactive, nC)

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

        .. math::

            \sigma(m) = \sigma_1 + (\sigma_2 - \sigma_1)\left(\\arctan\left(100*\sqrt{(\\vec{x}-x_0)^2 + (\\vec{y}-y_0)}-r\\right) \pi^{-1} + 0.5\\right)

        Define the model as:

        .. math::

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

        Can take in an actInd vector to account for topography.

    """
    def __init__(self, mesh, order, logSigma=True, normal='X', actInd = None):
        IdentityMap.__init__(self, mesh)
        self.logSigma = logSigma
        self.order = order
        self.normal = normal
        self.actInd = actInd

        if getattr(self, 'actInd', None) is None:
            self.actInd = range(self.mesh.nC)
            self.nC = self.mesh.nC

        else:
            self.nC = len(self.actInd)

    slope = 1e4

    @property
    def shape(self):
        return (self.nC, self.nP)

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
            X = self.mesh.gridCC[self.actInd,0]
            Y = self.mesh.gridCC[self.actInd,1]
            if self.normal =='X':
                f = polynomial.polyval(Y, c) - X
            elif self.normal =='Y':
                f = polynomial.polyval(X, c) - Y
            else:
                raise(Exception("Input for normal = X or Y or Z"))
        #3D
        elif self.mesh.dim == 3:
            X = self.mesh.gridCC[self.actInd,0]
            Y = self.mesh.gridCC[self.actInd,1]
            Z = self.mesh.gridCC[self.actInd,2]
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
            X = self.mesh.gridCC[self.actInd,0]
            Y = self.mesh.gridCC[self.actInd,1]

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
            X = self.mesh.gridCC[self.actInd,0]
            Y = self.mesh.gridCC[self.actInd,1]
            Z = self.mesh.gridCC[self.actInd,2]

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





class ParametrizedBlockInLayer(IdentityMap):
    """
        Parametrized Block in a Layered Space

        For 2D:
        m = [val_background, val_layer, val_block, layer_center, layer_thickness, block_x0, block_dx]

        For 3D:
        m = [val_background, val_layer, val_block, layer_center, layer_thickness, block_x0, block_y0, block_dx, block_dy]

        .. plot::
            :include-source:

            from SimPEG import Mesh, Maps, np
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1,1,figsize=(2,3))

            mesh = Mesh.TensorMesh([50,50],x0='CC')
            mapping = Maps.ParametrizedBlockInLayer(mesh)
            m = np.hstack(np.r_[1., 2., 3., -0.1, 0.2, 0.3, 0.2])
            rho = mapping._transform(m)
            mesh.plotImage(rho, ax=ax)

        **Required**

        :param Mesh mesh: SimPEG Mesh, 2D or 3D

        **Optional**

        :param float slope_fact: arctan slope factor - divided by the minimum h spacing to give the slope of the arctan functions
        :param float slope: slope of the arctan function
        :param numpy.ndarray indActive: bool vector with

    """

    slope_fact = 1e2 # will be scaled by the mesh.
    slope = None
    indActive = None

    def __init__(self, mesh, **kwargs):

        super(ParametrizedBlockInLayer, self).__init__(mesh, **kwargs)


        if self.slope is None:
            self.slope = self.slope_fact / np.hstack(self.mesh.h).min()

        self.x = [self.mesh.gridCC[:,0] if self.indActive is None else self.mesh.gridCC[indActive,0]][0]

        if self.mesh.dim > 1:
            self.y = [self.mesh.gridCC[:,1] if self.indActive is None else self.mesh.gridCC[indActive,1]][0]

        if self.mesh.dim > 2:
            self.z = [self.mesh.gridCC[:,2] if self.indActive is None else self.mesh.gridCC[indActive,2]][0]

    @property
    def nP(self):
        if self.mesh.dim == 2:
            return 7
        elif self.mesh.dim == 3:
            return 9

    def _validate_m(self, m):
        # TODO: more sanity checks here
        if self.mesh.dim == 2:
            assert len(m) == 7, 'm must be length 7 not {0}: [val_back, val_layer, val_block, layer_center, layer_thickness, x0_block, dx_block'.format(len(m))
        elif self.mesh.dim == 3:
            assert len(m) == 9, 'm must be length 9 not {0}: [val_back, val_layer, val_block, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block]'.format(len(m))
        else:
            raise NotImplementedError('Only 2D and 3D meshes are implemented for the Parametrized_Block_in_Layer Map')

    def _atanfct(self, xyz, xyzi, slope):
        return np.arctan(slope * (xyz - xyzi))/np.pi + 0.5

    def _atanfctDeriv(self, xyz, xyzi, slope):
        # d/dx(atan(x)) = 1/(1+x**2)
        x = slope * (xyz - xyzi)
        dx = - slope
        return (1./(1 + x**2))/np.pi * dx

    def _atanlayer(self, layer_center, layer_thickness):
        if self.mesh.dim == 2:
            z = self.y
        elif self.mesh.dim == 3:
            z = self.z

        layer_bottom = layer_center - layer_thickness / 2.
        layer_top = layer_center + layer_thickness / 2.
        return self._atanfct(z, layer_bottom, self.slope)*self._atanfct(z, layer_top, -self.slope)

    def _atanlayerDeriv_layer_center(self, layer_center, layer_thickness):
        if self.mesh.dim == 2:
            z = self.y
        elif self.mesh.dim == 3:
            z = self.z

        layer_bottom = layer_center - layer_thickness / 2.
        layer_top = layer_center + layer_thickness / 2.

        return (self._atanfctDeriv(z, layer_bottom, self.slope)*self._atanfct(z, layer_top, -self.slope)
                + self._atanfct(z, layer_bottom, self.slope)*self._atanfctDeriv(z, layer_top, -self.slope))

    def _atanlayerDeriv_layer_thickness(self, layer_center, layer_thickness):
        if self.mesh.dim == 2:
            z = self.y
        elif self.mesh.dim == 3:
            z = self.z

        layer_bottom = layer_center - layer_thickness / 2.
        layer_top = layer_center + layer_thickness / 2.

        return (-0.5*self._atanfctDeriv(z, layer_bottom, self.slope)*self._atanfct(z, layer_top, -self.slope)
                + 0.5*self._atanfct(z, layer_bottom, self.slope)*self._atanfctDeriv(z, layer_top, -self.slope))

    def _atanblock2d(self, layer_center, layer_thickness, x0_block, dx_block):

        return (self._atanlayer(layer_center, layer_thickness)
                          * self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                          * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope))

    def _atanblock2dDeriv_layer_center(self, layer_center, layer_thickness, x0_block, dx_block):

        return (self._atanlayerDeriv_layer_center(layer_center, layer_thickness)
                          * self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                          * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope))

    def _atanblock2dDeriv_layer_thickness(self, layer_center, layer_thickness, x0_block, dx_block):

        return (self._atanlayerDeriv_layer_thickness(layer_center, layer_thickness)
                          * self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                          * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope))


    def _atanblock2dDeriv_x0(self, layer_center, layer_thickness, x0_block, dx_block):

        return self._atanlayer(layer_center, layer_thickness) * (
                    (self._atanfctDeriv(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope))
                    +
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfctDeriv(self.x, x0_block + 0.5*dx_block, -self.slope))
                    )

    def _atanblock2dDeriv_dx(self, layer_center, layer_thickness, x0_block, dx_block):

        return self._atanlayer(layer_center, layer_thickness) * (
                    (self._atanfctDeriv(self.x, x0_block - 0.5*dx_block, self.slope) * -0.5
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope))
                    +
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfctDeriv(self.x, x0_block + 0.5*dx_block, -self.slope) * 0.5)
                    )

    def _atanblock3d(self, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block):

        return (self._atanlayer(layer_center, layer_thickness)
                          * self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                          * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                          * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                          * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))


    def _atanblock3dDeriv_layer_center(self, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block):

        return (self._atanlayerDeriv_layer_center(layer_center, layer_thickness)
                          * self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                          * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                          * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                          * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))

    def _atanblock3dDeriv_layer_thickness(self, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block):

        return (self._atanlayerDeriv_layer_thickness(layer_center, layer_thickness)
                          * self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                          * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                          * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                          * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))


    def _atanblock3dDeriv_x0(self, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block):

        return self._atanlayer(layer_center, layer_thickness) * (
                    (self._atanfctDeriv(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                    * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                    * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))
                    +
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfctDeriv(self.x, x0_block + 0.5*dx_block, -self.slope)
                    * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                    * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))
                    )

    def _atanblock3dDeriv_y0(self, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block):

        return self._atanlayer(layer_center, layer_thickness) * (
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                    * self._atanfctDeriv(self.y, y0_block - 0.5*dy_block, self.slope)
                    * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))
                    +
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                    * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                    * self._atanfctDeriv(self.y, y0_block + 0.5*dy_block, -self.slope))
                    )

    def _atanblock3dDeriv_dx(self, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block):

        return self._atanlayer(layer_center, layer_thickness) * (
                    (self._atanfctDeriv(self.x, x0_block - 0.5*dx_block, self.slope) * -0.5
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                    * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                    * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))
                    +
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfctDeriv(self.x, x0_block + 0.5*dx_block, -self.slope) * 0.5
                    * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                    * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))
                    )

    def _atanblock3dDeriv_dy(self, layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block):

        return self._atanlayer(layer_center, layer_thickness) * (
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                    * self._atanfctDeriv(self.y, y0_block - 0.5*dy_block, self.slope) * -0.5
                    * self._atanfct(self.y, y0_block + 0.5*dy_block, -self.slope))
                    +
                    (self._atanfct(self.x, x0_block - 0.5*dx_block, self.slope)
                    * self._atanfct(self.x, x0_block + 0.5*dx_block, -self.slope)
                    * self._atanfct(self.y, y0_block - 0.5*dy_block, self.slope)
                    * self._atanfctDeriv(self.y, y0_block + 0.5*dy_block, -self.slope) * 0.5)
                    )


    def _transform2d(self, m):

        # parse model
        vals = m[:3]    # model values
        layer_center = m[3]
        layer_thickness = m[4]
        x0_block = m[5] # x-center of the block
        dx_block = m[6] # block width

        # assemble the model
        layer_cont = vals[0] + (vals[1]-vals[0])*self._atanlayer(layer_center, layer_thickness) # contribution from the layered background
        block_cont = (vals[2]-layer_cont)*self._atanblock2d(layer_center, layer_thickness, x0_block, dx_block) # perturbation due to the block

        return layer_cont + block_cont

    def _deriv2d(self, m):
        # [val_back, val_layer, val_block, x0_block, dx_block]
        # parse model
        vals = m[:3]    # model values
        layer_center = m[3]
        layer_thickness = m[4]
        x0_block = m[5] # x-center of the block
        dx_block = m[6] # block width

        layer_cont = vals[0] + (vals[1]-vals[0])*self._atanlayer(layer_center, layer_thickness) # contribution to background from layer

        # background value
        d_layer_dval0 = np.ones_like(self.x) + (-1.)*self._atanlayer(layer_center, layer_thickness)
        d_block_dval0 = (-d_layer_dval0)*self._atanblock2d(layer_center, layer_thickness, x0_block, dx_block)
        val0_deriv = d_layer_dval0 + d_block_dval0

        # layer value
        d_layer_dval1 = self._atanlayer(layer_center, layer_thickness)
        d_block_dval1 = (-d_layer_dval1)*self._atanblock2d(layer_center, layer_thickness, x0_block, dx_block)
        val1_deriv = d_layer_dval1 + d_block_dval1

        # block value
        d_layer_dval2 = Zero()
        d_block_dval2 = (1.-d_layer_dval2)*self._atanblock2d(layer_center, layer_thickness, x0_block, dx_block)
        val2_deriv = d_layer_dval2 + d_block_dval2

        # layer_center
        d_layer_dlayer_center = (vals[1]-vals[0])*self._atanlayerDeriv_layer_center(layer_center, layer_thickness)
        d_block_dlayer_center = ((vals[2]-layer_cont)*self._atanblock2dDeriv_layer_center(layer_center, layer_thickness, x0_block, dx_block)
                                    - d_layer_dlayer_center*self._atanblock2d(layer_center, layer_thickness, x0_block, dx_block))
        layer_center_deriv = d_layer_dlayer_center + d_block_dlayer_center

        # layer_thickness
        d_layer_dlayer_thickness = (vals[1]-vals[0])*self._atanlayerDeriv_layer_thickness(layer_center, layer_thickness)
        d_block_dlayer_thickness = ((vals[2]-layer_cont)*self._atanblock2dDeriv_layer_thickness(layer_center, layer_thickness, x0_block, dx_block)
                                    - d_layer_dlayer_thickness*self._atanblock2d(layer_center, layer_thickness, x0_block, dx_block))
        layer_thickness_deriv = d_layer_dlayer_thickness + d_block_dlayer_thickness

        # x0 of the block
        d_layer_dx0 = Zero()
        d_block_dx0 = (vals[2]-layer_cont)*self._atanblock2dDeriv_x0(layer_center, layer_thickness, x0_block, dx_block)
        x0_deriv = d_layer_dx0 + d_block_dx0

        # dx of the block
        d_layer_ddx = Zero()
        d_block_ddx = (vals[2]-layer_cont)*self._atanblock2dDeriv_dx(layer_center, layer_thickness, x0_block, dx_block)
        dx_deriv = d_layer_ddx + d_block_ddx

        return np.vstack([val0_deriv, val1_deriv, val2_deriv, layer_center_deriv, layer_thickness_deriv, x0_deriv, dx_deriv]).T


    def _transform3d(self, m):
        # parse model
        vals = m[:3]    # model values
        layer_center = m[3]
        layer_thickness = m[4]
        x0_block = m[5] # x-center of the block
        y0_block = m[6] # y-center of the block
        dx_block = m[7] # block x-width
        dy_block = m[8] # block y-width

        # assemble the model
        layer_cont = vals[0] + (vals[1]-vals[0])*self._atanlayer(layer_center, layer_thickness) # contribution from the layered background
        block_cont = (vals[2]-layer_cont)*self._atanblock3d(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block) # perturbation due to the block

        return layer_cont + block_cont

    def _deriv3d(self, m):

        # parse model
        vals = m[:3]    # model values
        layer_center = m[3]
        layer_thickness = m[4]
        x0_block = m[5] # x-center of the block
        y0_block = m[6] # y-center of the block
        dx_block = m[7] # block x-width
        dy_block = m[8] # block y-width

        layer_cont = vals[0] + (vals[1]-vals[0])*self._atanlayer(layer_center, layer_thickness) # contribution to background from layer

        # background value
        d_layer_dval0 = np.ones_like(self.x) + (-1.)*self._atanlayer(layer_center, layer_thickness)
        d_block_dval0 = (-d_layer_dval0)*self._atanblock3d(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
        val0_deriv = d_layer_dval0 + d_block_dval0

        # layer value
        d_layer_dval1 = self._atanlayer(layer_center, layer_thickness)
        d_block_dval1 = (-d_layer_dval1)*self._atanblock3d(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
        val1_deriv = d_layer_dval1 + d_block_dval1

        # block value
        d_layer_dval2 = Zero()
        d_block_dval2 = (1.-d_layer_dval2)*self._atanblock3d(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
        val2_deriv = d_layer_dval2 + d_block_dval2

        # layer_center
        d_layer_dlayer_center = (vals[1]-vals[0])*self._atanlayerDeriv_layer_center(layer_center, layer_thickness)
        d_block_dlayer_center = ((vals[2]-layer_cont)*self._atanblock3dDeriv_layer_center(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
                                    - d_layer_dlayer_center*self._atanblock3d(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block))
        layer_center_deriv = d_layer_dlayer_center + d_block_dlayer_center

        # layer_thickness
        d_layer_dlayer_thickness = (vals[1]-vals[0])*self._atanlayerDeriv_layer_thickness(layer_center, layer_thickness)
        d_block_dlayer_thickness = ((vals[2]-layer_cont)*self._atanblock3dDeriv_layer_thickness(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
                                - d_layer_dlayer_thickness*self._atanblock3d(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block))
        layer_thickness_deriv = d_layer_dlayer_thickness + d_block_dlayer_thickness

        # x0 of the block
        d_layer_dx0 = Zero()
        d_block_dx0 = (vals[2]-layer_cont)*self._atanblock3dDeriv_x0(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
        x0_deriv = d_layer_dx0 + d_block_dx0

        # y0 of the block
        d_layer_dy0 = Zero()
        d_block_dy0 = (vals[2]-layer_cont)*self._atanblock3dDeriv_y0(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
        y0_deriv = d_layer_dy0 + d_block_dy0

        # dx of the block
        d_layer_ddx = Zero()
        d_block_ddx = (vals[2]-layer_cont)*self._atanblock3dDeriv_dx(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
        dx_deriv = d_layer_ddx + d_block_ddx

        # dy of the block
        d_layer_ddy = Zero()
        d_block_ddy = (vals[2]-layer_cont)*self._atanblock3dDeriv_dy(layer_center, layer_thickness, x0_block, y0_block, dx_block, dy_block)
        dy_deriv = d_layer_ddy + d_block_ddy

        return np.vstack([val0_deriv, val1_deriv, val2_deriv, layer_center_deriv, layer_thickness_deriv, x0_deriv, y0_deriv, dx_deriv, dy_deriv]).T

    def _transform(self, m):

        self._validate_m(m) # make sure things are the right sizes

        if self.mesh.dim == 2:
            return self._transform2d(m)
        elif self.mesh.dim == 3:
            return self._transform3d(m)

    def deriv(self, m):

        self._validate_m(m) # make sure things are the right sizes

        if self.mesh.dim == 2:
            return sp.csr_matrix(self._deriv2d(m))
        elif self.mesh.dim == 3:
            return sp.csr_matrix(self._deriv3d(m))

