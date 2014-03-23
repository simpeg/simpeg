import Utils, Parameters, numpy as np, scipy.sparse as sp
from Tests import checkDerivative

class BaseModel(object):
    """
    SimPEG Model

    """

    __metaclass__ = Utils.SimPEGMetaClass

    counter = None   #: A SimPEG.Utils.Counter object
    mesh = None      #: A SimPEG Mesh

    def __init__(self, mesh):
        self.mesh = mesh

    def transform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The *transform* changes the model into the physical property.

        """
        return m

    def transformInverse(self, D):
        """
            :param numpy.array D: physical property
            :rtype: numpy.array
            :return: model

            The *transformInverse* changes the physical property into the model.

            .. note:: The *transformInverse* may not be easy to create in general.

        """
        raise NotImplementedError('The transformInverse is not implemented.')

    def transformDeriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The *transform* changes the model into the physical property.
            The *transformDeriv* provides the derivative of the *transform*.
        """
        return sp.identity(m.size)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh.nC

    def example(self):
        return np.random.rand(self.nP)

    def test(self, m=None, **kwargs):
        print 'Testing the %s Class!' % self.__class__.__name__
        if m is None:
            m = self.example()
            if 'plotIt' not in kwargs:
                kwargs['plotIt'] = False
        return checkDerivative(lambda m : [self.transform(m), self.transformDeriv(m)], m, **kwargs)

class BaseNonLinearModel(object):
    """
    SimPEG BaseNonLinearModel

    """

    __metaclass__ = Utils.SimPEGMetaClass

    counter = None   #: A SimPEG.Utils.Counter object
    mesh = None      #: A SimPEG Mesh

    def __init__(self, mesh):
        self.mesh = mesh

    def transform(self, u, m):
        """
            :param numpy.array u: fields
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The *transform* changes the model into the physical property.

        """
        return m

    def transformDerivU(self, u, m):
        """
            :param numpy.array u: fields
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The *transform* changes the model into the physical property.
            The *transformDerivU* provides the derivative of the *transform* with respect to the fields.
        """
        raise NotImplementedError('The transformDerivU is not implemented.')


    def transformDerivM(self, u, m):
        """
            :param numpy.array u: fields
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The *transform* changes the model into the physical property.
            The *transformDerivU* provides the derivative of the *transform* with respect to the model.
        """
        raise NotImplementedError('The transformDerivM is not implemented.')

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh.nC

    def example(self):
        raise NotImplementedError('The example is not implemented.')

    def test(self, m=None):
        raise NotImplementedError('The test is not implemented.')


class LogModel(BaseModel):
    """SimPEG LogModel"""

    def __init__(self, mesh, **kwargs):
        BaseModel.__init__(self, mesh, **kwargs)

    def transform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The *transform* changes the model into the physical property.

            A common example of this is to invert for electrical conductivity
            in log space. In this case, your model will be log(sigma) and to
            get back to sigma, you can take the exponential:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma
        """
        return np.exp(Utils.mkvc(m))


    def transformInverse(self, D):
        """
            :param numpy.array D: physical property
            :rtype: numpy.array
            :return: model

            The *transformInverse* changes the physical property into the model.

            .. math::

                m = \log{\sigma}

        """
        return np.log(Utils.mkvc(D))


    def transformDeriv(self, m):
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

class Vertical1DModel(BaseModel):
    """Vertical1DModel

        Given a 1D vector through the last dimension
        of the mesh, this will extend to the full
        model space.
    """

    def __init__(self, mesh, **kwargs):
        BaseModel.__init__(self, mesh, **kwargs)

    @property
    def nP(self):
        """Number of model properties.

           The number of cells in the
           last dimension of the mesh."""
        return self.mesh.vnC[self.mesh.dim-1]

    def transform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model
        """
        repNum = self.mesh.vnC[:self.mesh.dim-1].prod()
        return Utils.mkvc(m).repeat(repNum)

    def transformDeriv(self, m):
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

class Mesh2Mesh(BaseModel):
    """
        Takes a model on one mesh are translates it to another mesh.

        .. plot::

            from SimPEG import *
            M = Mesh.TensorMesh([100,100])
            h1 = Utils.meshTensors(((7,6,1.5),(10,6),(7,6,1.5)))
            h1 = h1/h1.sum()
            M2 = Mesh.TensorMesh([h1,h1])
            V = Utils.ModelBuilder.randomModel(M.vnC, seed=79, its=50)
            v = Utils.mkvc(V)
            modh = Model.Mesh2Mesh([M,M2])
            modH = Model.Mesh2Mesh([M2,M])
            H = modH.transform(v)
            h = modh.transform(H)
            ax = plt.subplot(131)
            M.plotImage(v, ax=ax)
            ax.set_title('Fine Mesh (Original)')
            ax = plt.subplot(132)
            M2.plotImage(H,clim=[0,1],ax=ax)
            ax.set_title('Course Mesh')
            ax = plt.subplot(133)
            M.plotImage(h,clim=[0,1],ax=ax)
            ax.set_title('Fine Mesh (Interpolated)')

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
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh2.nC
    def transform(self, m):
        return self.P*m
    def transformDeriv(self, m):
        return self.P


class ActiveModel(BaseModel):
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
        if type(valInactive) in [float, int, long]:
            valInactive = np.ones(self.nC)*float(valInactive)

        valInactive[self.indActive] = 0
        self.valInactive = valInactive

        inds = np.nonzero(self.indActive)[0]
        self.P = sp.csr_matrix((np.ones(inds.size),(inds, range(inds.size))), shape=(self.nC, self.nP))

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.indActive.sum()

    def transform(self, m):
        return self.P*m + self.valInactive
    def transformDeriv(self, m):
        return self.P

class ComboModel(BaseModel):
    """Combination of various models."""

    def __init__(self, mesh, models, **kwargs):
        BaseModel.__init__(self, mesh, **kwargs)

        self.models = []
        for m in models:
            if not isinstance(m, BaseModel):
                self.models += [m(mesh, **kwargs)]
            else:
                self.models += [m]

    @property
    def nP(self):
        """Number of model properties.

           The number of cells in the
           last dimension of the mesh."""
        return self.models[-1].nP

    def transform(self, m):
        for model in reversed(self.models):
            m = model.transform(m)
        return m

    def transformDeriv(self, m):
        deriv = 1
        mi = m
        for model in reversed(self.models):
            deriv = model.transformDeriv(mi) * deriv
            mi = model.transform(mi)
        return deriv

if __name__ == '__main__':
    from SimPEG import *
    mesh = Mesh.TensorMesh([10,8])
    combo = ComboModel(mesh, [LogModel, Vertical1DModel])
    m = combo.example()
    print m.shape
    print combo.test(np.arange(8))
