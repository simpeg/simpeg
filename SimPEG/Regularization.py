import Utils, Maps, Mesh, numpy as np, scipy.sparse as sp

class BaseRegularization(object):
    """
    **Base Regularization Class**

    This is used to regularize the model space::

        reg = Regularization(mesh)

    """

    __metaclass__ = Utils.SimPEGMetaClass

    counter = None

    mapPair = Maps.IdentityMap    #: A SimPEG.Map Class

    mapping = None    #: A SimPEG.Map instance.
    mesh    = None    #: A SimPEG.Mesh instance.
    mref = None       #: Reference model.

    def __init__(self, mesh, mapping=None, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.mesh = mesh
        assert isinstance(mesh, Mesh.BaseMesh), "mesh must be a SimPEG.Mesh object."
        self.mapping = mapping or Maps.IdentityMap(mesh)
        self.mapping._assertMatchesPair(self.mapPair)

    @property
    def parent(self):
        """This is the parent of the regularization."""
        return getattr(self,'_parent',None)
    @parent.setter
    def parent(self, p):
        if getattr(self,'_parent',None) is not None:
            print 'Regularization has switched to a new parent!'
        self._parent = p

    @property
    def inv(self): return self.parent.inv
    @property
    def objFunc(self): return self.parent
    @property
    def reg(self): return self
    @property
    def opt(self): return self.parent.opt
    @property
    def prob(self): return self.parent.prob
    @property
    def survey(self): return self.parent.survey


    @property
    def W(self):
        """Full regularization weighting matrix W."""
        return sp.identity(self.mapping.nP)


    @Utils.timeIt
    def eval(self, m):
        r = self.W * ( self.mapping * (m - self.mref) )
        return 0.5*r.dot(r)

    @Utils.timeIt
    def evalDeriv(self, m):
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """
        mD = self.mapping.deriv(m - self.mref)
        r = self.W * ( self.mapping * (m - self.mref) )
        return mD.T * ( self.W.T * r )

    @Utils.timeIt
    def eval2Deriv(self, m, v=None):
        """

            :param numpy.array m: geophysical model
            :param numpy.array v: vector to multiply
            :rtype: scipy.sparse.csr_matrix or numpy.ndarray
            :return: WtW or WtW*v

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

        So the second derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W}

        """
        mD = self.mapping.deriv(m - self.mref)
        if v is None:
            return mD.T * self.W.T * self.W * mD

        return mD.T * ( self.W.T * ( self.W * ( mD * v) ) )




class Tikhonov(BaseRegularization):
    """**Tikhonov Regularization**

        Here we will define regularization of a model, m, in general however, this should be thought of as (m-m_ref) but otherwise it is exactly the same:

        .. math::

            R(m) = \int_\Omega \\frac{\\alpha_x}{2}\left(\\frac{\partial m}{\partial x}\\right)^2 + \\frac{\\alpha_y}{2}\left(\\frac{\partial m}{\partial y}\\right)^2 \partial v

        Our discrete gradient operator works on cell centers and gives the derivative on the cell faces, which is not where we want to be evaluating this integral. We need to average the values back to the cell-centers before we integrate. To avoid null spaces, we square first and then average. In 2D with ij notation it looks like this:

        .. math::

            R(m) \\approx \sum_{ij} \left[\\frac{\\alpha_x}{2}\left[\left(\\frac{m_{i+1,j} - m_{i,j}}{h}\\right)^2 + \left(\\frac{m_{i,j} - m_{i-1,j}}{h}\\right)^2\\right]
            + \\frac{\\alpha_y}{2}\left[\left(\\frac{m_{i,j+1} - m_{i,j}}{h}\\right)^2 + \left(\\frac{m_{i,j} - m_{i,j-1}}{h}\\right)^2\\right]
            \\right]h^2

        If we let D_1 be the derivative matrix in the x direction

        .. math::

            \mathbf{D}_1 = \mathbf{I}_2\otimes\mathbf{d}_1

        .. math::

            \mathbf{D}_2 = \mathbf{d}_2\otimes\mathbf{I}_1

        Where d_1 is the one dimensional derivative:

        .. math::

            \mathbf{d}_1 = \\frac{1}{h} \left[ \\begin{array}{cccc}
            -1 & 1 & & \\\\
             & \ddots & \ddots&\\\\
             &  & -1 & 1\end{array} \\right]

        .. math::

            R(m) \\approx \mathbf{v}^\\top \left[\\frac{\\alpha_x}{2}\mathbf{A}_1 (\mathbf{D}_1 m) \odot (\mathbf{D}_1 m) + \\frac{\\alpha_y}{2}\mathbf{A}_2 (\mathbf{D}_2 m) \odot (\mathbf{D}_2 m) \\right]

        Recall that this is really a just point wise multiplication, or a diagonal matrix times a vector. When we multiply by something in a diagonal we can interchange and it gives the same results (i.e. it is point wise)

        .. math::

            \mathbf{a\odot b} = \\text{diag}(\mathbf{a})\mathbf{b} = \\text{diag}(\mathbf{b})\mathbf{a} = \mathbf{b\odot a}

        and the transpose also is true (but the sizes have to make sense...):

        .. math::

            \mathbf{a}^\\top\\text{diag}(\mathbf{b}) = \mathbf{b}^\\top\\text{diag}(\mathbf{a})

        So R(m) can simplify to:

        .. math::

            R(m) \\approx  \mathbf{m}^\\top \left[\\frac{\\alpha_x}{2}\mathbf{D}_1^\\top \\text{diag}(\mathbf{A}_1^\\top\mathbf{v}) \mathbf{D}_1 +  \\frac{\\alpha_y}{2}\mathbf{D}_2^\\top \\text{diag}(\mathbf{A}_2^\\top \mathbf{v}) \mathbf{D}_2 \\right] \mathbf{m}

        We will define W_x as:

        .. math::

            \mathbf{W}_x = \sqrt{\\alpha_x}\\text{diag}\left(\sqrt{\mathbf{A}_1^\\top\mathbf{v}}\\right) \mathbf{D}_1


        And then W as a tall matrix of all of the different regularization terms:

        .. math::

            \mathbf{W} = \left[ \\begin{array}{c}
            \mathbf{W}_s\\\\
            \mathbf{W}_x\\\\
            \mathbf{W}_y\end{array} \\right]

        Then we can write

        .. math::

            R(m) \\approx \\frac{1}{2}\mathbf{m^\\top W^\\top W m}


    """

    alpha_s  = Utils.dependentProperty('_alpha_s', 1e-6, ['_W', '_Ws'], "Smallness weight")
    alpha_x  = Utils.dependentProperty('_alpha_x', 1.0, ['_W', '_Wx'], "Weight for the first derivative in the x direction")
    alpha_y  = Utils.dependentProperty('_alpha_y', 1.0, ['_W', '_Wy'], "Weight for the first derivative in the y direction")
    alpha_z  = Utils.dependentProperty('_alpha_z', 1.0, ['_W', '_Wz'], "Weight for the first derivative in the z direction")
    alpha_xx = Utils.dependentProperty('_alpha_xx', 0.0, ['_W', '_Wxx'], "Weight for the second derivative in the x direction")
    alpha_yy = Utils.dependentProperty('_alpha_yy', 0.0, ['_W', '_Wyy'], "Weight for the second derivative in the y direction")
    alpha_zz = Utils.dependentProperty('_alpha_zz', 0.0, ['_W', '_Wzz'], "Weight for the second derivative in the z direction")

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseRegularization.__init__(self, mesh, mapping=mapping, **kwargs)

    @property
    def Ws(self):
        """Regularization matrix Ws"""
        if getattr(self,'_Ws', None) is None:
            self._Ws = Utils.sdiag((self.mesh.vol*self.alpha_s)**0.5)
        return self._Ws

    @property
    def Wx(self):
        """Regularization matrix Wx"""
        if getattr(self, '_Wx', None) is None:
            Ave_x_vol = self.mesh.aveF2CC[:,:self.mesh.nFx].T*self.mesh.vol
            self._Wx = Utils.sdiag((Ave_x_vol*self.alpha_x)**0.5)*self.mesh.cellGradx
        return self._Wx

    @property
    def Wy(self):
        """Regularization matrix Wy"""
        if getattr(self, '_Wy', None) is None:
            Ave_y_vol = self.mesh.aveF2CC[:,self.mesh.nFx:np.sum(self.mesh.vnF[:2])].T*self.mesh.vol
            self._Wy = Utils.sdiag((Ave_y_vol*self.alpha_y)**0.5)*self.mesh.cellGrady
        return self._Wy

    @property
    def Wz(self):
        """Regularization matrix Wz"""
        if getattr(self, '_Wz', None) is None:
            Ave_z_vol = self.mesh.aveF2CC[:,np.sum(self.mesh.vnF[:2]):].T*self.mesh.vol
            self._Wz = Utils.sdiag((Ave_z_vol*self.alpha_z)**0.5)*self.mesh.cellGradz
        return self._Wz

    @property
    def Wxx(self):
        """Regularization matrix Wxx"""
        if getattr(self, '_Wxx', None) is None:
            self._Wxx = Utils.sdiag((self.mesh.vol*self.alpha_xx)**0.5)*self.mesh.faceDivx*self.mesh.cellGradx
        return self._Wxx

    @property
    def Wyy(self):
        """Regularization matrix Wyy"""
        if getattr(self, '_Wyy', None) is None:
            self._Wyy = Utils.sdiag((self.mesh.vol*self.alpha_yy)**0.5)*self.mesh.faceDivy*self.mesh.cellGrady
        return self._Wyy

    @property
    def Wzz(self):
        """Regularization matrix Wzz"""
        if getattr(self, '_Wzz', None) is None:
            self._Wzz = Utils.sdiag((self.mesh.vol*self.alpha_zz)**0.5)*self.mesh.faceDivz*self.mesh.cellGradz
        return self._Wzz

    @property
    def W(self):
        """Full regularization matrix W"""
        if getattr(self, '_W', None) is None:
            wlist = (self.Ws, self.Wx, self.Wxx)
            if self.mesh.dim > 1:
                wlist += (self.Wy, self.Wyy)
            if self.mesh.dim > 2:
                wlist += (self.Wz, self.Wzz)
            self._W = sp.vstack(wlist)
        return self._W

