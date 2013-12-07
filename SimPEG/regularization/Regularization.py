from SimPEG import utils, np, sp

class Regularization(object):
    """**Regularization**

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

    __metaclass__ = utils.Save.Savable

    alpha_s  = 1e-6  #: Smallness weight
    alpha_x  = 1.0   #: Weight for the first derivative in the x direction
    alpha_y  = 1.0   #: Weight for the first derivative in the y direction
    alpha_z  = 1.0   #: Weight for the first derivative in the z direction
    alpha_xx = 0.0   #: Weight for the second derivative in the x direction
    alpha_yy = 0.0   #: Weight for the second derivative in the y direction
    alpha_zz = 0.0   #: Weight for the second derivative in the z direction

    counter = None

    def __init__(self, mesh, **kwargs):
        utils.setKwargs(self, **kwargs)
        self.mesh = mesh

    @property
    def mref(self):
        if getattr(self, '_mref', None) is None:
            return np.zeros(self.mesh.nC);
        return self._mref
    @mref.setter
    def mref(self, value):
        self._mref = value

    @property
    def Ws(self):
        if getattr(self,'_Ws', None) is None:
            self._Ws = utils.sdiag(self.mesh.vol)
        return self._Ws

    @property
    def Wx(self):
        if getattr(self, '_Wx', None) is None:
            Ave_x_vol = self.mesh.aveCC2F[:self.mesh.nFv[0],:]*self.mesh.vol
            self._Wx = utils.sdiag(Ave_x_vol**0.5)*self.mesh.cellGradx
        return self._Wx

    @property
    def Wy(self):
        if getattr(self, '_Wy', None) is None:
            Ave_y_vol = self.mesh.aveCC2F[self.mesh.nFv[0]:np.sum(self.mesh.nFv[:2]),:]*self.mesh.vol
            self._Wy = utils.sdiag(Ave_y_vol**0.5)*self.mesh.cellGrady
        return self._Wy

    @property
    def Wz(self):
        if getattr(self, '_Wz', None) is None:
            Ave_z_vol = self.mesh.aveCC2F[np.sum(self.mesh.nFv[:2]):,:]*self.mesh.vol
            self._Wz = utils.sdiag(Ave_z_vol**0.5)*self.mesh.cellGradz
        return self._Wz

    @property
    def Wxx(self):
        if getattr(self, '_Wxx', None) is None:
            self._Wxx = self.mesh.faceDivx*self.mesh.cellGradx*utils.sdiag(self.mesh.vol)
        return self._Wxx

    @property
    def Wyy(self):
        if getattr(self, '_Wyy', None) is None:
            self._Wyy = self.mesh.faceDivy*self.mesh.cellGrady*utils.sdiag(self.mesh.vol)
        return self._Wyy

    @property
    def Wzz(self):
        if getattr(self, '_Wzz', None) is None:
            self._Wzz = self.mesh.faceDivz*self.mesh.cellGradz*utils.sdiag(self.mesh.vol)
        return self._Wzz


    def pnorm(self, r):
        return 0.5*r.dot(r)

    @utils.timeIt
    def modelObj(self, m):
        mresid = m - self.mref

        mobj = self.alpha_s * self.pnorm( self.Ws * mresid )

        mobj += self.alpha_x * self.pnorm( self.Wx * mresid )
        mobj += self.alpha_xx * self.pnorm( self.Wxx * mresid )

        if self.mesh.dim > 1:
            mobj += self.alpha_y * self.pnorm( self.Wy * mresid )
            mobj += self.alpha_yy * self.pnorm( self.Wyy * mresid )
        if self.mesh.dim > 2:
            mobj += self.alpha_z * self.pnorm( self.Wz * mresid )
            mobj += self.alpha_zz * self.pnorm( self.Wzz * mresid )

        return mobj

    @utils.timeIt
    def modelObjDeriv(self, m):
        """

        In 1D:

        .. math::

            m_{\\text{obj}} = {1 \over 2}\\alpha_s  \left\| W_s  (m- m_{\\text{ref}})\\right\|^2_2
                            + {1 \over 2}\\alpha_x  \left\| W_x  (m- m_{\\text{ref}})\\right\|^2_2

            \\frac{ \partial m_{\\text{obj}} }{\partial m} =
                            \\alpha_s  W_s^{\\top} W_s  (m - m_{\\text{ref}}) +
                            \\alpha_x  W_x^{\\top} W_x  (m - m_{\\text{ref}})


            \\frac{ \partial^2 m_{\\text{obj}} }{\partial m^2} =
                            \\alpha_s  W_s^{\\top} W_s +
                            \\alpha_x  W_x^{\\top} W_x

        """

        mresid = m - self.mref

        mobjDeriv = self.alpha_s * self.Ws.T * ( self.Ws * mresid)

        mobjDeriv = mobjDeriv + self.alpha_x * self.Wx.T * ( self.Wx * mresid)
        mobjDeriv = mobjDeriv + self.alpha_xx * self.Wxx.T * ( self.Wxx * mresid)

        if self.mesh.dim > 1:
            mobjDeriv = mobjDeriv + self.alpha_y * self.Wy.T * ( self.Wy * mresid)
            mobjDeriv = mobjDeriv + self.alpha_yy * self.Wyy.T * ( self.Wyy * mresid)
        if self.mesh.dim > 2:
            mobjDeriv = mobjDeriv + self.alpha_z * self.Wz.T * ( self.Wz * mresid)
            mobjDeriv = mobjDeriv + self.alpha_zz * self.Wzz.T * ( self.Wzz * mresid)

        return mobjDeriv


    @utils.timeIt
    def modelObj2Deriv(self):

        mobj2Deriv = self.alpha_s * self.Ws.T * self.Ws

        mobj2Deriv = mobj2Deriv + self.alpha_x * self.Wx.T * self.Wx
        mobj2Deriv = mobj2Deriv + self.alpha_xx * self.Wxx.T * self.Wxx

        if self.mesh.dim > 1:
            mobj2Deriv = mobj2Deriv + self.alpha_y * self.Wy.T * self.Wy
            mobj2Deriv = mobj2Deriv + self.alpha_yy * self.Wyy.T * self.Wyy
        if self.mesh.dim > 2:
            mobj2Deriv = mobj2Deriv + self.alpha_z * self.Wz.T * self.Wz
            mobj2Deriv = mobj2Deriv + self.alpha_zz * self.Wzz.T * self.Wzz

        return mobj2Deriv

