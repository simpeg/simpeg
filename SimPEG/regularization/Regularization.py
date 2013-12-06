from SimPEG import utils, np

class Regularization(object):
    """docstring for Regularization"""

    __metaclass__ = utils.Save.Savable

    alpha_s = 1e-6
    alpha_x = 1.0
    alpha_y = 1.0
    alpha_z = 1.0

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
            self._Wx = self.mesh.cellGradx*utils.sdiag(self.mesh.vol)
        return self._Wx

    @property
    def Wy(self):
        if getattr(self, '_Wy', None) is None:
            self._Wy = self.mesh.cellGrady*utils.sdiag(self.mesh.vol)
        return self._Wy

    @property
    def Wz(self):
        if getattr(self, '_Wz', None) is None:
            self._Wz = self.mesh.cellGradz*utils.sdiag(self.mesh.vol)
        return self._Wz


    def pnorm(self, r):
        return 0.5*r.dot(r)

    @utils.timeIt
    def modelObj(self, m):
        mresid = m - self.mref

        mobj = self.alpha_s * self.pnorm( self.Ws * mresid )

        mobj += self.alpha_x * self.pnorm( self.Wx * mresid )

        if self.mesh.dim > 1:
            mobj += self.alpha_y * self.pnorm( self.Wy * mresid )
        if self.mesh.dim > 2:
            mobj += self.alpha_z * self.pnorm( self.Wz * mresid )

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

        if self.mesh.dim > 1:
            mobjDeriv = mobjDeriv + self.alpha_y * self.Wy.T * ( self.Wy * mresid)
        if self.mesh.dim > 2:
            mobjDeriv = mobjDeriv + self.alpha_z * self.Wz.T * ( self.Wz * mresid)

        return mobjDeriv


    @utils.timeIt
    def modelObj2Deriv(self):

        mobj2Deriv = self.alpha_s * self.Ws.T * self.Ws

        mobj2Deriv = mobj2Deriv + self.alpha_x * self.Wx.T * self.Wx

        if self.mesh.dim > 1:
            mobj2Deriv = mobj2Deriv + self.alpha_y * self.Wy.T * self.Wy
        if self.mesh.dim > 2:
            mobj2Deriv = mobj2Deriv + self.alpha_z * self.Wz.T * self.Wz

        return mobj2Deriv

