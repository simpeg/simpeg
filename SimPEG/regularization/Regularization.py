from SimPEG.utils import sdiag, count, timeIt
import numpy as np

class Regularization(object):
    """docstring for Regularization"""

    @property
    def mref(self):
        if getattr(self, '_mref', None) is None:
            self._mref = np.zeros(self.mesh.nC);
        return self._mref
    @mref.setter
    def mref(self, value):
        self._mref = value

    @property
    def Ws(self):
        if getattr(self,'_Ws', None) is None:
            self._Ws = sdiag(self.mesh.vol)
        return self._Ws

    @property
    def Wx(self):
        if getattr(self, '_Wx', None) is None:
            a = self.mesh.r(self.mesh.area,'F','Fx','V')
            self._Wx = sdiag(a)*self.mesh.cellGradx
        return self._Wx

    @property
    def Wy(self):
        if getattr(self, '_Wy', None) is None:
            a = self.mesh.r(self.mesh.area,'F','Fy','V')
            self._Wy = sdiag(a)*self.mesh.cellGrady
        return self._Wy

    @property
    def Wz(self):
        if getattr(self, '_Wz', None) is None:
            a = self.mesh.r(self.mesh.area,'F','Fz','V')
            self._Wz = sdiag(a)*self.mesh.cellGradz
        return self._Wz

    alpha_s = 1e-6
    alpha_x = 1
    alpha_y = 1
    alpha_z = 1

    counter = None

    def __init__(self, mesh):
        self.mesh = mesh

    def pnorm(self, r):
        return 0.5*r.dot(r)

    @timeIt
    def modelObj(self, m):
        mresid = m - self.mref

        mobj = self.alpha_s * self.pnorm( self.Ws * mresid )

        mobj += self.alpha_x * self.pnorm( self.Wx * mresid )

        if self.mesh.dim > 1:
            mobj += self.alpha_y * self.pnorm( self.Wy * mresid )
        if self.mesh.dim > 2:
            mobj += self.alpha_z * self.pnorm( self.Wz * mresid )

        return mobj

    @timeIt
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


    @timeIt
    def modelObj2Deriv(self, m):
        mresid = m - self.mref

        mobj2Deriv = self.alpha_s * self.Ws.T * self.Ws

        mobj2Deriv = mobj2Deriv + self.alpha_x * self.Wx.T * self.Wx

        if self.mesh.dim > 1:
            mobj2Deriv = mobj2Deriv + self.alpha_y * self.Wy.T * self.Wy
        if self.mesh.dim > 2:
            mobj2Deriv = mobj2Deriv + self.alpha_z * self.Wz.T * self.Wz

        return mobj2Deriv

