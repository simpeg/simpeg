from SimPEG import Model, Data, Utils, np, sp
from scipy.constants import mu_0


class BaseMagData(Data.BaseData):
    """Base Magnetics Data"""

    rxLoc = None #: receiver locations
    rxType = None #: receiver type

    def __init__(self, **kwargs):
        Data.BaseData.__init__(self, **kwargs)

    #TODO: change to inc, dec, intensity
    def setBackgroundField(self, x=1., y=0., z=0.):
        # Primary field in x-direction (background)
        self.B0 = np.r_[x,y,z]

    @property
    def Qfx(self):
        if getattr(self, '_Qfx', None) is None:
            self._Qfx = self.prob.mesh.getInterpolationMat(self.rxLoc,'Fx')
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, '_Qfy', None) is None:
            self._Qfy = self.prob.mesh.getInterpolationMat(self.rxLoc,'Fy')
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, '_Qfz', None) is None:
            self._Qfz = self.prob.mesh.getInterpolationMat(self.rxLoc,'Fz')
        return self._Qfz

    def projectFields(self, B):
        """
            This function projects the fields onto the data space.


            .. math::
                d_\\text{pred} = \mathbf{P} u(m)
        """

        # bfx = self.Qfx*B
        # bfy = self.Qfy*B
        bfz = self.Qfz*B
        return bfz

        # return np.sqrt(bfx**2 + bfy**2 + bfz**2)
    

        # return np.sqrt(bfx**2 + bfy**2 + bfz**2)        

    @Utils.count
    def projectFieldsDeriv(self, B):
        """
            This function projects the fields onto the data space.


            .. math::

                \\frac{\partial d_\\text{pred}}{\partial u} = \mathbf{P}
        """
        return self.Qfz


    def projectFieldsAsVector(self, B):

        bfx = self.Qfx*B
        bfy = self.Qfy*B
        bfz = self.Qfz*B

        return np.r_[bfx, bfy, bfz]

class MagDataBx(object):
    """docstring for MagDataBx"""
    def __init__(self, **kwargs):
        Data.BaseData.__init__(self, **kwargs)

    def projectFields(self, B):
        bfx = self.Qfx*B
        return bfx


class BaseMagModel(Model.BaseModel):
    """BaseMagModel"""

    def __init__(self, mesh, **kwargs):
        Model.BaseModel.__init__(self, mesh)

    def transform(self, m, asMu=True):
        if asMu:
            return mu_0*(1 + m)
        return m

    def transformDeriv(self, m, asMu=True):
        if asMu:
            return mu_0*sp.identity(self.nP)
        return sp.identity(self.nP)

