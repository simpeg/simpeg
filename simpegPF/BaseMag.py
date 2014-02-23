from SimPEG import Model, Data, np, sp
from scipy.constants import mu_0


class BaseMagData(Data.BaseData):
    """Base Magnetics Data"""

    rxLoc = None #: receiver locations
    rxType = None #: receiver type

    def __init__(self, **kwargs):
        Data.BaseData.__init__(**kwargs)


    #TODO: change to inc, dec, intensity
    def setBackgroundField(self, x, y, z):
        # Primary field in x-direction (background)
        self.B0 = np.r_[1.,0,0]


class BaseMagModel(Model.BaseModel):
    """BaseMagModel"""

    def __init__(self, mesh, **kwargs):
        Model.BaseModel.__init__(mesh, **kwargs)

    def transform(self, m, asMu=True):
        if asMu:
            return mu_0*(1 + m)
        return m

    def transformDeriv(self, m, asMu=True):
        if asMu:
            return mu_0*sp.identity(self.nP)
        return sp.identity(self.nP)

