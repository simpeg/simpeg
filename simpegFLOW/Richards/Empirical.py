from SimPEG import Model, Utils, np


class RichardsModel(object):
    """docstring for RichardsModel"""

    mesh       = None  #: SimPEG mesh

    @property
    def thetaModel(self):
        """Model for moisture content"""
        return self._thetaModel

    @property
    def kModel(self):
        """Model for hydraulic conductivity"""
        return self._kModel

    def __init__(self, mesh, thetaModel, kModel):
        self.mesh = mesh
        assert isinstance(thetaModel, Model.BaseNonLinearModel)
        assert isinstance(kModel, Model.BaseNonLinearModel)

        self._thetaModel = thetaModel
        self._kModel = kModel

    def theta(self, u, m):
        return self.thetaModel.transform(u, m)

    def thetaDerivM(self, u, m):
        return self.thetaModel.transformDerivM(u, m)

    def thetaDerivU(self, u, m):
        return self.thetaModel.transformDerivU(u, m)

    def k(self, u, m):
        return self.kModel.transform(u, m)

    def kDerivM(self, u, m):
        return self.kModel.transformDerivM(u, m)

    def kDerivU(self, u, m):
        return self.kModel.transformDerivU(u, m)


def _ModelProperty(name, model, doc=None, default=None):

    def fget(self):
        if getattr(self, model, None) is not None:
            MOD = getattr(self, model)
            return getattr(MOD, name, default)
        return default

    def fset(self, value):
        if getattr(self, model, None) is not None:
            MOD = getattr(self, model)
            setattr(MOD, name, value)

    return property(fget, fset=fset, doc=doc)


class HaverkampParams(object):
    """Holds some default parameterizations for the Haverkamp model."""
    def __init__(self): pass
    @property
    def celia1990(self):
        """
            Parameters used in:

                Celia, Michael A., Efthimios T. Bouloutas, and Rebecca L. Zarba.
                "A general mass-conservative numerical solution for the unsaturated flow equation."
                Water Resources Research 26.7 (1990): 1483-1496.

        """
        return {'alpha':1.611e+06, 'beta':3.96,
                'theta_r':0.075, 'theta_s':0.287,
                'Ks':np.log(9.44e-03), 'A':1.175e+06,
                'gamma':4.74}


class _haverkamp_theta(Model.BaseNonLinearModel):

    theta_s = 0.430
    theta_r = 0.078
    alpha   = 0.036
    beta    = 3.960

    def __init__(self, mesh, **kwargs):
        Model.BaseNonLinearModel.__init__(self, mesh)
        Utils.setKwargs(self, **kwargs)

    def setModel(self, m):
        self._currentModel = m

    def transform(self, u, m):
        self.setModel(m)
        f = (self.alpha*(self.theta_s  -    self.theta_r  )/
                        (self.alpha    + abs(u)**self.beta) + self.theta_r)
        f[u >= 0] = self.theta_s
        return f

    def transformDerivM(self, u, m):
        self.setModel(m)

    def transformDerivU(self, u, m):
        self.setModel(m)
        g = (self.alpha*((self.theta_s - self.theta_r)/
             (self.alpha + abs(u)**self.beta)**2)
             *(-self.beta*abs(u)**(self.beta-1)*np.sign(u)))
        g[u >= 0] = 0
        g = Utils.sdiag(g)
        return g


class _haverkamp_k(Model.BaseNonLinearModel):

    A       = 1.175e+06
    gamma   = 4.74
    Ks      = np.log(24.96)

    def __init__(self, mesh, **kwargs):
        Model.BaseNonLinearModel.__init__(self, mesh)
        Utils.setKwargs(self, **kwargs)

    def setModel(self, m):
        self._currentModel = m
        #TODO: Fix me!
        self.Ks = m

    def transform(self, u, m):
        self.setModel(m)
        f = np.exp(self.Ks)*self.A/(self.A+abs(u)**self.gamma)
        if type(self.Ks) is np.ndarray and self.Ks.size > 1:
            f[u >= 0] = np.exp(self.Ks[u >= 0])
        else:
            f[u >= 0] = np.exp(self.Ks)
        return f

    def transformDerivM(self, u, m):
        self.setModel(m)
        #A
        # dA = np.exp(self.Ks)/(self.A+abs(u)**self.gamma) - np.exp(self.Ks)*self.A/(self.A+abs(u)**self.gamma)**2
        #gamma
        # dgamma = -(self.A*np.exp(self.Ks)*np.log(abs(u))*abs(u)**self.gamma)/(self.A + abs(u)**self.gamma)**2

        # This assumes that the the model is Ks
        return Utils.sdiag(self.transform(u, m))

    def transformDerivU(self, u, m):
        self.setModel(m)
        g = -(np.exp(self.Ks)*self.A*self.gamma*abs(u)**(self.gamma-1)*np.sign(u))/((self.A+abs(u)**self.gamma)**2)
        g[u >= 0] = 0
        g = Utils.sdiag(g)
        return g

class Haverkamp(RichardsModel):
    """Haverkamp Model"""

    alpha   = _ModelProperty('alpha',   'thetaModel', default=1.6110e+06)
    beta    = _ModelProperty('beta',    'thetaModel', default=3.96)
    theta_r = _ModelProperty('theta_r', 'thetaModel', default=0.075)
    theta_s = _ModelProperty('theta_s', 'thetaModel', default=0.287)

    Ks    = _ModelProperty('Ks',    'kModel', default=np.log(24.96))
    A     = _ModelProperty('A',     'kModel', default=1.1750e+06)
    gamma = _ModelProperty('gamma', 'kModel', default=4.74)

    def __init__(self, mesh, **kwargs):
        RichardsModel.__init__(self, mesh,
                               _haverkamp_theta(mesh),
                               _haverkamp_k(mesh))
        Utils.setKwargs(self, **kwargs)





# class Haverkamp(object):
#     """docstring for Haverkamp"""

#     empiricalModelName = "VanGenuchten"

#     theta_s = 0.430
#     theta_r = 0.078
#     alpha   = 0.036
#     beta    = 3.960
#     A       = 1.175e+06
#     gamma   = 4.74
#     Ks      = np.log(24.96)

#     def __init__(self, **kwargs):
#         Utils.setKwargs(self, **kwargs)

#     def setModel(self, m):
#         self.Ks = m

#     def moistureContent(self, h):
#         f = (self.alpha*(self.theta_s  -    self.theta_r  )/
#                         (self.alpha    + abs(h)**self.beta) + self.theta_r)
#         f[h > 0] = self.theta_s
#         return f

#     def moistureContentDeriv(self, h):
#         g = (self.alpha*((self.theta_s - self.theta_r)/
#              (self.alpha + abs(h)**self.beta)**2)
#              *(-self.beta*abs(h)**(self.beta-1)*np.sign(h)));
#         g[h >= 0] = 0
#         g = Utils.sdiag(g)
#         return g

#     def hydraulicConductivity(self, h):
#         f = np.exp(self.Ks)*self.A/(self.A+abs(h)**self.gamma)
#         if type(self.Ks) is np.ndarray and self.Ks.size > 1:
#             f[h >= 0] = np.exp(self.Ks[h >= 0])
#         else:
#             f[h >= 0] = np.exp(self.Ks)
#         return f

#     def hydraulicConductivityModelDeriv(self, h):
#         #A
#         # dA = np.exp(self.Ks)/(self.A+abs(h)**self.gamma) - np.exp(self.Ks)*self.A/(self.A+abs(h)**self.gamma)**2;
#         #gamma
#         # dgamma = -(self.A*np.exp(self.Ks)*np.log(abs(h))*abs(h)**self.gamma)/(self.A + abs(h)**self.gamma)**2;
#         return Utils.sdiag(self.hydraulicConductivity(h)) # This assumes that the the model is Ks

#     def hydraulicConductivityDeriv(self, h):
#         g = -(np.exp(self.Ks)*self.A*self.gamma*abs(h)**(self.gamma-1)*np.sign(h))/((self.A+abs(h)**self.gamma)**2)
#         g[h >= 0] = 0
#         g = Utils.sdiag(g)
#         return g


# class VanGenuchten(object):
#     """

#     .. math::

#         \\theta(h) = \\frac{\\alpha (\\theta_s - \\theta_r)}{\\alpha + |h|^\\beta} + \\theta_r

#     Where parameters alpha, beta, gamma, A are constants in the media;
#     theta_r and theta_s are the residual and saturated moisture
#     contents; and K_s is the saturated hydraulic conductivity.

#     Celia1990

#     """

#     empiricalModelName = "VanGenuchten"

#     theta_s = 0.430
#     theta_r = 0.078
#     alpha   = 0.036
#     n       = 1.560
#     beta    = 3.960
#     I       = 0.500
#     Ks      = np.log(24.96)

#     def __init__(self, **kwargs):
#         Utils.setKwargs(self, **kwargs)

#     def setModel(self, m):
#         self.Ks = m

#     def moistureContent(self, h):
#         m = 1 - 1.0/self.n;
#         f = ((  self.theta_s  -  self.theta_r  )/
#              ((1+abs(self.alpha*h)**self.n)**m)   +  self.theta_r)
#         f[h > 0] = self.theta_s
#         return f

#     def moistureContentDeriv(self, h):
#         g = -self.alpha*self.n*abs(self.alpha*h)**(self.n - 1)*np.sign(self.alpha*h)*(1./self.n - 1)*(self.theta_r - self.theta_s)*(abs(self.alpha*h)**self.n + 1)**(1./self.n - 2)
#         g[h > 0] = 0
#         g = Utils.sdiag(g)
#         return g

#     def hydraulicConductivity(self, h):
#         alpha = self.alpha
#         I = self.I
#         n = self.n
#         Ks = self.Ks
#         m = 1.0 - 1.0/n

#         theta_e = 1.0/((1.0+abs(alpha*h)**n)**m)
#         f = np.exp(Ks)*theta_e**I* ( ( 1.0 - ( 1.0 - theta_e**(1.0/m) )**m )**2 )
#         if type(self.Ks) is np.ndarray and self.Ks.size > 1:
#             f[h >= 0] = np.exp(self.Ks[h >= 0])
#         else:
#             f[h >= 0] = np.exp(self.Ks)
#         return f

#     def hydraulicConductivityModelDeriv(self, h):
#         #alpha
#         # dA = I*h*n*np.exp(Ks)*abs(alpha*h)**(n - 1)*np.sign(alpha*h)*(1.0/n - 1)*((abs(alpha*h)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2*(abs(alpha*h)**n + 1)**(1.0/n - 2) - (2*h*n*np.exp(Ks)*abs(alpha*h)**(n - 1)*np.sign(alpha*h)*(1.0/n - 1)*((abs(alpha*h)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)*(abs(alpha*h)**n + 1)**(1.0/n - 2))/(((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)*(1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n));
#         #n
#         # dn = 2*np.exp(Ks)*((np.log(1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))*(1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n))/n**2 + ((1.0/n - 1)*(((np.log(abs(alpha*h)**n + 1)*(abs(alpha*h)**n + 1)**(1.0/n - 1))/n**2 - abs(alpha*h)**n*np.log(abs(alpha*h))*(1.0/n - 1)*(abs(alpha*h)**n + 1)**(1.0/n - 2))/((1.0/n - 1)*((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)) - np.log((abs(alpha*h)**n + 1)**(1.0/n - 1))/(n**2*(1.0/n - 1)**2*((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))))/(1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n))*((abs(alpha*h)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1) - I*np.exp(Ks)*((np.log(abs(alpha*h)**n + 1)*(abs(alpha*h)**n + 1)**(1.0/n - 1))/n**2 - abs(alpha*h)**n*np.log(abs(alpha*h))*(1.0/n - 1)*(abs(alpha*h)**n + 1)**(1.0/n - 2))*((abs(alpha*h)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2;
#         #I
#         # dI = np.exp(Ks)*np.log((abs(alpha*h)**n + 1)**(1.0/n - 1))*((abs(alpha*h)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2;
#         return Utils.sdiag(self.hydraulicConductivity(h)) # This assumes that the the model is Ks

#     def hydraulicConductivityDeriv(self, h):
#         alpha = self.alpha
#         I = self.I
#         n = self.n
#         Ks = self.Ks
#         m = 1.0 - 1.0/n

#         g = I*alpha*n*np.exp(Ks)*abs(alpha*h)**(n - 1.0)*np.sign(alpha*h)*(1.0/n - 1.0)*((abs(alpha*h)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2*(abs(alpha*h)**n + 1)**(1.0/n - 2) - (2*alpha*n*np.exp(Ks)*abs(alpha*h)**(n - 1)*np.sign(alpha*h)*(1.0/n - 1)*((abs(alpha*h)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)*(abs(alpha*h)**n + 1)**(1.0/n - 2))/(((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)*(1 - 1.0/((abs(alpha*h)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n))
#         g[h >= 0] = 0
#         g = Utils.sdiag(g)
#         return g
