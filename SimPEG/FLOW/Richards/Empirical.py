from SimPEG import Mesh, Utils, np


class NonLinearMap(object):
    """
    SimPEG NonLinearMap

    """

    counter = None   #: A SimPEG.Utils.Counter object
    mesh = None      #: A SimPEG Mesh

    def __init__(self, mesh):
        self.mesh = mesh

    def _transform(self, u, m):
        """
            :param numpy.array u: fields
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The *transform* changes the model into the physical property.

        """
        return m

    def derivU(self, u, m):
        """
            :param numpy.array u: fields
            :param numpy.array m: model
            :rtype: scipy.sparse.csr_matrix
            :return: derivative of transformed model

            The *transform* changes the model into the physical property.
            The *transformDerivU* provides the derivative of the *transform* with respect to the fields.
        """
        raise NotImplementedError('The transformDerivU is not implemented.')


    def derivM(self, u, m):
        """
            :param numpy.array u: fields
            :param numpy.array m: model
            :rtype: scipy.sparse.csr_matrix
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


class RichardsMap(object):
    """docstring for RichardsMap"""

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
        assert isinstance(thetaModel, NonLinearMap)
        assert isinstance(kModel, NonLinearMap)

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

    def plot(self, m):
        import matplotlib.pyplot as plt

        m = m[0]
        h = np.linspace(-100, 20, 1000)
        ax = plt.subplot(121)
        ax.plot(self.theta(h, m), h)
        ax = plt.subplot(122)
        ax.semilogx(self.k(h, m), h)

    def _assertMatchesPair(self, pair):
        assert isinstance(self, pair), "Mapping object must be an instance of a {0!s} class.".format((pair.__name__))



def _ModelProperty(name, models, doc=None, default=None):

    def fget(self):
        model = models[0]
        if getattr(self, model, None) is not None:
            MOD = getattr(self, model)
            return getattr(MOD, name, default)
        return default

    def fset(self, value):
        for model in models:
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
                'Ks':9.44e-03, 'A':1.175e+06,
                'gamma':4.74}


class _haverkamp_theta(NonLinearMap):

    theta_s = 0.430
    theta_r = 0.078
    alpha   = 0.036
    beta    = 3.960

    def __init__(self, mesh, **kwargs):
        NonLinearMap.__init__(self, mesh)
        Utils.setKwargs(self, **kwargs)

    def setModel(self, m):
        self._currentModel = m

    def transform(self, u, m):
        self.setModel(m)
        f = (self.alpha*(self.theta_s  -    self.theta_r  )/
                        (self.alpha    + abs(u)**self.beta) + self.theta_r)
        if Utils.isScalar(self.theta_s):
            f[u >= 0] = self.theta_s
        else:
            f[u >= 0] = self.theta_s[u >= 0]
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


class _haverkamp_k(NonLinearMap):

    A       = 1.175e+06
    gamma   = 4.74
    Ks      = np.log(24.96)

    def __init__(self, mesh, **kwargs):
        NonLinearMap.__init__(self, mesh)
        Utils.setKwargs(self, **kwargs)

    def setModel(self, m):
        self._currentModel = m
        #TODO: Fix me!
        self.Ks = m

    def transform(self, u, m):
        self.setModel(m)
        f = np.exp(self.Ks)*self.A/(self.A+abs(u)**self.gamma)
        if Utils.isScalar(self.Ks):
            f[u >= 0] = np.exp(self.Ks)
        else:
            f[u >= 0] = np.exp(self.Ks[u >= 0])
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

class Haverkamp(RichardsMap):
    """Haverkamp Model"""

    alpha   = _ModelProperty('alpha',   ['thetaModel'], default=1.6110e+06)
    beta    = _ModelProperty('beta',    ['thetaModel'], default=3.96)
    theta_r = _ModelProperty('theta_r', ['thetaModel'], default=0.075)
    theta_s = _ModelProperty('theta_s', ['thetaModel'], default=0.287)

    Ks    = _ModelProperty('Ks',    ['kModel'], default=np.log(24.96))
    A     = _ModelProperty('A',     ['kModel'], default=1.1750e+06)
    gamma = _ModelProperty('gamma', ['kModel'], default=4.74)

    def __init__(self, mesh, **kwargs):
        RichardsMap.__init__(self, mesh,
                               _haverkamp_theta(mesh),
                               _haverkamp_k(mesh))
        Utils.setKwargs(self, **kwargs)




class _vangenuchten_theta(NonLinearMap):

    theta_s = 0.430
    theta_r = 0.078
    alpha   = 0.036
    n       = 1.560

    def __init__(self, mesh, **kwargs):
        NonLinearMap.__init__(self, mesh)
        Utils.setKwargs(self, **kwargs)

    def setModel(self, m):
        self._currentModel = m

    def transform(self, u, m):
        self.setModel(m)
        m = 1 - 1.0/self.n
        f = ((  self.theta_s  -  self.theta_r  )/
             ((1+abs(self.alpha*u)**self.n)**m)   +  self.theta_r)
        if Utils.isScalar(self.theta_s):
            f[u >= 0] = self.theta_s
        else:
            f[u >= 0] = self.theta_s[u >= 0]

        return f

    def transformDerivM(self, u, m):
        self.setModel(m)

    def transformDerivU(self, u, m):
        g = -self.alpha*self.n*abs(self.alpha*u)**(self.n - 1)*np.sign(self.alpha*u)*(1./self.n - 1)*(self.theta_r - self.theta_s)*(abs(self.alpha*u)**self.n + 1)**(1./self.n - 2)
        g[u >= 0] = 0
        g = Utils.sdiag(g)
        return g


class _vangenuchten_k(NonLinearMap):

    I       = 0.500
    alpha   = 0.036
    n       = 1.560
    Ks      = np.log(24.96)

    def __init__(self, mesh, **kwargs):
        NonLinearMap.__init__(self, mesh)
        Utils.setKwargs(self, **kwargs)

    def setModel(self, m):
        self._currentModel = m
        #TODO: Fix me!
        self.Ks = m

    def transform(self, u, m):
        self.setModel(m)

        alpha = self.alpha
        I = self.I
        n = self.n
        Ks = self.Ks
        m = 1.0 - 1.0/n

        theta_e = 1.0/((1.0+abs(alpha*u)**n)**m)
        f = np.exp(Ks)*theta_e**I* ( ( 1.0 - ( 1.0 - theta_e**(1.0/m) )**m )**2 )
        if Utils.isScalar(self.Ks):
            f[u >= 0] = np.exp(self.Ks)
        else:
            f[u >= 0] = np.exp(self.Ks[u >= 0])
        return f

    def transformDerivM(self, u, m):
        self.setModel(m)
#         #alpha
#         # dA = I*u*n*np.exp(Ks)*abs(alpha*u)**(n - 1)*np.sign(alpha*u)*(1.0/n - 1)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2*(abs(alpha*u)**n + 1)**(1.0/n - 2) - (2*u*n*np.exp(Ks)*abs(alpha*u)**(n - 1)*np.sign(alpha*u)*(1.0/n - 1)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)*(abs(alpha*u)**n + 1)**(1.0/n - 2))/(((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)*(1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n));
#         #n
#         # dn = 2*np.exp(Ks)*((np.log(1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))*(1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n))/n**2 + ((1.0/n - 1)*(((np.log(abs(alpha*u)**n + 1)*(abs(alpha*u)**n + 1)**(1.0/n - 1))/n**2 - abs(alpha*u)**n*np.log(abs(alpha*u))*(1.0/n - 1)*(abs(alpha*u)**n + 1)**(1.0/n - 2))/((1.0/n - 1)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)) - np.log((abs(alpha*u)**n + 1)**(1.0/n - 1))/(n**2*(1.0/n - 1)**2*((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))))/(1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n))*((abs(alpha*u)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1) - I*np.exp(Ks)*((np.log(abs(alpha*u)**n + 1)*(abs(alpha*u)**n + 1)**(1.0/n - 1))/n**2 - abs(alpha*u)**n*np.log(abs(alpha*u))*(1.0/n - 1)*(abs(alpha*u)**n + 1)**(1.0/n - 2))*((abs(alpha*u)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2;
#         #I
#         # dI = np.exp(Ks)*np.log((abs(alpha*u)**n + 1)**(1.0/n - 1))*((abs(alpha*u)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2;
        return Utils.sdiag(self.transform(u, m)) # This assumes that the the model is Ks

    def transformDerivU(self, u, m):
        self.setModel(m)
        alpha = self.alpha
        I = self.I
        n = self.n
        Ks = self.Ks
        m = 1.0 - 1.0/n

        g = I*alpha*n*np.exp(Ks)*abs(alpha*u)**(n - 1.0)*np.sign(alpha*u)*(1.0/n - 1.0)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2*(abs(alpha*u)**n + 1)**(1.0/n - 2) - (2*alpha*n*np.exp(Ks)*abs(alpha*u)**(n - 1)*np.sign(alpha*u)*(1.0/n - 1)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)*(abs(alpha*u)**n + 1)**(1.0/n - 2))/(((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)*(1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n))
        g[u >= 0] = 0
        g = Utils.sdiag(g)
        return g

class VanGenuchten(RichardsMap):
    """vanGenuchten Model"""

    theta_r = _ModelProperty('theta_r', ['thetaModel'], default=0.075)
    theta_s = _ModelProperty('theta_s', ['thetaModel'], default=0.287)

    alpha   = _ModelProperty('alpha',   ['thetaModel', 'kModel'], default=0.036)
    n       = _ModelProperty('n',       ['thetaModel', 'kModel'], default=1.560)

    Ks    = _ModelProperty('Ks',    ['kModel'], default=np.log(24.96))
    I     = _ModelProperty('I',     ['kModel'], default=0.500)

    def __init__(self, mesh, **kwargs):
        RichardsMap.__init__(self, mesh,
                               _vangenuchten_theta(mesh),
                               _vangenuchten_k(mesh))
        Utils.setKwargs(self, **kwargs)


class VanGenuchtenParams(object):
    """
        The RETC code for quantifying the hydraulic functions of unsaturated soils,
        Van Genuchten, M Th, Leij, F J, Yates, S R

        Table 3: Average values for selected soil water retention and hydraulic
        conductivity parameters for 11 major soil textural groups
        according to Rawls et al. [1982]

    """
    def __init__(self): pass
    @property
    def sand(self):
        return {"theta_r": 0.020, "theta_s": 0.417, "alpha": 0.138*100., "n": 1.592, "Ks": 504.0/100./24./60./60.}
    @property
    def loamySand(self):
        return {"theta_r": 0.035, "theta_s": 0.401, "alpha": 0.115*100., "n": 1.474, "Ks": 146.6/100./24./60./60.}
    @property
    def sandyLoam(self):
        return {"theta_r": 0.041, "theta_s": 0.412, "alpha": 0.068*100., "n": 1.322, "Ks": 62.16/100./24./60./60.}
    @property
    def loam(self):
        return {"theta_r": 0.027, "theta_s": 0.434, "alpha": 0.090*100., "n": 1.220, "Ks": 16.32/100./24./60./60.}
    @property
    def siltLoam(self):
        return {"theta_r": 0.015, "theta_s": 0.486, "alpha": 0.048*100., "n": 1.211, "Ks": 31.68/100./24./60./60.}
    @property
    def sandyClayLoam(self):
        return {"theta_r": 0.068, "theta_s": 0.330, "alpha": 0.036*100., "n": 1.250, "Ks": 10.32/100./24./60./60.}
    @property
    def clayLoam(self):
        return {"theta_r": 0.075, "theta_s": 0.390, "alpha": 0.039*100., "n": 1.194, "Ks": 5.52/100./24./60./60.}
    @property
    def siltyClayLoam(self):
        return {"theta_r": 0.040, "theta_s": 0.432, "alpha": 0.031*100., "n": 1.151, "Ks": 3.60/100./24./60./60.}
    @property
    def sandyClay(self):
        return {"theta_r": 0.109, "theta_s": 0.321, "alpha": 0.034*100., "n": 1.168, "Ks": 2.88/100./24./60./60.}
    @property
    def siltyClay(self):
        return {"theta_r": 0.056, "theta_s": 0.423, "alpha": 0.029*100., "n": 1.127, "Ks": 2.16/100./24./60./60.}
    @property
    def clay(self):
        return {"theta_r": 0.090, "theta_s": 0.385, "alpha": 0.027*100., "n": 1.131, "Ks": 1.44/100./24./60./60.}


    # From: INDIRECT METHODS FOR ESTIMATING THE HYDRAULIC PROPERTIES OF UNSATURATED SOILS
    # @property
    # def siltLoamGE3(self):
    #     """Soil Index: 3310"""
    #     return {"theta_r": 0.139, "theta_s": 0.394, "alpha": 0.00414, "n": 2.15}
    # @property
    # def yoloLightClayK_WC(self):
    #     """Soil Index: None"""
    #     return {"theta_r": 0.205, "theta_s": 0.499, "alpha": 0.02793, "n": 1.71}
    # @property
    # def yoloLightClayK_H(self):
    #     """Soil Index: None"""
    #     return {"theta_r": 0.205, "theta_s": 0.499, "alpha": 0.02793, "n": 1.71}
    # @property
    # def hygieneSandstone(self):
    #     """Soil Index: 4130"""
    #     return {"theta_r": 0.000, "theta_s": 0.256, "alpha": 0.00562, "n": 3.27}
    # @property
    # def lambcrgClay(self):
    #     """Soil Index: 1003"""
    #     return {"theta_r": 0.000, "theta_s": 0.502, "alpha": 0.140, "n": 1.93}
    # @property
    # def beitNetofaClaySoil(self):
    #     """Soil Index: 1006"""
    #     return {"theta_r": 0.000, "theta_s": 0.447, "alpha": 0.00156, "n": 1.17}
    # @property
    # def shiohotSiltyClay(self):
    #     """Soil Index: 1101"""
    #     return {"theta_r": 0.000, "theta_s": 0.456, "alpha": 183, "n":1.17}
    # @property
    # def siltColumbia(self):
    #     """Soil Index: 2001"""
    #     return {"theta_r": 0.146, "theta_s": 0.397,  "alpha": 0.0145, "n": 1.85}
    # @property
    # def siltMontCenis(self):
    #     """Soil Index: 2002"""
    #     return {"theta_r": 0.000, "theta_s": 0.425, "alpha": 0.0103, "n": 1.34}
    # @property
    # def slateDust(self):
    #     """Soil Index: 2004"""
    #     return {"theta_r": 0.000, "theta_s": 0.498, "alpha": 0.00981, "n": 6.75}
    # @property
    # def weldSiltyClayLoam(self):
    #     """Soil Index: 3001"""
    #     return {"theta_r": 0.159, "theta_s": 0.496, "alpha": 0.0136, "n": 5.45}
    # @property
    # def rideauClayLoam_Wetting(self):
    #     """Soil Index: 3101a"""
    #     return {"theta_r": 0.279, "theta_s": 0.419, "alpha": 0.0661, "n": 1.89}
    # @property
    # def rideauClayLoam_Drying(self):
    #     """Soil Index: 3101b"""
    #     return {"theta_r": 0.290, "theta_s": 0.419, "alpha": 0.0177, "n": 3.18}
    # @property
    # def caribouSiltLoam_Drying(self):
    #     """Soil Index: 3301a"""
    #     return {"theta_r": 0.000, "theta_s": 0.451, "alpha": 0.00845, "n": 1.29}
    # @property
    # def caribouSiltLoam_Wetting(self):
    #     """Soil Index: 3301b"""
    #     return {"theta_r": 0.000, "theta_s": 0.450, "alpha": 0.140, "n": 1.09}
    # @property
    # def grenvilleSiltLoam_Wetting(self):
    #     """Soil Index: 3302a"""
    #     return {"theta_r": 0.013, "theta_s": 0523,  "alpha": 0.0630, "n": 1.24}
    # @property
    # def grenvilleSiltLoam_Drying(self):
    #     """Soil Index: 3302c"""
    #     return {"theta_r": 0.000, "theta_s": 0.488, "alpha": 0.0112, "n": 1.23}
    # @property
    # def touchetSiltLoam(self):
    #     """Soil Index: 3304"""
    #     return {"theta_r": 0.183, "theta_s": 0.498, "alpha": 0.0104, "n": 5.78}
    # @property
    # def gilatLoam(self):
    #     """Soil Index: 3402a"""
    #     return {"theta_r": 0.000, "theta_s": 0.454, "alpha": 0.0291, "n": 1.47}
    # @property
    # def pachapaLoam(self):
    #     """Soil Index: 3403"""
    #     return {"theta_r": 0.000, "theta_s": 0.472, "alpha": 0.00829, "n": 1.62}
    # @property
    # def adelantoLoam(self):
    #     """Soil Index: 3404"""
    #     return {"theta_r": 0.000, "theta_s": 0.444, "alpha": 0.00710, "n": 1.26}
    # @property
    # def indioLoam(self):
    #     """Soil Index: 3405a"""
    #     return {"theta_r": 0.000, "theta_s": 0.507, "alpha": 0.00847, "n": 1.60}
    # @property
    # def guclphLoam(self):
    #     """Soil Index: 3407a"""
    #     return {"theta_r": 0.000, "theta_s": 0.563, "alpha": 0.0275, "n": 1.27}
    # @property
    # def guclphLoam(self):
    #     """Soil Index: 3407b"""
    #     return {"theta_r": 0.236, "theta_s": 0.435, "alpha": 0.0271, "n": 262}
    # @property
    # def rubiconSandyLoam(self):
    #     """Soil Index: 3501a"""
    #     return {"theta_r": 0.000, "theta_s": 0.393,  "alpha": 0.00972, "n": 2.18}
    # @property
    # def rubiconSandyLoam(self):
    #     """Soil Index: 350lb"""
    #     return {"theta_r": 0.000, "theta_s": 0.433, "alpha": 0.147, "n": 1.28}
    # @property
    # def pachapaFmeSandyClay(self):
    #     """Soil Index: 3503a"""
    #     return {"theta_r": 0.000, "theta_s": 0.340, "alpha": 0.0194, "n": 1.45}
    # @property
    # def gilatSandyLoam(self):
    #     """Soil Index: 3504"""
    #     return {"theta_r": 0.000, "theta_s": 0.432, "alpha": 0.0103, "n": 1.48}
    # @property
    # def plainfieldSand_210to250(self):
    #     """Soil Index: 4101a"""
    #     return {"theta_r": 0.000, "theta_s": 0.351, "alpha": 0.0236, "n": 12.30}
    # @property
    # def plainfieldSand_210to250(self):
    #     """Soil Index: 4101b"""
    #     return {"theta_r": 0.000, "theta_s": 0.312, "alpha": 0.0387, "n": 4.48}
    # @property
    # def plainfieldSand_177to210(self):
    #     """Soil Index: 4102a"""
    #     return {"theta_r": 0.000, "theta_s": 0.361, "alpha": 0.0207, "n": 10.0}
    # @property
    # def plainfieldSand_177to210(self):
    #     """Soil Index: 4102b"""
    #     return {"theta_r": 0.022, "theta_s": 0.309, "alpha": 0.0328, "n": 6.23}
    # @property
    # def plainfieldSand_149to177(self):
    #     """Soil Index: 4103a"""
    #     return {"theta_r": 0.000, "theta_s": 0.387, "alpha": 0.0173, "n": 7.80}
    # @property
    # def plainfieldSand_149to177(self):
    #     """Soil Index: 4103b"""
    #     return {"theta_r": 0.025, "theta_s": 0.321, "alpha": 0.0272, "n": 6.69}
    # @property
    # def plainfieldSand_l25to149(self):
    #     """Soil Index: 4104a"""
    #     return {"theta_r": 0.000, "theta_s": 03770, "alpha": 0.0145, "n": 10.60}
    # @property
    # def plainfieldSand_125to149(self):
    #     """Soil Index: 4104b"""
    #     return {"theta_r": 0.000, "theta_s": 0.342, "alpha": 0.0230, "n": 5.18}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    M = Mesh.TensorMesh([10])
    VGparams = VanGenuchtenParams()
    leg = []
    for p in dir(VGparams):
        if p[0] == '_': continue
        leg += [p]
        params = getattr(VGparams, p)
        model = VanGenuchten(M, **params)
        ks = np.log(np.r_[params['Ks']])
        model.plot(ks)

    plt.legend(leg)

    plt.show()
