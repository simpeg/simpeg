from SimPEG.forward import Problem
import numpy as np
from SimPEG.utils import sdiag, mkvc, setKwargs, Solver
from SimPEG.inverse import NewtonRoot


class RichardsProblem(Problem):
    """docstring for RichardsProblem"""

    timeStep = None
    timeEnd  = None
    boundaryConditions = None
    initialConditions  = None

    @property
    def numIts(self):
        """The number of iterations in the time domain problem."""
        return int(self.timeEnd/self.timeStep)

    _method = 'mixed'
    @property
    def method(self):
        """

            Method must be either 'mixed' or 'head'.

            There are two different forms of Richards equation that differ
            on how they deal with the non-linearity in the time-stepping term.

            The most fundamental form, referred to as the
            'mixed'-form of Richards Equation [Celia et al., 1990]

            .. math::

                \\frac{\partial \\theta(\psi)}{\partial t} - \\nabla \cdot k(\psi) \\nabla \psi - \\frac{\partial k(\psi)}{\partial z} = 0
                \quad \psi \in \Omega

            where theta is water content, and psi is pressure head.
            This formulation of Richards equation is called the
            'mixed'-form because the equation is parameterized in psi
            but the time-stepping is in terms of theta.

            As noted in [Celia et al., 1990] the 'head'-based form of Richards
            equation can be written in the continuous form as:

            .. math::

                \\frac{\partial \\theta}{\partial \psi}\\frac{\partial \psi}{\partial t} - \\nabla \cdot k(\psi) \\nabla \psi - \\frac{\partial k(\psi)}{\partial z} = 0
                \quad \psi \in \Omega

            However, it can be shown that this does not conserve mass in the discrete formulation.


        """
        return self._method
    @method.setter
    def method(self, value):
        assert value in ['mixed','head'], "method must be 'mixed' or 'head'."
        self._method = value

    @property
    def doNewton(self):
        """Do a Newton iteration. If False, a Picard iteration will be completed."""
        return self._doNewton
    @doNewton.setter
    def doNewton(self, value):
        assert type(value) is bool, 'doNewton must be a boolean.'
        self.rootFinder = NewtonRoot(doLS=value)
        self._doNewton = value

    @property
    def dataType(self):
        """Choose how your data is collected, must be 'saturation' or 'pressureHead'."""
        return self._dataType
    @dataType.setter
    def dataType(self, value):
        assert value in ['saturation','pressureHead'], "dataType must be 'saturation' or 'pressureHead'."
        self._dataType = value


    def __init__(self, mesh, empirical, **kwargs):
        Problem.__init__(self, mesh)
        self.empirical = empirical
        self.mesh.setCellGradBC('dirichlet')
        self.dataType = 'pressureHead'
        self.doNewton = False  # This also sets the rootFinder algorithm.
        setKwargs(self, **kwargs)

    def field(self, m):
        self.empirical.setModel(m)
        Hs = range(self.numIts+1)
        Hs[0] = self.initialConditions
        for ii in range(self.numIts):
            Hs[ii+1] = self.rootFinder.root(lambda hn1: self.getResidual(Hs[ii],hn1), Hs[ii])
        return Hs

    def diagsJacobian(self, hn, hn1):

        DIV  = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC   = self.mesh.cellGradBC
        AV   = self.mesh.aveCC2F
        Dz   = self.mesh.faceDiv #TODO: fix this for more than one dimension.

        bc   = self.boundaryConditions
        dt   = self.timeStep

        dT   = self.empirical.moistureContentDeriv(hn)
        dT1  = self.empirical.moistureContentDeriv(hn1)
        K1   = self.empirical.hydraulicConductivity(hn1)
        dK1  = self.empirical.hydraulicConductivityDeriv(hn1)
        dKa1 = self.empirical.hydraulicConductivityModelDeriv(hn1)

        # Compute part of the derivative of:
        #
        #       DIV*diag(GRAD*hn1+BC*bc)*(AV*(1/K))^-1

        DdiagGh1 = DIV*sdiag(GRAD*hn1+BC*bc)
        diagAVk2_AVdiagK2 = sdiag((AV*(1./K1))**(-2)) * AV*sdiag(K1**(-2))

        # The matrix that we are computing has the form:
        #
        #   -                                      -   -  -     -  -
        #  |  Adiag                                 | | h1 |   | b1 |
        #  |   Asub    Adiag                        | | h2 |   | b2 |
        #  |            Asub    Adiag               | | h3 | = | b3 |
        #  |                 ...     ...            | | .. |   | .. |
        #  |                         Asub    Adiag  | | hn |   | bn |
        #   -                                      -   -  -     -  -

        Asub = (-1/dt)*dT

        Adiag = (
                  (1/dt)*dT1
                 -DdiagGh1*diagAVk2_AVdiagK2*dK1
                 -DIV*sdiag(1./(AV*(1./K1)))*GRAD
                 -Dz*diagAVk2_AVdiagK2*dK1
                )

        B = DdiagGh1*diagAVk2_AVdiagK2*dKa1 + Dz*diagAVk2_AVdiagK2*dKa1

        return Asub, Adiag, B

    def getResidual(self, hn, h):
        """
            Where h is the proposed value for the next time iterate (h_{n+1})
        """
        DIV  = self.mesh.faceDiv
        GRAD = self.mesh.cellGrad
        BC   = self.mesh.cellGradBC
        AV   = self.mesh.aveCC2F
        Dz   = self.mesh.faceDiv #TODO: fix this for more than one dimension.

        bc = self.boundaryConditions
        dt = self.timeStep

        T  = self.empirical.moistureContent(h)
        dT = self.empirical.moistureContentDeriv(h)
        Tn = self.empirical.moistureContent(hn)
        K  = self.empirical.hydraulicConductivity(h)
        dK = self.empirical.hydraulicConductivityDeriv(h)

        aveK = 1./(AV*(1./K));

        RHS = DIV*sdiag(aveK)*(GRAD*h+BC*bc) + Dz*aveK
        if self.method is 'mixed':
            r = (T-Tn)/dt - RHS
        elif self.method is 'head':
            r = dT*(h - hn)/dt - RHS

        J = dT/dt - DIV*sdiag(aveK)*GRAD
        if self.doNewton:
            DDharmAve = sdiag(aveK**2)*AV*sdiag(K**(-2)) * dK
            J = J - DIV*sdiag(GRAD*h + BC*bc)*DDharmAve - Dz*DDharmAve

        return r, J

    def J(self, m, v, u=None):
        if u is None:
            u = self.field(m)
        Hs = u
        JvC = range(len(Hs)-1) # Cell to hold each row of the long vector.

        # This is done via forward substitution.
        temp, Adiag, B = self.diagsJacobian(Hs[0],Hs[1])
        Adiaginv = Solver(Adiag)
        JvC[0] = Adiaginv.solve(B*v)

        # M = @(x) tril(Adiag)\(diag(Adiag).*(triu(Adiag)\x));
        # JvC{1} = bicgstab(Adiag,(B*v),tolbcg,500,M);

        for ii in range(1,len(Hs)-1):
            Asub, Adiag, B = self.diagsJacobian(Hs[ii],Hs[ii+1])
            Adiaginv = Solver(Adiag)
            JvC[ii] = Adiaginv.solve(B*v - Asub*JvC[ii-1])

        if self.dataType is 'pressureHead':
            Jv = self.P*np.concatenate(JvC)
        elif self.dataType is 'saturation':
            dT = self.empirical.moistureContentDeriv(np.concatenate(Hs[1:]))
            Jv = self.P*dT*np.concatenate(JvC)

        return Jv

    def Jt(self, m, v, u=None):
        if u is None:
            u = self.field(m)
        Hs = u

        if self.dataType is 'pressureHead':
            PTv = self.P.T*v;
        elif self.dataType is 'saturation':
            dT = self.empirical.moistureContentDeriv(np.concatenate(Hs[1:]))
            PTv = dT.T*self.P.T*v

        # This is done via backward substitution.
        minus = 0
        BJtv = 0
        for ii in range(len(Hs)-1,0,-1):
            Asub, Adiag, B = self.diagsJacobian(Hs[ii-1], Hs[ii])
            #select the correct part of v
            vpart = range((ii-1)*Adiag.shape[0], (ii)*Adiag.shape[0])
            AdiaginvT = Solver(Adiag.T)
            JTvC = AdiaginvT.solve(PTv[vpart] - minus)
            minus = Asub.T*JTvC  # this is now the super diagonal.
            BJtv = BJtv + B.T*JTvC

        return BJtv


class Haverkamp(object):
    """docstring for Haverkamp"""

    empiricalModelName = "VanGenuchten"

    theta_s = 0.430
    theta_r = 0.078
    alpha   = 0.036
    beta    = 3.960
    A       = 1.175e+06
    gamma   = 4.74
    Ks      = np.log(24.96)

    def __init__(self, **kwargs):
        setKwargs(self, **kwargs)

    def setModel(self, m):
        self.Ks = m

    def moistureContent(self, h):
        f = (self.alpha*(self.theta_s  -    self.theta_r  )/
                        (self.alpha    + abs(h)**self.beta) + self.theta_r)
        f[h > 0] = self.theta_s
        return f

    def moistureContentDeriv(self, h):
        g = (self.alpha*((self.theta_s - self.theta_r)/
             (self.alpha + abs(h)**self.beta)**2)
             *(-self.beta*abs(h)**(self.beta-1)*np.sign(h)));
        g[h >= 0] = 0
        g = sdiag(g)
        return g

    def hydraulicConductivity(self, h):
        f = np.exp(self.Ks)*self.A/(self.A+abs(h)**self.gamma)
        if type(self.Ks) is np.ndarray and self.Ks.size > 1:
            f[h >= 0] = np.exp(self.Ks[h >= 0])
        else:
            f[h >= 0] = np.exp(self.Ks)
        return f

    def hydraulicConductivityModelDeriv(self, h):
        #A
        # dA = np.exp(self.Ks)/(self.A+abs(h)**self.gamma) - np.exp(self.Ks)*self.A/(self.A+abs(h)**self.gamma)**2;
        #gamma
        # dgamma = -(self.A*np.exp(self.Ks)*np.log(abs(h))*abs(h)**self.gamma)/(self.A + abs(h)**self.gamma)**2;
        return sdiag(self.hydraulicConductivity(h)) # This assumes that the the model is Ks

    def hydraulicConductivityDeriv(self, h):
        g = -(np.exp(self.Ks)*self.A*self.gamma*abs(h)**(self.gamma-1)*np.sign(h))/((self.A+abs(h)**self.gamma)**2)
        g[h >= 0] = 0
        g = sdiag(g)
        return g


class VanGenuchten(object):
    """

    .. math::

        \\theta(h) = \\frac{\\alpha (\\theta_s - \\theta_r)}{\\alpha + |h|^\\beta} + \\theta_r

    Where parameters alpha, beta, gamma, A are constants in the media;
    theta_r and theta_s are the residual and saturated moisture
    contents; and K_s is the saturated hydraulic conductivity.

    Celia1990

    """

    empiricalModelName = "VanGenuchten"

    theta_s = 0.430
    theta_r = 0.078
    alpha   = 0.036
    n       = 1.560
    beta    = 3.960
    I       = 0.500
    Ks      = np.log(24.96)

    def __init__(self, **kwargs):
        setKwargs(self, **kwargs)

    def setModel(self, m):
        self.Ks = m

    def moistureContent(self, h):
        m = 1 - 1/self.n;
        f = ((  self.theta_s  -  self.theta_r  )/
             ((1+abs(self.alpha*h)**self.n)**m)   +  self.theta_r)
        f[h > 0] = self.theta_s
        return f

    def moistureContentDeriv(self, h):
        g = -self.alpha*self.n*abs(self.alpha*h)**(self.n - 1)*np.sign(self.alpha*h)*(1./self.n - 1)*(self.theta_r - self.theta_s)*(abs(self.alpha*h)**self.n + 1)**(1./self.n - 2)
        g[h > 0] = 0
        g = sdiag(g)
        return g

    def hydraulicConductivity(self, h):
        alpha = self.alpha
        I = self.I
        n = self.n
        Ks = self.Ks
        m = 1 - 1/n

        theta_e = 1/((1+abs(alpha*h)**n)**m)
        f = np.exp(Ks)*theta_e**I* ( ( 1 - ( 1 - theta_e**(1/m) )**m )**2 )
        if type(self.Ks) is np.ndarray and self.Ks.size > 1:
            f[h >= 0] = np.exp(self.Ks[h >= 0])
        else:
            f[h >= 0] = np.exp(self.Ks)
        return f

    def hydraulicConductivityModelDeriv(self, h):
        #alpha
        # dA = I*h*n*np.exp(Ks)*abs(alpha*h)**(n - 1)*np.sign(alpha*h)*(1/n - 1)*((abs(alpha*h)**n + 1)**(1/n - 1))**(I - 1)*((1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n) - 1)**2*(abs(alpha*h)**n + 1)**(1/n - 2) - (2*h*n*np.exp(Ks)*abs(alpha*h)**(n - 1)*np.sign(alpha*h)*(1/n - 1)*((abs(alpha*h)**n + 1)**(1/n - 1))**I*((1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n) - 1)*(abs(alpha*h)**n + 1)**(1/n - 2))/(((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1) + 1)*(1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1/n));
        #n
        # dn = 2*np.exp(Ks)*((np.log(1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))*(1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n))/n**2 + ((1/n - 1)*(((np.log(abs(alpha*h)**n + 1)*(abs(alpha*h)**n + 1)**(1/n - 1))/n**2 - abs(alpha*h)**n*np.log(abs(alpha*h))*(1/n - 1)*(abs(alpha*h)**n + 1)**(1/n - 2))/((1/n - 1)*((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1) + 1)) - np.log((abs(alpha*h)**n + 1)**(1/n - 1))/(n**2*(1/n - 1)**2*((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))))/(1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1/n))*((abs(alpha*h)**n + 1)**(1/n - 1))**I*((1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n) - 1) - I*np.exp(Ks)*((np.log(abs(alpha*h)**n + 1)*(abs(alpha*h)**n + 1)**(1/n - 1))/n**2 - abs(alpha*h)**n*np.log(abs(alpha*h))*(1/n - 1)*(abs(alpha*h)**n + 1)**(1/n - 2))*((abs(alpha*h)**n + 1)**(1/n - 1))**(I - 1)*((1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n) - 1)**2;
        #I
        # dI = np.exp(Ks)*np.log((abs(alpha*h)**n + 1)**(1/n - 1))*((abs(alpha*h)**n + 1)**(1/n - 1))**I*((1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n) - 1)**2;
        return sdiag(self.hydraulicConductivity(h)) # This assumes that the the model is Ks

    def hydraulicConductivityDeriv(self, h):
        alpha = self.alpha
        I = self.I
        n = self.n
        Ks = self.Ks
        m = 1 - 1/n

        g = I*alpha*n*np.exp(Ks)*abs(alpha*h)**(n - 1)*np.sign(alpha*h)*(1/n - 1)*((abs(alpha*h)**n + 1)**(1/n - 1))**(I - 1)*((1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n) - 1)**2*(abs(alpha*h)**n + 1)**(1/n - 2) - (2*alpha*n*np.exp(Ks)*abs(alpha*h)**(n - 1)*np.sign(alpha*h)*(1/n - 1)*((abs(alpha*h)**n + 1)**(1/n - 1))**I*((1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1 - 1/n) - 1)*(abs(alpha*h)**n + 1)**(1/n - 2))/(((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1) + 1)*(1 - 1/((abs(alpha*h)**n + 1)**(1/n - 1))**(1/(1/n - 1)))**(1/n))
        g[h >= 0] = 0
        g = sdiag(g)
        return g


if __name__ == '__main__':
    from SimPEG.mesh import TensorMesh
    from SimPEG.tests import checkDerivative
    M = TensorMesh([np.ones(40)])
    Ks = 9.4400e-03
    E = Haverkamp(Ks=np.log(Ks), A=1.1750e+06, gamma=4.74, alpha=1.6110e+06, theta_s=0.287, theta_r=0.075, beta=3.96)

    prob = RichardsProblem(M,E)
    prob.timeStep = 1
    prob.boundaryConditions = np.array([-61.5,-20.7])
    prob.doNewton = True
    prob.method = 'mixed'

    h = np.zeros(M.nC) + prob.boundaryConditions[0]

    numIts = 10
    Hs = range(numIts+1)
    Hs[0] = h
    for ii in range(numIts):
        hn = Hs[ii]
        hn1 = NewtonRoot().root(lambda hn1: prob.getResidual(hn,hn1), hn)
        Hs[ii+1] = hn1
        M.plotImage(hn1,showIt=True)

