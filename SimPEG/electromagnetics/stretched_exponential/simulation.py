from SimPEG import simulation, utils, maps, props
import discretize as Mesh
from SimPEG import survey
import numpy as np
import scipy.sparse as sp


# TODO: deprecate this later
class SEInvSimulation(simulation.BaseSimulation):

    sigmaInf, sigmaInfMap, sigmaInfDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )

    eta, etaMap, etaDeriv = props.Invertible(
        "Cole-Cole chargeability (V/V)"
    )

    tau, tauMap, tauDeriv = props.Invertible(
        "Cole-Cole time constant (s)"
    )

    c, cMap, cDeriv = props.Invertible(
        "Cole-Cole frequency dependency"
    )

    P = None
    J = None
    time = None

    def __init__(self, mesh, **kwargs):
        simulation.BaseSimulation.__init__(self, mesh, **kwargs)

    def fields(self, m=None, f=None):
        if m is not None:
            self.model = m
        self.J = self.get_peta_deriv(self.time)
        return self.get_peta(self.time)

    def Jvec(self, m, v, f=None):
        jvec = self.J.dot(v)
        return jvec

    def Jtvec(self, m, v, f=None):
        jtvec = self.J.T.dot(v)
        return jtvec

    def get_peta(self, time):
        return self.eta * np.exp(-(time / self.tau)**self.c)

    def get_peta_deriv(self, time):
        kerneleta = lambda t, eta, tau, c: np.exp(-(time/tau)**c)
        kerneltau = lambda t, eta, tau, c: (c * eta/tau)*((t/tau)**c) * np.exp(-(t/tau)**c)
        kernelc = lambda t, eta, tau, c: -eta * ((t/tau)**c) * np.exp(-(t/tau)**c) * np.log(t/tau)

        tempeta = kerneleta(time, self.eta, self.tau, self.c).reshape([-1,1])
        temptau = kerneltau(time, self.eta, self.tau, self.c).reshape([-1,1])
        tempc = kernelc(time, self.eta, self.tau, self.c).reshape([-1,1])
        J = tempeta * self.etaDeriv + temptau * self.tauDeriv + tempc * self.cDeriv
        return J

    def dpred(self, m, f=None):
        return self.fields(m)

    def residual(self, m, dobs, f=None):
        if dobs.size == 1:
            return utils.mkvc(np.r_[self.dpred(m, f=f) - dobs])
        else:
            return utils.mkvc(self.dpred(m, f=f) - dobs)
