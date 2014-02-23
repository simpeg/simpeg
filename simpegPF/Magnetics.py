from SimPEG import Problem, Utils
import BaseMag
from scipy.constants import mu_0



class MagneticsDiffSecondary(Problem.BaseProblem):
    """Secondary field approach using differential equations!"""

    dataPair = BaseMag.BaseMagData
    modelPair = BaseMag.BaseMagModel

    def __init__(self, mesh, model, **kwargs):
        Problem.BaseProblem.__init__(mesh, model, **kwargs)

        self._Pbc, self._Pin, self._Pout = /
            self.mesh.getBCProjWF('neumann', discretization='CC')

        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)
        self._Div = Mc*Dface*self._Pin.T*self._Pin

    @property
    def MfMuI(self): return self._MfMuI

    @property
    def Mfmu0(self): return self._Mfmu0

    def makeMassMatrices(self, m):
        mu = self.model.transform(m)
        Mfmui = self.mesh.getFaceInnerProduct(1./mu)
        #TODO: this will break if tensor mu
        self._MfmuI = Utils.sdiag(1./Mfmui.diagonal())
        self._Mfmu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    def getRHS(self, m):
        b0 = self.data.B0
        B0 = np.r_[b0[0]*np.ones(self.mesh.nFx),
                   b0[1]*np.ones(self.mesh.nFy),
                   b0[2]*np.ones(self.mesh.nFz)]

        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)

        chi = self.model.transform(m, asMu=False)
        Bbc = CongruousMagBC(self.mesh, self.data.B0, chi)

        return -self._Div*self.MfmuI*self.Mfmu0*B0 + self._Div*B0 - Mc*Dface*self._Pout.T*Bbc

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math::

            \mathbf{A} = \mathbf{D}\mu\mathbf{G}u



        """
        return -self._Div*self.MfmuI*self._Div.T


    def fields(self, m):
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)

        # F = self.getInitialFields()
        # return self.forward(m, self.getRHS, self.calcFields, F=F)


