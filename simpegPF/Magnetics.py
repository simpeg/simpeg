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
        self._Div = Mc*Dface*self.Pin.T*self.Pin

    @property
    def Pbc(self): return self._Pbc

    @property
    def Pin(self): return self._Pin

    @property
    def Pout(self): return self._Pout

    @property
    def Div(self): return self._Div

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

    def getRHS(self):
        b0 = self.data.B0
        B0 = np.r_[b0[0]*np.ones(M3.nFx),
                   b0[1]*np.ones(M3.nFy),
                   b0[2]*np.ones(M3.nFz)]

        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)

        Bbc = CongruousMagBC(M3, np.array([Box, Boy, Boz]), chi)

        rhs = -self.Div*self.MfmuI*self.Mfmu0*B0 + self.Div*B0 - Mc*Dface*self.Pout.T*Bbc

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math::

            \mathbf{A} = \mathbf{D}\mu\mathbf{G}u



        """
        return -self.Div*self.MfmuI*self.Div.T


    def fields(self, m):
        self.makeMassMatrices(m)
        # F = self.getInitialFields()
        # return self.forward(m, self.getRHS, self.calcFields, F=F)


