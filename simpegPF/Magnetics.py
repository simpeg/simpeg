from SimPEG import Mesh, Problem, Utils, np, sp
import BaseMag
from scipy.constants import mu_0
from MagAnalytics import spheremodel, CongruousMagBC



class MagneticsDiffSecondary(Problem.BaseProblem):
    """Secondary field approach using differential equations!"""

    dataPair = BaseMag.BaseMagData
    modelPair = BaseMag.BaseMagModel

    def __init__(self, mesh, model, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, model, **kwargs)

        Pbc, Pin, self._Pout = \
            self.mesh.getBCProjWF('neumann', discretization='CC')

        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)
        self._Div = Mc*Dface*Pin.T*Pin

    @property
    def MfMuI(self): return self._MfMuI

    @property
    def MfMu0(self): return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.model.transform(m)
        MfMui = self.mesh.getFaceInnerProduct(1./mu)
        #TODO: this will break if tensor mu
        self._MfMuI = Utils.sdiag(1./MfMui.diagonal())
        self._MfMu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    def getRHS(self, m):
        b0 = self.data.B0
        B0 = np.r_[b0[0]*np.ones(self.mesh.nFx),
                   b0[1]*np.ones(self.mesh.nFy),
                   b0[2]*np.ones(self.mesh.nFz)]

        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)

        chi = self.model.transform(m, asMu=False)
        Bbc = CongruousMagBC(self.mesh, self.data.B0, chi)

        return -self._Div*self.MfMuI*self.MfMu0*B0 + self._Div*B0 - Mc*Dface*self._Pout.T*Bbc

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math::

            \mathbf{A} = \mathbf{D}\mu\mathbf{G}u



        """
        return -self._Div*self.MfMuI*self._Div.T


    def fields(self, m):
        self.makeMassMatrices(m)
        #TODO: change to pos def A
        A = self.getA(m)
        rhs = self.getRHS(m)

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(-1/A.diagonal()))
        phi, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)

        #TODO: make onPair function call
        b0 = self.data.B0
        B0 = np.r_[b0[0]*np.ones(self.mesh.nFx),
                   b0[1]*np.ones(self.mesh.nFy),
                   b0[2]*np.ones(self.mesh.nFz)]

        B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*phi

        return B

        # F = self.getInitialFields()
        # return self.forward(m, self.getRHS, self.calcFields, F=F)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hxind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
    hyind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
    hzind = ((5,25,1.3),(40, 12.5),(1,25,1.3))
    hx, hy, hz = Utils.meshTensors(hxind, hyind, hzind)
    mesh = Mesh.TensorMesh([hx, hy, hz], [-hx.sum()/2,-hy.sum()/2,-hz.sum()/2])

    chibkg = 0.
    chiblk = 0.01
    chi = np.ones(mesh.nC)*chibkg
    sph_ind = spheremodel(mesh, 0., 0., 0., 100)
    chi[sph_ind] = chiblk
    model = BaseMag.BaseMagModel(mesh)
    # mu = (1.+chi)*mu_0

    data = BaseMag.BaseMagData()
    data.setBackgroundField(x=1., y=1., z=0.)
    xr = np.linspace(-300, 300, 41)
    yr = np.linspace(-300, 300, 41)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones((xr.size, yr.size))*150
    rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
    data.rxLoc = rxLoc

    prob = MagneticsDiffSecondary(mesh, model)

    prob.pair(data)

    B = prob.fields(chi)
    mesh.plotSlice(B, 'F', view='vec', showIt=True)

    dpred = data.dpred(chi, u=B)

    plt.pcolor(X, Y, dpred.reshape(X.shape, order='F'))

    plt.show()








