from SimPEG import Mesh, Problem, Utils, np, sp, Tests
import BaseMag
from scipy.constants import mu_0
from MagAnalytics import spheremodel, CongruousMagBC



class MagneticsDiffSecondary(Problem.BaseProblem):
    """
        Secondary field approach using differential equations!
    """

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
    def MfMui(self): return self._MfMui

    @property
    def MfMu0(self): return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.model.transform(m)
        self._MfMui = self.mesh.getFaceMass(1./mu)
        # self._MfMui = self.mesh.getFaceInnerProduct(1./mu)
        #TODO: this will break if tensor mu
        self._MfMuI = Utils.sdiag(1./self._MfMui.diagonal())
        self._MfMu0 = self.mesh.getFaceMass(1/mu_0)
        # self._MfMu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    def getB0(self):
        b0 = self.data.B0
        B0 = np.r_[b0[0]*np.ones(self.mesh.nFx),
                   b0[1]*np.ones(self.mesh.nFy),
                   b0[2]*np.ones(self.mesh.nFz)]
        return B0

    def getRHS(self, m):

        B0 = self.getB0()
        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)

        chi = self.model.transform(m, asMu=False)
        Bbc, const = CongruousMagBC(self.mesh, self.data.B0, chi)
        self.Bbc_const = const
        self.Bbc = Bbc
        #TODO: put congrous BC back in
        # return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 #+ Mc*Dface*self._Pout.T*Bbc
        return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 + Mc*Dface*self._Pout.T*Bbc

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

            .. math ::

                \mathbf{A}\mathbf{u} = \mathbf{rhs}

                \mathbf{A} = - \Div(\MfMui)^{-1}\Div^{T}
                
                \mathbf{rhs} = - \Div(\MfMui)^{-1}\mathbf{M}^f_{\\frac{1}{\mu_0}}\mathbf{B}_0 + \Div\mathbf{B}_0-\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}


        """
        return self._Div*self.MfMuI*self._Div.T


    def fields(self, m):
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/A.diagonal()))
        u, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)

        B0 = self.getB0()

        B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u

        #TODO: Create a mag fields object class.
        # F = self.getInitialFields()
        # e.g. {'B': B, 'u': u}

        return {'B': B, 'u': u}

        # return self.forward(m, self.getRHS, self.calcFields, F=F)

    @Utils.timeIt
    def Jvec(self, m, v, u=None):
        """
            Computing Jacobian multiplied by vector
            
            By setting our problem as 

            .. math ::

                \mathbf{C}(\mathbf{m}, \mathbf{u}) = \mathbf{A}\mathbf{u} - \mathbf{rhs} = 0

            And taking derivative w.r.t m

            .. math ::

                \\nabla \mathbf{C}(\mathbf{m}, \mathbf{u}) = \\nabla_m \mathbf{C}(\mathbf{m}) \delta \mathbf{m} +
                                                             \\nabla_u \mathbf{C}(\mathbf{u}) \delta \mathbf{u} = 0

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} = - [\\nabla_u \mathbf{C}(\mathbf{u})]^{-1}\\nabla_m \mathbf{C}(\mathbf{m})

            With some linear algebra we can have 

            .. math ::

                \\nabla_u \mathbf{C}(\mathbf{u}) = \mathbf{A}

                \\nabla_m \mathbf{C}(\mathbf{m}) = 
                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}}                                

            .. math :: 

                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} = 
                \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[\Div \diag (\Div^T \mathbf{u}) \dMfMuI \\right]

                \dMfMuI = \diag(\MfMui)^{-1}_{vec} \mathbf{Av}_{F2CC}^T\diag(\mathbf{v})\diag(\\frac{1}{\mu^2})

                \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} =  \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[ 
                \Div \diag(\M^f_{\mu_{0}^{-1} \mathbf{B}_0}) \dMfMuI \\right] - \diag(\mathbf{v})\mathbf{D} \mathbf{P}_{out}^T\\frac{\partial B_{sBC}}{\partial \mathbf{m}}


        """
        if u is None:
            u = self.fields(m)

        #TODO: B, u = u['B'], u['u']

        B, u = u['B'], u['u']

        mu = self.model.transform(m, asMu=True)
        dmudm = self.model.transform(m, asMu=True)

        P = self.data.projectFieldsDeriv(u)

        A = self.getA(m)
        dCdu = A
        # (Av_m)^-1
        # -(Av_m)^-2 * MfMu_dm * d/dm(1/mu(m))
        # (Av_m)^-2 * MfMu_dm * diag(mu(m)^-2) * mT_dm

        #TODO: only works for diagonal MfMui
        # Some chain rule!


        # harm_dm = Utils.sdiag(self.MfMui.diagonal()**(-2))
        # MfMu_dm = self.mesh.getFaceMassDeriv()
        # dmuI_dm = Utils.sdiag(mu**(-2))
        # mT_dm   = self.model.transformDeriv(m, asMu=True)

        getFIPconst = 1./3
        MfMuIvec = 1/self.MfMui.diagonal()*getFIPconst
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(self.mesh.vol*1./mu**2)

        Div = self._Div
        # lots-o-bracket for vector multiplication first!
        # MfMu_dmXv =  harm_dm * ( MfMu_dm * ( dmuI_dm * ( mT_dm * v ) ) )
        #dCdm_A = D * ( Utils.sdiag( D.T * u ) * MfMu_dmXv )

        dCdm_A = dmudm*Div * ( Utils.sdiag( Div.T * u * dMfMuI ) )

        # rhs = D * MfMuI * MfMu0 * B0

        B0 = self.getB0()

        #TODO: add congrous stuff
        dCdm_RHS = dmudm* Div * Utils.sdiag( self.MfMu0*B0  ) * dMfMuI - Utils.sdiag(self.mesh.vol)*self.mesh.faceDiv*self.Pout.T*self.Bbc*self.Bbc_const


        # c(m,u) = A(m)u - rhs(m)
        dCdm = dCdm_A - dCdm_RHS

        solve = Solver(dCdu)

        #TODO: Multiply by the dP(u(m))/du
        # We transformed u in our fields object.
        #         ( dBdu *                   + dBdm(u)  )
        Jv = - P *           solve.solve(dCdm)
        return Utils.mkvc(Jv)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    hxind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
    hyind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
    hzind = ((5,25,1.3),(40, 12.5),(5,25,1.3))
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
    # mesh.plotSlice(B, 'F', view='vec', showIt=True)

    dpred = data.dpred(chi, u=B)


    # ##################
    # # Test J
    # ##################

    # d_chi = 0.8*chi #np.random.rand(mesh.nCz)
    # d_sph_ind = spheremodel(mesh, 0., 0., -100., 50)
    # d_chi[d_sph_ind] = 0.02

    # from SimPEG.Tests import checkDerivative

    # derChk = lambda m: [prob.data.dpred(m), lambda mx: -prob.Jvec(chi, mx)]
    # print '\n'
    # passed = checkDerivative(derChk, chi, plotIt=False, dx=d_chi, num=2)


    # # plt.pcolor(X, Y, dpred.reshape(X.shape, order='F'))

    # # plt.show()








