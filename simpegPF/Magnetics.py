from SimPEG import *
import BaseMag
from scipy.constants import mu_0
from MagAnalytics import spheremodel, CongruousMagBC



class MagneticsDiffSecondary(Problem.BaseProblem):
    """
        Secondary field approach using differential equations!
    """

    surveyPair = BaseMag.BaseMagSurvey
    modelPair = BaseMag.BaseMagModel

    def __init__(self, model, **kwargs):
        Problem.BaseProblem.__init__(self, model, **kwargs)

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

    @Utils.requires('survey')
    def getB0(self):
        b0 = self.survey.B0
        B0 = np.r_[b0[0]*np.ones(self.mesh.nFx),
                   b0[1]*np.ones(self.mesh.nFy),
                   b0[2]*np.ones(self.mesh.nFz)]
        return B0

    def getRHS(self, m):
        """

        .. math ::

            \mathbf{rhs} =  \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0 - \Div\mathbf{B}_0+\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()
        Dface = self.mesh.faceDiv
        Mc = Utils.sdiag(self.mesh.vol)

        mu = self.model.transform(m)
        chi = mu/mu_0-1

        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.B0, chi)
        self.Bbc = Bbc
        self.Bbc_const = Bbc_const
        return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 + Mc*Dface*self._Pout.T*Bbc

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div*self.MfMuI*self._Div.T


    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell center [nC x 1]
            B: defined on the cell center [nF x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)
        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/A.diagonal()))
        u, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)
        B0 = self.getB0()
        B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u

        return {'B': B, 'u': u}

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
                \Div \diag(\M^f_{\mu_{0}^{-1}}\mathbf{B}_0) \dMfMuI \\right] - \diag(\mathbf{v})\mathbf{D} \mathbf{P}_{out}^T\\frac{\partial B_{sBC}}{\partial \mathbf{m}}

            In the end,

            .. math ::

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} =
                - [ \mathbf{A} ]^{-1}\left[ \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u}
                - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} \\right]

            A little tricky point here is we are not interested in potential (u), but interested in magnetic flux (B).
            Thus, we need sensitivity for B. Now we take derivative of B w.r.t m and have

            .. math ::

                \\frac{\delta \mathbf{B}} {\delta \mathbf{m}} = \\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
                \left[
                \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
                 -  \diag (\Div^T\mathbf{u})\dMfMuI
                \\right ]

                 -  (\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}}

            Finally we evaluate the above, but we should remember that

            .. note ::

                We only want to evalute

                .. math ::

                    \mathbf{J}\mathbf{v} = \\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}}\mathbf{v}

                Since forming sensitivity matrix is very expensive in that this monster is "big" and "dense" matrix!!


        """
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.model.transform(m)
        dmudm = self.model.transformDeriv(m)
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(vol*1./mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
        dCdm_RHS1 = Div * (Utils.sdiag( self.MfMu0*B0  ) * dMfMuI)
        temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_v = dCdm_A*v - dCdm_RHSv

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/dCdu.diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu, dCdm_v, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print "Iterative solver did not work well (Jvec)"
            # raise Exception ("Iterative solver did not work well")

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu

        dudm = -sol
        dBdmv =     (  Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm*v)) \
                     - Utils.sdiag(Div.T*u)*(dMfMuI* (dmudm*v)) \
                     - self.MfMuI*(Div.T* (dudm)) )

        return Utils.mkvc(P*dBdmv)

    @Utils.timeIt
    def Jtvec(self, m, v, u=None):
        """
            Computing Jacobian^T multiplied by vector.

        .. math ::

            (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} = \left[ \mathbf{P}_{deriv}\\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
            \left[
            \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
             -  \diag (\Div^T\mathbf{u})\dMfMuI
            \\right ]\\right]^{T}

             -  \left[\mathbf{P}_{deriv}(\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}} \\right]^{T}

        where

        .. math ::

            \mathbf{P}_{derv} = \\frac{\partial \mathbf{P}}{\partial\mathbf{B}}

        .. note ::

            Here we only want to compute

            .. math ::

                \mathbf{J}^{T}\mathbf{v} = (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} \mathbf{v}

        """
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.model.transform(m)
        dmudm = self.model.transformDeriv(m)
        dchidmu = Utils.sdiag(1/mu_0*np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1/self.MfMui.diagonal()
        dMfMuI = Utils.sdiag(MfMuIvec**2)*self.mesh.aveF2CC.T*Utils.sdiag(vol*1./mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        s = Div * ( self.MfMuI.T * ( P.T*v ) )

        m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/(dCdu.T).diagonal()))
        sol, info = sp.linalg.bicgstab(dCdu.T, s, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print "Iterative solver did not work well (Jtvec)"
            # raise Exception ("Iterative solver did not work well")


        # dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
        # dCdm_Atsol = ( dMfMuI.T*( Utils.sdiag( Div.T * u ) * (Div.T * dmudm)) ) * sol
        dCdm_Atsol = ( dmudm.T * dMfMuI.T*( Utils.sdiag( Div.T * u ) * Div.T ) ) * sol

        # dCdm_RHS1 = Div * (Utils.sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( Utils.sdiag( self.MfMu0*B0  ) ) * Div.T * dmudm) * sol
        dCdm_RHS1tsol = ( dmudm.T * dMfMuI.T*( Utils.sdiag( self.MfMu0*B0  ) ) * Div.T ) * sol


        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        temp1sol = ( Dface.T*( Utils.sdiag(vol)*sol ) )
        temp2 = self.Bbc_const*(self._Pout.T*self.Bbc).T
        # dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        dCdm_RHS2tsol  = (dmudm.T*dchidmu.T*vol)*np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = Utils.sdiag(self.MfMu0*B0)*(dMfMuI * (dmudm))
        Btemp = Utils.sdiag(Div.T*u)*(dMfMuI* (dmudm))
        Jtv = Atemp.T*(P.T*v) - Btemp.T*(P.T*v) - Ctv

        return Utils.mkvc(Jtv)

def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):

    """
        Inversion module for MagneticsDiffSecondary

    """
    from SimPEG import Optimization, Regularization, Parameters, ObjFunction, Inversion
    prob = MagneticsDiffSecondary(mesh, model)

    miter = kwargs.get('maxIter', 10)

    if prob.ispaired:
        prob.unpair()
    if data.ispaired:
        data.unpair()
    prob.pair(data)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP),flag='D')
    # Create a regularization program
    reg = Regularization.Tikhonov(model)
    # Create an objective function
    beta = Parameters.BetaSchedule(beta0=1e0)
    obj = ObjFunction.BaseObjFunction(data, reg, beta=beta)
    # Create an inversion object
    inv = Inversion.BaseInversion(obj, opt)

    return inv, reg



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
    Inc = 90.
    Dec = 0.
    Btot = 51000

    data.setBackgroundField(Inc, Dec, Btot)

    xr = np.linspace(-300, 300, 41)
    yr = np.linspace(-300, 300, 41)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones((xr.size, yr.size))*150
    rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
    data.rxLoc = rxLoc

    prob = MagneticsDiffSecondary(mesh, model)
    prob.pair(data)

    dpred = data.dpred(chi)

    # fig = plt.figure( figsize = (8,5) )
    # ax = plt.subplot(111)
    # dat = plt.imshow(np.reshape(dpred, (xr.size, yr.size), order='F'), extent=[min(xr), max(xr), min(yr), max(yr)])
    # plt.colorbar(dat, ax = ax)
    # plt.show()










