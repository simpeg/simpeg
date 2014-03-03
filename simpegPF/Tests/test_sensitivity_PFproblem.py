import unittest
from SimPEG import *
from simpegPF import BaseMag
import matplotlib.pyplot as plt
import simpegPF as PF
from scipy.constants import mu_0


class MagSensProblemTests(unittest.TestCase):

    def setUp(self):

        hxind = ((5,25,1.3),(21, 25.),(5,25,1.3))
        hyind = ((5,25,1.3),(21, 25.),(5,25,1.3))
        hzind = ((5,25,1.3),(20, 25.),(5,25,1.3))
        hx, hy, hz = Utils.meshTensors(hxind, hyind, hzind)
        M = Mesh.TensorMesh([hx, hy, hz], [-hx.sum()/2,-hy.sum()/2,-hz.sum()/2])
        chibkg = 0.001
        chiblk = 0.01
        chi = np.ones(M.nC)*chibkg

        Inc = 90.
        Dec = 0.
        Btot = 51000

        b0 = PF.MagAnalytics.IDTtoxyz(Inc, Dec, Btot)
        sph_ind = PF.MagAnalytics.spheremodel(M, 0., 0., 0., 100)
        chi[sph_ind] = chiblk
        model = PF.BaseMag.BaseMagModel(M)

        data = BaseMag.BaseMagData()

        data.setBackgroundField(Inc, Dec, Btot)
        xr = np.linspace(-300, 300, 41)
        yr = np.linspace(-300, 300, 41)
        X, Y = np.meshgrid(xr, yr)
        Z = np.ones((xr.size, yr.size))*150
        rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
        data.rxLoc = rxLoc



        prob = PF.Magnetics.MagneticsDiffSecondary(M, model)
        prob.pair(data)
        dpre = data.dpred(chi)

        fields = prob.fields(chi)
        self.u = fields['u']
        self.B = fields['B']

        self.data = data
        self.model = model
        self.prob = prob
        self.M = M
        self.chi = chi



    def test_mass(self):
        print '\n >>Derivative for MfMuI works.'
        mu = self.model.transform(self.chi)
        def MfmuI(mu):

            chi = mu/mu_0-1
            self.prob.makeMassMatrices(chi)
            vol = self.prob.mesh.vol
            aveF2CC = self.prob.mesh.aveF2CC
            MfMuI = self.prob.MfMuI.diagonal()

            return MfMuI

        def dMfmuI(mu, v):

            chi = mu/mu_0-1
            self.prob.makeMassMatrices(chi)
            vol = self.prob.mesh.vol
            aveF2CC = self.prob.mesh.aveF2CC
            MfMuI = self.prob.MfMuI.diagonal()
            dMfMuI = Utils.sdiag(MfMuI**2)*aveF2CC.T*Utils.sdiag(vol*1./mu**2)

            return dMfMuI*v

        d_mu = mu*0.8
        derChk = lambda m: [MfmuI(m), lambda mx: dMfmuI(self.chi, mx)]
        passed = Tests.checkDerivative(derChk, mu, num=4, dx = d_mu, plotIt=False)


        self.assertTrue(passed)


    def test_dCdm_Av(self):
        print '\n >>Derivative for Cm_A.'
        Div = self.prob._Div
        vol = self.prob.mesh.vol
        aveF2CC = self.prob.mesh.aveF2CC

        def Cm_A(chi):
            dmudm = self.model.transformDeriv(chi)
            u = self.u
            # chi = mu/mu_0-1
            self.prob.makeMassMatrices(chi)
            mu = self.model.transform(self.chi)
            A = self.prob.getA(self.chi)
            MfMuIvec = 1/self.prob.MfMui.diagonal()
            dMfMuI = Utils.sdiag(MfMuIvec**2)*aveF2CC.T*Utils.sdiag(vol*1./mu**2)

            Cm_A =  A*u

            return Cm_A

        def dCdm_A(chi, v):

            dmudm = self.model.transformDeriv(chi)
            u = self.u
            self.prob.makeMassMatrices(chi)
            mu = self.model.transform(self.chi)
            A = self.prob.getA(self.chi)
            MfMuIvec = 1/self.prob.MfMui.diagonal()
            dMfMuI = Utils.sdiag(MfMuIvec**2)*aveF2CC.T*Utils.sdiag(vol*1./mu**2)

            Cm_A =  A*u
            dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )

            return dCdm_A*v

        d_chi = self.chi*0.8
        derChk = lambda m: [Cm_A(m), lambda mx: dCdm_A(self.chi, mx)]
        passed = Tests.checkDerivative(derChk, self.chi, num=4, dx = d_chi, plotIt=False)
        self.assertTrue(passed)


    def test_dCdmu_RHS(self):
        print '\n >>Derivative for Cm_RHS.'
        u = self.u
        Div = self.prob._Div
        mu = self.model.transform(self.chi)
        vol = self.prob.mesh.vol
        Mc = Utils.sdiag(vol)
        aveF2CC = self.prob.mesh.aveF2CC
        B0 = self.prob.getB0()
        Dface = self.prob.mesh.faceDiv

        def Cm_RHS(chi):

            self.prob.makeMassMatrices(chi)
            dmudm = self.model.transformDeriv(chi)
            dchidmu = Utils.sdiag(1/(dmudm.diagonal()))
            Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.data.B0, chi)
            MfMuIvec = 1/self.prob.MfMui.diagonal()
            dMfMuI = Utils.sdiag(MfMuIvec**2)*aveF2CC.T*Utils.sdiag(vol*1./mu**2)
            RHS1 = Div*self.prob.MfMuI*self.prob.MfMu0*B0
            RHS2 =  Mc*Dface*self.prob._Pout.T*Bbc
            RHS = RHS1 + RHS2 + Div*B0

            return RHS


        def dCdm_RHS(chi, v):


            self.prob.makeMassMatrices(chi)
            dmudm = self.model.transformDeriv(chi)
            dmdmu = Utils.sdiag(1/(dmudm.diagonal()))
            Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.data.B0, chi)
            MfMuIvec = 1/self.prob.MfMui.diagonal()
            dMfMuI = Utils.sdiag(MfMuIvec**2)*aveF2CC.T*Utils.sdiag(vol*1./mu**2)
            dCdm_RHS1 = Div * (Utils.sdiag( self.prob.MfMu0*B0  ) * dMfMuI)
            temp1 = (Dface*(self.prob._Pout.T*Bbc_const*Bbc))
            dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, v)
            dCdm_RHSv =  dCdm_RHS1*(dmudm*v) + dCdm_RHS2v

            return dCdm_RHSv

        d_chi = self.chi*0.8
        derChk = lambda m: [Cm_RHS(m), lambda mx: dCdm_RHS(self.chi, mx)]
        passed = Tests.checkDerivative(derChk, self.chi, num=4, dx = d_chi, plotIt=False)
        self.assertTrue(passed)


    def test_dudm(self):
        print ">> Derivative test for dudm"
        u = self.u
        Div = self.prob._Div
        mu = self.model.transform(self.chi)
        vol = self.prob.mesh.vol
        Mc = Utils.sdiag(vol)
        aveF2CC = self.prob.mesh.aveF2CC
        B0 = self.prob.getB0()
        Dface = self.prob.mesh.faceDiv

        def ufun(chi):
            u = self.prob.fields(chi)['u']
            return u

        def dudm(chi, v):

            chi = mu/mu_0-1
            self.prob.makeMassMatrices(chi)
            u = self.u
            dmudm = self.model.transformDeriv(chi)
            dmdmu = Utils.sdiag(1/(dmudm.diagonal()))
            Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.data.B0, chi)
            MfMuIvec = 1/self.prob.MfMui.diagonal()
            dMfMuI = Utils.sdiag(MfMuIvec**2)*aveF2CC.T*Utils.sdiag(vol*1./mu**2)
            dCdu = self.prob.getA(chi)
            dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
            dCdm_RHS1 = Div * (Utils.sdiag( self.prob.MfMu0*B0  ) * dMfMuI)
            temp1 = (Dface*(self.prob._Pout.T*Bbc_const*Bbc))
            dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, v)
            dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
            dCdm_v = dCdm_A*v - dCdm_RHSv
            m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/dCdu.diagonal()))
            sol, info = sp.linalg.bicgstab(dCdu, dCdm_v, tol=1e-8, maxiter=1000, M=m1)

            dudm = -sol

            return dudm

        d_chi = 10.0*self.chi #np.random.rand(mesh.nCz)
        d_sph_ind = PF.MagAnalytics.spheremodel(self.prob.mesh, 0., 0., -50., 50)
        d_chi[d_sph_ind] = 0.1

        derChk = lambda m: [ufun(m), lambda mx: dudm(self.chi, mx)]
        # TODO: I am not sure why the order get worse as step decreases .. --;
        passed = Tests.checkDerivative(derChk, self.chi, num=2, dx = d_chi, plotIt=False)
        self.assertTrue(passed)


    def test_dBdm(self):
        print ">> Derivative test for dBdm"
        u = self.u
        Div = self.prob._Div
        mu = self.model.transform(self.chi)
        vol = self.prob.mesh.vol
        Mc = Utils.sdiag(vol)
        aveF2CC = self.prob.mesh.aveF2CC
        B0 = self.prob.getB0()
        Dface = self.prob.mesh.faceDiv

        def Bfun(chi):
            B = self.prob.fields(chi)['B']
            return B

        def dBdm(chi, v):

            chi = mu/mu_0-1
            self.prob.makeMassMatrices(chi)
            u = self.u
            dmudm = self.model.transformDeriv(chi)
            dmdmu = Utils.sdiag(1/(dmudm.diagonal()))
            Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.data.B0, chi)
            MfMuIvec = 1/self.prob.MfMui.diagonal()
            dMfMuI = Utils.sdiag(MfMuIvec**2)*aveF2CC.T*Utils.sdiag(vol*1./mu**2)
            dCdu = self.prob.getA(chi)
            dCdm_A = Div * ( Utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
            dCdm_RHS1 = Div * (Utils.sdiag( self.prob.MfMu0*B0  ) * dMfMuI)
            temp1 = (Dface*(self.prob._Pout.T*Bbc_const*Bbc))
            dCdm_RHS2v  = (Utils.sdiag(vol)*temp1)*np.inner(vol, v)
            dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
            dCdm_v = dCdm_A*v - dCdm_RHSv
            m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/dCdu.diagonal()))
            sol, info = sp.linalg.bicgstab(dCdu, dCdm_v, tol=1e-8, maxiter=1000, M=m1)

            dudm = -sol
            dBdmv =       (  Utils.sdiag(self.prob.MfMu0*B0)*(dMfMuI * (dmudm*v)) \
                         - Utils.sdiag(Div.T*u)*(dMfMuI* (dmudm*v)) \
                         - self.prob.MfMuI*(Div.T* (dudm)) )

            return dBdmv

        d_chi = 10.0*self.chi #np.random.rand(mesh.nCz)
        d_sph_ind = PF.MagAnalytics.spheremodel(self.prob.mesh, 0., 0., -50., 50)
        d_chi[d_sph_ind] = 0.1

        derChk = lambda m: [Bfun(m), lambda mx: dBdm(self.chi, mx)]
        # TODO: I am not sure why the order get worse as step decreases .. --;
        passed = Tests.checkDerivative(derChk, self.chi, num=2, dx = d_chi, plotIt=False)
        self.assertTrue(passed)



    def test_Jvec(self):
        print ">> Derivative test for Jvec"
        mu = self.model.transform(self.chi)

        d_chi = 10.0*self.chi #np.random.rand(mesh.nCz)
        d_sph_ind = PF.MagAnalytics.spheremodel(self.prob.mesh, 0., 0., -50., 50)
        d_chi[d_sph_ind] = 0.1

        a = self.prob.Jvec(self.chi, d_chi)

        derChk = lambda m: [self.data.dpred(m), lambda mx: self.prob.Jvec(self.chi, mx)]
        # TODO: I am not sure why the order get worse as step decreases .. --;
        passed = Tests.checkDerivative(derChk, self.chi, num=2, dx = d_chi, plotIt=False)
        self.assertTrue(passed)

    def test_Jtvec(self):
        print ">> Derivative test for Jtvec"
        mu = self.model.transform(self.chi)
        dobs = self.data.dpred(self.chi)

        def misfit (m, dobs):
            dpre = self.data.dpred(m)
            misfit = 0.5*np.linalg.norm(dpre-dobs)**2
            residual = dpre-dobs
            dmisfit = self.prob.Jtvec(self.chi, residual)

            return misfit, dmisfit

        # TODO: I am not sure why the order get worse as step decreases .. --;
        derChk = lambda m: misfit(m, dobs)
        passed = Tests.checkDerivative(derChk, self.chi, num=4, plotIt=False)
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
