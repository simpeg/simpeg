#
# import unittest
# import numpy as np
# #from simpegPF import BaseMag
# #import matplotlib.pyplot as plt
# import discretize
# #import simpeg.PF as PF
# from simpeg import maps, utils
# from simpeg.potential_fields import magnetics as mag
# from simpeg.utils.model_builder import get_indices_sphere
# from scipy.constants import mu_0
#
#
# class MagSensProblemTests(unittest.TestCase):
#
#     def setUp(self):
#         cs = 25.
#         hxind = [(cs, 5, -1.3), (cs, 21), (cs, 5, 1.3)]
#         hyind = [(cs, 5, -1.3), (cs, 21), (cs, 5, 1.3)]
#         hzind = [(cs, 5, -1.3), (cs, 20), (cs, 5, 1.3)]
#         M = discretize.TensorMesh([hxind, hyind, hzind], 'CCC')
#         chibkg = 0.001
#         chiblk = 0.01
#         susceptibility = np.ones(M.nC)*chibkg
#
#         Inc = 90.
#         Dec = 0.
#         Btot = 51000
#         H0 = (Btot, Inc, Dec)
#
#         b0 = mag.analytics.IDTtoxyz(-Inc, Dec, Btot)
#         sph_ind = get_indices_sphere([0., 0., 0.], 100, M.gridCC)
#         susceptibility[sph_ind] = chiblk
#
#         xr = np.linspace(-300, 300, 41)
#         yr = np.linspace(-300, 300, 41)
#         X, Y = np.meshgrid(xr, yr)
#         Z = np.ones((xr.size, yr.size))*150
#         rxLoc = np.c_[utils.mkvc(X), utils.mkvc(Y), utils.mkvc(Z)]
#
#         components = ['bx', 'by', 'bz']
#         receivers = mag.Point(rxLoc, components=components)
#         srcField = mag.UniformBackgroundField([receivers], parameters=H0)
#
#         self.survey = mag.Survey(srcField)
#
#         self.sim = mag.simulation.Simulation3DDifferential(
#             M,
#             survey=self.survey,
#             muMap=maps.ChiMap(M),
#         )
#         dpre = self.sim.dpred(susceptibility)
#
#         fields = self.sim.fields(susceptibility)
#         self.u = fields['u']
#         self.B = fields['B']
#
#         self.M = M
#         self.susceptibility = susceptibility
#
#     def test_mass(self):
#         print('\n >>Derivative for MfMuI works.')
#         #mu = self.model*self.susceptibility
#
#         def MfmuI(susceptibility):
#             self.sim.makeMassMatrices(susceptibility)
#             return self.sim.MfMuI
#
#         def dMfmuI(susceptibility, v):
#             self.sim.makeMassMatrices(susceptibility)
#             vol = self.M.cell_volumes
#             aveF2CC = self.M.aveF2CC
#             MfMuI = self.sim.MfMuI.diagonal()
#             dMfMuI = utils.sdiag(MfMuI**2)*aveF2CC.T*utils.sdiag(vol*1./mu**2)
#
#             return dMfMuI*v
#
#         d_mu = mu*0.8
#         derChk = lambda m: [MfmuI(m), lambda mx: dMfmuI(self.susceptibility, mx)]
#         passed = Tests.check_derivative(derChk, mu, num=4, dx = d_mu, plotIt=False, random_seed=0)
#
#         self.assertTrue(passed)
#
#
#     def test_dCdm_Av(self):
#         print('\n >>Derivative for Cm_A.')
#         Div = self.prob._Div
#         vol = self.prob.mesh.cell_volumes
#         aveF2CC = self.prob.mesh.aveF2CC
#
#         def Cm_A(susceptibility):
#             dmudm = self.model.deriv(susceptibility)
#             u = self.u
#             # susceptibility = mu/mu_0-1
#             self.prob.makeMassMatrices(susceptibility)
#             mu = self.model*(self.susceptibility)
#             A = self.prob.getA(self.susceptibility)
#             MfMuIvec = 1/self.prob._Mf__perm_inv.diagonal()
#             dMfMuI = utils.sdiag(MfMuIvec**2)*aveF2CC.T*utils.sdiag(vol*1./mu**2)
#
#             Cm_A =  A*u
#
#             return Cm_A
#
#         def dCdm_A(susceptibility, v):
#
#             dmudm = self.model.deriv(susceptibility)
#             u = self.u
#             self.prob.makeMassMatrices(susceptibility)
#             mu = self.model*self.susceptibility
#             A = self.prob.getA(self.susceptibility)
#             MfMuIvec = 1/self.prob._Mf__perm_inv.diagonal()
#             dMfMuI = utils.sdiag(MfMuIvec**2)*aveF2CC.T*utils.sdiag(vol*1./mu**2)
#
#             Cm_A =  A*u
#             dCdm_A = Div * (utils.sdiag( Div.T * u ) * dMfMuI * dmudm)
#
#             return dCdm_A*v
#
#         d_susceptibility = self.susceptibility*0.8
#         derChk = lambda m: [Cm_A(m), lambda mx: dCdm_A(self.susceptibility, mx)]
#         passed = Tests.check_derivative(derChk, self.susceptibility, num=4, dx = d_chi, plotIt=False, random_seed=0)
#         self.assertTrue(passed)
#
#
#     def test_dCdmu_RHS(self):
#         print('\n >>Derivative for Cm_RHS.')
#         u = self.u
#         Div = self.prob._Div
#         mu = self.model*self.susceptibility
#         vol = self.prob.mesh.cell_volumes
#         Mc = utils.sdiag(vol)
#         aveF2CC = self.prob.mesh.aveF2CC
#         B0 = self.prob.getB0()
#         Dface = self.prob.mesh.face_divergence
#
#         def Cm_RHS(susceptibility):
#
#             self.prob.makeMassMatrices(susceptibility)
#             dmudm = self.model.deriv(susceptibility)
#             dchidmu = utils.sdiag(1/(dmudm.diagonal()))
#             Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.survey.B0, susceptibility)
#             MfMuIvec = 1/self.prob._Mf__perm_inv.diagonal()
#             dMfMuI = utils.sdiag(MfMuIvec**2)*aveF2CC.T*utils.sdiag(vol*1./mu**2)
#             RHS1 = Div*self.prob.MfMuI*self.prob.MfMu0*B0
#             RHS2 =  Mc*Dface*self.prob._Pout.T*Bbc
#             RHS = RHS1 + RHS2 + Div*B0
#
#             return RHS
#
#
#         def dCdm_RHS(susceptibility, v):
#
#
#             self.prob.makeMassMatrices(susceptibility)
#             dmudm = self.model.deriv(susceptibility)
#             dmdmu = utils.sdiag(1/(dmudm.diagonal()))
#             Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.survey.B0, susceptibility)
#             MfMuIvec = 1/self.prob._Mf__perm_inv.diagonal()
#             dMfMuI = utils.sdiag(MfMuIvec**2)*aveF2CC.T*utils.sdiag(vol*1./mu**2)
#             dCdm_RHS1 = Div * (utils.sdiag( self.prob.MfMu0*B0  ) * dMfMuI)
#             temp1 = (Dface*(self.prob._Pout.T*Bbc_const*Bbc))
#             dCdm_RHS2v  = (utils.sdiag(vol)*temp1)*np.inner(vol, v)
#             dCdm_RHSv =  dCdm_RHS1*(dmudm*v) + dCdm_RHS2v
#
#             return dCdm_RHSv
#
#         d_susceptibility = self.susceptibility*0.8
#         derChk = lambda m: [Cm_RHS(m), lambda mx: dCdm_RHS(self.susceptibility, mx)]
#         passed = Tests.check_derivative(derChk, self.susceptibility, num=4, dx = d_chi, plotIt=False, random_seed=0)
#         self.assertTrue(passed)
#
#
#     # def test_dudm(self):
#     #     print(">> Derivative test for dudm")
#     #     u = self.u
#     #     Div = self.prob._Div
#     #     mu = self.model*(self.susceptibility)
#     #     vol = self.prob.mesh.cell_volumes
#     #     Mc = utils.sdiag(vol)
#     #     aveF2CC = self.prob.mesh.aveF2CC
#     #     B0 = self.prob.getB0()
#     #     Dface = self.prob.mesh.face_divergence
#
#     #     def ufun(susceptibility):
#     #         u = self.prob.fields(susceptibility)['u']
#     #         return u
#
#     #     def dudm(susceptibility, v):
#
#     #         susceptibility = mu/mu_0-1
#     #         self.prob.makeMassMatrices(susceptibility)
#     #         u = self.u
#     #         dmudm = self.model.deriv(susceptibility)
#     #         dmdmu = utils.sdiag(1/(dmudm.diagonal()))
#     #         Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.survey.B0, susceptibility)
#     #         MfMuIvec = 1/self.prob._Mf__perm_inv.diagonal()
#     #         dMfMuI = utils.sdiag(MfMuIvec**2)*aveF2CC.T*utils.sdiag(vol*1./mu**2)
#     #         dCdu = self.prob.getA(susceptibility)
#     #         dCdm_A = Div * ( utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
#     #         dCdm_RHS1 = Div * (utils.sdiag( self.prob.MfMu0*B0  ) * dMfMuI)
#     #         temp1 = (Dface*(self.prob._Pout.T*Bbc_const*Bbc))
#     #         dCdm_RHS2v  = (utils.sdiag(vol)*temp1)*np.inner(vol, v)
#     #         dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
#     #         dCdm_v = dCdm_A*v - dCdm_RHSv
#     #         m1 = sp.linalg.interface.aslinearoperator(utils.sdiag(1/dCdu.diagonal()))
#     #         sol, info = sp.linalg.bicgstab(dCdu, dCdm_v, tol=1e-8, maxiter=1000, M=m1)
#
#     #         dudm = -sol
#
#     #         return dudm
#
#     #     d_susceptibility = 10.0*self.susceptibility #np.random.rand(mesh.shape_cells[2])
#     #     d_sph_ind = PF.MagAnalytics.spheremodel(self.prob.mesh, 0., 0., -50., 50)
#     #     d_chi[d_sph_ind] = 0.1
#
#     #     derChk = lambda m: [ufun(m), lambda mx: dudm(self.susceptibility, mx)]
#     #     # TODO: I am not sure why the order get worse as step decreases .. --;
#     #     passed = Tests.check_derivative(derChk, self.susceptibility, num=2, dx = d_chi, plotIt=False, random_seed=0)
#     #     self.assertTrue(passed)
#
#
#     # def test_dBdm(self):
#     #     print(">> Derivative test for dBdm")
#     #     u = self.u
#     #     Div = self.prob._Div
#     #     mu = self.model*(self.susceptibility)
#     #     vol = self.prob.mesh.cell_volumes
#     #     Mc = utils.sdiag(vol)
#     #     aveF2CC = self.prob.mesh.aveF2CC
#     #     B0 = self.prob.getB0()
#     #     Dface = self.prob.mesh.face_divergence
#
#     #     def Bfun(susceptibility):
#     #         B = self.prob.fields(susceptibility)['B']
#     #         return B
#
#     #     def dBdm(susceptibility, v):
#
#     #         susceptibility = mu/mu_0-1
#     #         self.prob.makeMassMatrices(susceptibility)
#     #         u = self.u
#     #         dmudm = self.model.deriv(susceptibility)
#     #         dmdmu = utils.sdiag(1/(dmudm.diagonal()))
#     #         Bbc, Bbc_const = PF.MagAnalytics.CongruousMagBC(self.prob.mesh, self.survey.B0, susceptibility)
#     #         MfMuIvec = 1/self.prob._Mf__perm_inv.diagonal()
#     #         dMfMuI = utils.sdiag(MfMuIvec**2)*aveF2CC.T*utils.sdiag(vol*1./mu**2)
#     #         dCdu = self.prob.getA(susceptibility)
#     #         dCdm_A = Div * ( utils.sdiag( Div.T * u )* dMfMuI *dmudm  )
#     #         dCdm_RHS1 = Div * (utils.sdiag( self.prob.MfMu0*B0  ) * dMfMuI)
#     #         temp1 = (Dface*(self.prob._Pout.T*Bbc_const*Bbc))
#     #         dCdm_RHS2v  = (utils.sdiag(vol)*temp1)*np.inner(vol, v)
#     #         dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
#     #         dCdm_v = dCdm_A*v - dCdm_RHSv
#     #         m1 = sp.linalg.interface.aslinearoperator(utils.sdiag(1/dCdu.diagonal()))
#     #         sol, info = sp.linalg.bicgstab(dCdu, dCdm_v, tol=1e-8, maxiter=1000, M=m1)
#
#     #         dudm = -sol
#     #         dBdmv =       (  utils.sdiag(self.prob.MfMu0*B0)*(dMfMuI * (dmudm*v)) \
#     #                      - utils.sdiag(Div.T*u)*(dMfMuI* (dmudm*v)) \
#     #                      - self.prob.MfMuI*(Div.T* (dudm)) )
#
#     #         return dBdmv
#
#     #     d_susceptibility = 10.0*self.susceptibility #np.random.rand(mesh.shape_cells[2])
#     #     d_sph_ind = PF.MagAnalytics.spheremodel(self.prob.mesh, 0., 0., -50., 50)
#     #     d_chi[d_sph_ind] = 0.1
#
#     #     derChk = lambda m: [Bfun(m), lambda mx: dBdm(self.susceptibility, mx)]
#     #     # TODO: I am not sure why the order get worse as step decreases .. --;
#     #     passed = Tests.check_derivative(derChk, self.susceptibility, num=2, dx = d_chi, plotIt=False, random_seed=0)
#     #     self.assertTrue(passed)
#
#
#
#     def test_Jvec(self):
#         print(">> Derivative test for Jvec")
#
#         d_susceptibility = 10.0*self.susceptibility #np.random.rand(mesh.shape_cells[2])
#         d_sph_ind = PF.MagAnalytics.spheremodel(self.prob.mesh, 0., 0., -50., 50)
#         d_chi[d_sph_ind] = 0.1
#
#         derChk = lambda m: (self.survey.dpred(m), lambda v: self.prob.Jvec(m, v))
#         # TODO: I am not sure why the order get worse as step decreases .. --;
#         passed = Tests.check_derivative(derChk, self.susceptibility, num=2, dx = d_chi, plotIt=False, random_seed=0)
#         self.assertTrue(passed)
#
#     def test_Jtvec(self):
#         print(">> Derivative test for Jtvec")
#         dobs = self.survey.dpred(self.susceptibility)
#
#         def misfit(m):
#             dpre = self.survey.dpred(m)
#             misfit = 0.5*np.linalg.norm(dpre-dobs)**2
#             residual = dpre-dobs
#             dmisfit = self.prob.Jtvec(self.susceptibility, residual)
#
#             return misfit, dmisfit
#
#         # TODO: I am not sure why the order get worse as step decreases .. --;
#         passed = Tests.check_derivative(misfit, self.susceptibility, num=4, plotIt=False, random_seed=0)
#         self.assertTrue(passed)
#
# if __name__ == '__main__':
#     unittest.main()
