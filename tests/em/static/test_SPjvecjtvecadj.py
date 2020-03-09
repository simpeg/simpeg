# from __future__ import print_function
# import unittest
# import numpy as np
#
# import discretize
#
# from SimPEG import (
#     maps, data_misfit, regularization, inversion,
#     optimization, inverse_problem, tests, utils
# )
# from SimPEG.electromagnetics import spontaneous_potential as sp
# from pymatsolver import PardisoSolver
#
# np.random.seed(40)
#
#
# class SPProblemTestsCC_CurrentSource(unittest.TestCase):
#
#     def setUp(self):
#
#         mesh = discretize.TensorMesh([20, 20, 20], "CCN")
#         sigma = np.ones(mesh.nC)*1./100.
#         actind = mesh.gridCC[:, 2] < -0.2
#         # actMap = maps.InjectActiveCells(mesh, actind, 0.)
#
#         xyzM = utils.ndgrid(np.ones_like(mesh.vectorCCx[:-1])*-0.4, np.ones_like(mesh.vectorCCy)*-0.4, np.r_[-0.3])
#         xyzN = utils.ndgrid(mesh.vectorCCx[1:], mesh.vectorCCy, np.r_[-0.3])
#
#         rx = sp.receivers.Dipole(xyzN, xyzM)
#         src = sp.sources.StreamingCurrents([rx], L=np.ones(mesh.nC), mesh=mesh,
#                                        modelType="CurrentSource")
#         survey = sp.survey.Survey([src])
#
#         simulation = sp.simulation.Problem_CC(
#                 mesh=mesh, survey=survey, sigma=sigma, qMap=maps.IdentityMap(mesh), Solver=PardisoSolver
#                 )
#
#         q = np.zeros(mesh.nC)
#         inda = utils.closestPoints(mesh, np.r_[-0.5, 0., -0.8])
#         indb = utils.closestPoints(mesh, np.r_[0.5, 0., -0.8])
#         q[inda] = 1.
#         q[indb] = -1.
#
#         mSynth = q.copy()
#         dpred = simulation.make_synthetic_data(mSynth, add_noise=True)
#
#         # Now set up the problem to do some minimization
#         dmis = data_misfit.L2DataMisfit(data=dpred, simulation=simulation)
#         reg = regularization.Simple(mesh)
#         opt = optimization.InexactGaussNewton(
#             maxIterLS=20, maxIter=10, tolF=1e-6,
#             tolX=1e-6, tolG=1e-6, maxIterCG=6
#         )
#         invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e-2)
#         inv = inversion.BaseInversion(invProb)
#
#         self.inv = inv
#         self.reg = reg
#         self.p = simulation
#         self.mesh = mesh
#         self.m0 = mSynth
#         self.survey = survey
#         self.dmis = dmis
#
#     def test_misfit(self):
#         passed = tests.checkDerivative(
#             lambda m: [
#                 self.p.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
#             ],
#             self.m0,
#             plotIt=False,
#             num=3,
#             dx=self.m0*0.1,
#             eps = 1e-8
#         )
#         self.assertTrue(passed)
#
#     def test_adjoint(self):
#         v = self.m0
#         w = self.survey.dobs
#         wtJv = w.dot(self.p.Jvec(self.m0, v))
#         vtJtw = v.dot(self.p.Jtvec(self.m0, w))
#         passed = np.abs(wtJv - vtJtw) < 2e-8
#         print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
#         self.assertTrue(passed)
#
#     def test_dataObj(self):
#         passed = tests.checkDerivative(
#             lambda m: [self.dmis(m), self.dmis.deriv(m)],
#             self.m0,
#             plotIt=False,
#             num=3,
#             dx=self.m0*2
#         )
#         self.assertTrue(passed)
#
# if __name__ == '__main__':
#     unittest.main()
