from SimPEG import Mesh, Maps, Utils, Tests
from SimPEG.EM import FDEM
import numpy as np
from scipy.constants import mu_0

import unittest

MuMax = 200.
TOL = 1e-14
EPS = 1e-20


class MuTests(unittest.TestCase):

    def setUp(self):
        cs = 10.
        nc = 20.
        npad = 15.
        hx = [(cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]

        self.freq = 1
        self.mesh = Mesh.CylMesh([hx, 1., hz], '0CC')
        self.m0 = MuMax*np.random.rand(self.mesh.nC)

    def getProblem(self, prbtype='e'):
        rxcomp = ['real', 'imag']

        loc = Utils.ndgrid(
            [self.mesh.vectorCCx, np.r_[0.], self.mesh.vectorCCz]
        )

        rxList_edge = [
            getattr(FDEM.Rx, 'Point_{f}'.format(f=f))(
                loc, component=comp, orientation=orient
            )
            for f in ['e', 'h']
            for comp in rxcomp
            for orient in ['y']
        ]

        rxList_face = [
            getattr(FDEM.Rx, 'Point_{f}'.format(f=f))(
                loc, component=comp, orientation=orient
            )
            for f in ['b', 'j']
            for comp in rxcomp
            for orient in ['x', 'z']
        ]

        rxList = rxList_edge + rxList_face

        src_loc = np.r_[0., 0., 0.]

        if prbtype in ['e', 'b']:
            src = FDEM.Src.MagDipole(
                rxList=rxList, loc=src_loc, freq=self.freq
            )

        elif prbtype in ['h', 'j']:
            ind = Utils.closestPoints(self.mesh, src_loc, 'Fz') + self.mesh.vnF[0]
            vec = np.zeros(self.mesh.nF)
            vec[ind] = 1.

            src = FDEM.Src.RawVec_e(rxList=rxList, freq=self.freq, s_e=vec)

        self.prob = getattr(FDEM, 'Problem3D_{}'.format(prbtype))(
            self.mesh, sigma=self.m0, muMap=Maps.ChiMap(self.mesh)
        )
        self.survey = FDEM.Survey([src])
        self.prob.pair(self.survey)

    def AderivTest(self, prbtype='e'):
        self.getProblem(prbtype)
        if prbtype in ['b', 'j']:
            nu = self.mesh.nF
        elif prbtype in ['e', 'h']:
            nu = self.mesh.nE
        v = np.random.rand(nu)

        def AderivFun(m):
            self.prob.model = m
            A = self.prob.getA(self.freq)
            Av = A*v
            self.prob.model = self. m0

            def ADeriv_dm(dm):
                return self.prob.getADeriv(self.freq, v, dm)

            return Av, ADeriv_dm

        print('\n Testing ADeriv Mu {}'.format(prbtype))
        return Tests.checkDerivative(
            AderivFun, self.m0, plotIt=False, num=3, eps=EPS
        )

    def AadjointTest(self, prbtype='e'):
        self.getProblem(prbtype)
        print('\n Testing A_adjoint {} Mu'.format(prbtype))
        m = np.random.rand(self.prob.muMap.nP)

        if prbtype == 'e':
            u = np.random.rand(self.mesh.nE)
            v = np.random.rand(self.mesh.nE)
        else:
            u = np.random.rand(self.mesh.nF)
            v = np.random.rand(self.mesh.nF)

        self.prob.model = self.m0

        V1 = v.dot(self.prob.getADeriv(self.freq, u, m))
        V2 = m.dot(self.prob.getADeriv(self.freq, u, v, adjoint=True))
        diff = np.abs(V1-V2)
        tol = TOL * (np.abs(V1) + np.abs(V2))/2.
        passed = diff < tol
        print('AdjointTest {prbtype} {v1} {v2} {diff} {tol} {passed}'.format(
            prbtype=prbtype, v1=V1, v2=V2, diff=diff, tol=tol, passed=passed))
        return self.assertTrue(passed)

    def JvecTest(self, prbtype='e'):
        self.getProblem(prbtype)
        print('Testing Jvec {}'.format(prbtype))

        def fun(x):
            return (
                self.prob.survey.dpred(x), lambda x: self.prob.Jvec(self.m0, x)
            )
        return Tests.checkDerivative(
            fun, self.m0, num=2, plotIt=False, eps=EPS
        )

    def JtvecTest(self, prbtype='e'):
        self.getProblem(prbtype)
        print('Testing Jvec {}'.format(prbtype))

        m = np.random.rand(self.prob.muMap.nP)
        v = np.random.rand(self.survey.nD)

        self.prob.model = self.m0

        V1 = v.dot(self.prob.Jvec(self.m0, m))
        V2 = m.dot(self.prob.Jtvec(self.m0, v))
        diff = np.abs(V1-V2)
        tol = TOL * (np.abs(V1) + np.abs(V2))/2.
        passed = diff < tol
        print(
            'AdjointTest {prbtype} {v1} {v2} {diff} {tol} {passed}'.format(
                prbtype=prbtype, v1=V1, v2=V2, diff=diff, tol=tol,
                passed=passed
            )
        )
        return passed

    # def test_Aderiv_e(self):
    #     self.assertTrue(self.AderivTest('e'))

    # def test_Aderiv_b(self):
    #     self.assertTrue(self.AderivTest('b'))

    def test_Jvec_e(self):
        self.assertTrue(self.JvecTest('e'))

    def test_Jvec_b(self):
        self.assertTrue(self.JvecTest('b'))

    def test_Jvec_j(self):
        self.assertTrue(self.JvecTest('j'))

    def test_Jtvec_e(self):
        self.assertTrue(self.JtvecTest('e'))

    def test_Jtvec_b(self):
        self.assertTrue(self.JtvecTest('b'))

    def test_Jtvec_j(self):
        self.assertTrue(self.JvecTest('j'))


# class MuSigmaTests(unittest.TestCase):

#     def setUp(self):
#         cs = 10.
#         nc = 20.
#         npad = 15.
#         hx = [(cs, nc), (cs, npad, 1.3)]
#         hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]

#         self.freq = 1

#         self.mesh = Mesh.CylMesh([hx, 1., hz], '0CC')

#         mumod = MuMax*np.random.rand(self.mesh.nC)
#         sigmamod = np.random.randn(self.mesh.nC)
#         self.m0 = np.hstack([mumod, sigmamod])

#         wires = Maps.Wires(
#             ('mu', self.mesh.nC),
#             ('sigma', self.mesh.nC)
#         )

#         muMap = Maps.ChiMap(self.mesh) * wires.mu
#         sigmaMap = Maps.ExpMap(self.mesh) * wires.sigma

#         rxcomp = ['real', 'imag']

#         loc = Utils.ndgrid(
#             [self.mesh.vectorCCx, np.r_[0.], self.mesh.vectorCCz]
#         )

#         rxList_edge = [
#             getattr(FDEM.Rx, 'Point_{f}'.format(f=f))(
#                 loc, component=comp, orientation=orient
#             )
#             for f in ['e', 'j']
#             for comp in rxcomp
#             for orient in ['y']
#         ]

#         rxList_face = [
#             getattr(FDEM.Rx, 'Point_{f}'.format(f=f))(
#                 loc, component=comp, orientation=orient
#             )
#             for f in ['b', 'h']
#             for comp in rxcomp
#             for orient in ['x', 'z']
#         ]

#         rxList = rxList_edge + rxList_face

#         src = FDEM.Src.MagDipole(
#             rxList=rxList, loc=np.r_[0., 0., 0.], freq=self.freq
#         )

#         self.prob = FDEM.Problem3D_e(
#             self.mesh, muMap=muMap, sigmaMap=sigmaMap
#         )

#         self.survey = FDEM.Survey([src])
#         self.prob.pair(self.survey)

#     def test_Aderiv_musig(self, prbtype='e'):
#         if prbtype == 'b':
#             nu = mesh.nF
#         elif prbtype == 'e':
#             nu = self.mesh.nE
#         v = np.random.rand(nu)

#         def AderivFun(m):
#             self.prob.model = m
#             A = self.prob.getA(self.freq)
#             Av = A*v
#             self.prob.model = self.m0

#             def ADeriv_dm(dm):
#                 return self.prob.getADeriv(self.freq, v, dm)

#             return Av, ADeriv_dm

#         print('\n Testing ADeriv Mu Sigma {}'.format(prbtype))
#         Tests.checkDerivative(
#             AderivFun, self.m0, plotIt=False,
#             num=3, eps=1e-20
#             )

#     def test_Aadjoint_musig(self, prbtype='e'):

#         print('\n Testing A_adjoint Mu Sigma')
#         m = np.random.rand(self.m0.size)
#         u = np.random.rand(self.mesh.nE)
#         v = np.random.rand(self.mesh.nE)

#         self.prob.model = self.m0

#         V1 = v.dot(self.prob.getADeriv(self.freq, u, m))
#         V2 = m.dot(self.prob.getADeriv(self.freq, u, v, adjoint=True))
#         passed = np.abs(V1-V2) < TOL * (np.abs(V1) + np.abs(V2))/2.
#         print('AdjointTest {prbtype} {v1} {v2} {passed}'.format(
#             prbtype=prbtype, v1=V1, v2=V2, passed=passed))
#         self.assertTrue(passed)

#     def test_Jvec_e_musig(self):
#         print('Testing Jvec e mu sigma')

#         def fun(x):
#             return (
#                 self.prob.survey.dpred(x), lambda x: self.prob.Jvec(self.m0, x)
#             )
#         return Tests.checkDerivative(fun, self.m0, num=2, plotIt=False, eps=EPS)

#     def test_Jvec_e_adjoint_musig(self):
#         print('Testing Jvec e mu sigma')
#         prbtype = 'e'

#         print('\n Testing A_adjoint Mu')
#         m = np.random.rand(self.prob.muMap.nP)
#         v = np.random.rand(self.survey.nD)

#         self.prob.model = self.m0

#         V1 = v.dot(self.prob.Jvec(self.m0, m))
#         V2 = m.dot(self.prob.Jtvec(self.m0, v))
#         diff = np.abs(V1-V2)
#         tol = TOL * (np.abs(V1) + np.abs(V2))/2.
#         passed = diff < tol
#         print('AdjointTest {prbtype} {v1} {v2} {diff} {tol} {passed}'.format(
#             prbtype=prbtype, v1=V1, v2=V2, diff=diff, tol=tol, passed=passed))
#         self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()


