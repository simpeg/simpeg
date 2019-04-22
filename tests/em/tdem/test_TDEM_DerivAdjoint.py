from __future__ import division, print_function
import unittest
import numpy as np
import time
from SimPEG import Mesh, Maps, SolverLU, Tests
from SimPEG import EM

from pymatsolver import Pardiso as Solver

plotIt = False

testDeriv = True
testAdjoint = True

TOL = 1e-4

np.random.seed(10)


def get_mesh():
    cs = 10.
    ncx = 4
    ncy = 4
    ncz = 4
    npad = 2
    # hx = [(cs, ncx), (cs, npad, 1.3)]
    # hz = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    return Mesh.TensorMesh(
        [
            [(cs, npad, -1.5), (cs, ncx), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncy), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncz), (cs, npad, 1.5)]
        ], 'CCC'
    )


def get_mapping(mesh):
    active = mesh.vectorCCz < 0.
    activeMap = Maps.InjectActiveCells(
        mesh, active, np.log(1e-8), nC=mesh.nCz
    )
    return (
        Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap
    )


def get_prob(mesh, mapping, formulation):
    prb = getattr(EM.TDEM, 'Problem3D_{}'.format(formulation))(
        mesh, sigmaMap=mapping
    )
    prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
    prb.Solver = Solver
    return prb


def get_survey():
    src1 = EM.TDEM.Src.MagDipole([], loc=np.array([0., 0., 0.]))
    src2 = EM.TDEM.Src.MagDipole([], loc=np.array([0., 0., 8.]))
    return EM.TDEM.Survey([src1, src2])


# ====== TEST Jvec ========== #

class Base_DerivAdjoint_Test(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # create a prob where we will store the fields
        mesh = get_mesh()
        mapping = get_mapping(mesh)
        self.prob = get_prob(mesh, mapping, self.formulation)
        self.survey = get_survey()
        self.m = (
            np.log(1e-1)*np.ones(self.prob.sigmaMap.nP) +
            1e-3*np.random.randn(self.prob.sigmaMap.nP)
        )
        self.prob.pair(self.survey)
        print('Solving Fields for problem {}'.format(self.formulation))
        t = time.time()
        self.fields = self.prob.fields(self.m)
        print('... done. Time: {}\n'.format(time.time()-t))

        # create a prob where will be re-computing fields at each jvec
        # iteration
        mesh = get_mesh()
        mapping = get_mapping(mesh)
        self.probfwd = get_prob(mesh, mapping, self.formulation)
        self.surveyfwd = get_survey()
        self.probfwd.pair(self.surveyfwd)

    def get_rx(self, rxcomp):
        rxOffset = 15.
        rxlocs = np.array([[rxOffset, 0., -1e-2]])
        rxtimes = np.logspace(-4, -3, 20)
        return getattr(EM.TDEM.Rx, 'Point_{}'.format(rxcomp[:-1]))(
            locs=rxlocs, times=rxtimes, orientation=rxcomp[-1]
        )

    def set_rxList(self, rxcomp):
        # append the right rxlist to the surveys
        rx = [self.get_rx(rxcomp)]
        rxfwd = [self.get_rx(rxcomp)]

        # append to srclists
        for srcList, rxlist in zip(
            [self.survey.srcList, self.surveyfwd.srcList], [rx, rxfwd]
        ):
            for src in srcList:
                src.rxList = rxlist

    def JvecTest(self, rxcomp):
        self.set_rxList(rxcomp)

        def derChk(m):
            return [
                self.probfwd.survey.dpred(m),
                lambda mx: self.prob.Jvec(self.m, mx, f=self.fields)
            ]
        print('test_Jvec_{prbtype}_{rxcomp}'.format(
            prbtype=self.formulation, rxcomp=rxcomp)
        )
        Tests.checkDerivative(derChk, self.m, plotIt=False, num=2, eps=1e-20)

    def JvecVsJtvecTest(self, rxcomp):
        self.set_rxList(rxcomp)
        print(
            '\nAdjoint Testing Jvec, Jtvec prob {}, {}'.format(
                self.formulation, rxcomp
            )
        )

        m = np.random.rand(self.prob.sigmaMap.nP)
        d = np.random.randn(self.prob.survey.nD)
        V1 = d.dot(self.prob.Jvec(self.m, m, f=self.fields))
        V2 = m.dot(self.prob.Jtvec(self.m, d, f=self.fields))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
        passed = np.abs(V1-V2) < tol

        print('    {v1} {v2} {passed}'.format(
            prbtype=self.formulation, v1=V1, v2=V2, passed=passed))
        self.assertTrue(passed)


class TDEM_Fields_B_Pieces(Base_DerivAdjoint_Test):

    formulation = 'b'

    def test_eDeriv_m_adjoint(self):
        tInd = 0

        prb = self.prob
        f = self.fields
        v = np.random.rand(prb.mesh.nF)

        print('\n Testing eDeriv_m Adjoint')

        m = np.random.rand(len(self.m))
        e = np.random.randn(prb.mesh.nE)
        V1 = e.dot(f._eDeriv_m(1, prb.survey.srcList[0], m))
        V2 = m.dot(f._eDeriv_m(1, prb.survey.srcList[0], e, adjoint=True))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
        passed = np.abs(V1-V2) < tol

        print('    ', V1, V2, np.abs(V1-V2), tol, passed)
        self.assertTrue(passed)

    def test_eDeriv_u_adjoint(self):
        print('\n Testing eDeriv_u Adjoint')

        prb = self.prob
        f = self.fields

        b = np.random.rand(prb.mesh.nF)
        e = np.random.randn(prb.mesh.nE)
        V1 = e.dot(f._eDeriv_u(1, prb.survey.srcList[0], b))
        V2 = b.dot(f._eDeriv_u(1, prb.survey.srcList[0], e, adjoint=True))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
        passed = np.abs(V1-V2) < tol

        print('    ', V1, V2, np.abs(V1-V2), tol, passed)
        self.assertTrue(passed)


class DerivAdjoint_E(Base_DerivAdjoint_Test):

    formulation = 'e'

    if testDeriv:
        def test_Jvec_e_dbxdt(self):
            self.JvecTest('dbdtx')

        def test_Jvec_e_dbzdt(self):
            self.JvecTest('dbdtz')

        def test_Jvec_e_ey(self):
            self.JvecTest('ey')

        def test_Jvec_e_dhxdt(self):
            self.JvecTest('dhdtx')

        def test_Jvec_e_dhzdt(self):
            self.JvecTest('dhdtz')

        def test_Jvec_e_jy(self):
            self.JvecTest('jy')

    if testAdjoint:
        def test_Jvec_adjoint_e_dbdtx(self):
            self.JvecVsJtvecTest('dbdtx')

        def test_Jvec_adjoint_e_dbdtz(self):
            self.JvecVsJtvecTest('dbdtz')

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest('ey')

        def test_Jvec_adjoint_e_dhdtx(self):
            self.JvecVsJtvecTest('dhdtx')

        def test_Jvec_adjoint_e_dhdtz(self):
            self.JvecVsJtvecTest('dhdtz')

        def test_Jvec_adjoint_e_jy(self):
            self.JvecVsJtvecTest('jy')


class DerivAdjoint_B(Base_DerivAdjoint_Test):

    formulation = 'b'

    if testDeriv:
        def test_Jvec_b_bx(self):
            self.JvecTest('bx')

        def test_Jvec_b_bz(self):
            self.JvecTest('bz')

        def test_Jvec_b_dbdtx(self):
            self.JvecTest('dbdtx')

        def test_Jvec_b_dbdtz(self):
            self.JvecTest('dbdtz')

        def test_Jvec_b_jy(self):
            self.JvecTest('jy')

        def test_Jvec_b_hx(self):
            self.JvecTest('hx')

        def test_Jvec_b_hz(self):
            self.JvecTest('hz')

        def test_Jvec_b_dhdtx(self):
            self.JvecTest('dhdtx')

        def test_Jvec_b_dhdtz(self):
            self.JvecTest('dhdtz')

        def test_Jvec_b_jy(self):
            self.JvecTest('jy')

    if testAdjoint:
        def test_Jvec_adjoint_b_bx(self):
            self.JvecVsJtvecTest('bx')

        def test_Jvec_adjoint_b_bz(self):
            self.JvecVsJtvecTest('bz')

        def test_Jvec_adjoint_b_dbdtx(self):
            self.JvecVsJtvecTest('dbdtx')

        def test_Jvec_adjoint_b_dbdtz(self):
            self.JvecVsJtvecTest('dbdtz')

        def test_Jvec_adjoint_b_ey(self):
            self.JvecVsJtvecTest('ey')

        def test_Jvec_adjoint_b_hx(self):
            self.JvecVsJtvecTest('hx')

        def test_Jvec_adjoint_b_hz(self):
            self.JvecVsJtvecTest('hz')

        def test_Jvec_adjoint_b_dhdtx(self):
            self.JvecVsJtvecTest('dhdtx')

        def test_Jvec_adjoint_b_dhdtx(self):
            self.JvecVsJtvecTest('dhdtz')

        def test_Jvec_adjoint_b_ey(self):
            self.JvecVsJtvecTest('jy')


class DerivAdjoint_H(Base_DerivAdjoint_Test):

    formulation = 'h'

    if testDeriv:
        def test_Jvec_h_hx(self):
            self.JvecTest('hx')

        def test_Jvec_h_hz(self):
            self.JvecTest('hz')

        def test_Jvec_h_dhdtx(self):
            self.JvecTest('dhdtx')

        def test_Jvec_h_dhdtz(self):
            self.JvecTest('dhdtz')

        def test_Jvec_h_jy(self):
            self.JvecTest('jy')

        def test_Jvec_h_bx(self):
            self.JvecTest('bx')

        def test_Jvec_h_bz(self):
            self.JvecTest('bz')

        def test_Jvec_h_dbdtx(self):
            self.JvecTest('dbdtx')

        def test_Jvec_h_dbdtz(self):
            self.JvecTest('dbdtz')

        def test_Jvec_h_ey(self):
            self.JvecTest('ey')

    if testAdjoint:
        def test_Jvec_adjoint_h_hx(self):
            self.JvecVsJtvecTest('hx')

        def test_Jvec_adjoint_h_hz(self):
            self.JvecVsJtvecTest('hz')

        def test_Jvec_adjoint_h_dhdtx(self):
            self.JvecVsJtvecTest('dhdtx')

        def test_Jvec_adjoint_h_dhdtz(self):
            self.JvecVsJtvecTest('dhdtz')

        def test_Jvec_adjoint_h_jy(self):
            self.JvecVsJtvecTest('jy')

        def test_Jvec_adjoint_h_bx(self):
            self.JvecVsJtvecTest('bx')

        def test_Jvec_adjoint_h_bz(self):
            self.JvecVsJtvecTest('bz')

        def test_Jvec_adjoint_h_dbdtx(self):
            self.JvecVsJtvecTest('dbdtx')

        def test_Jvec_adjoint_h_dbdtz(self):
            self.JvecVsJtvecTest('dbdtz')

        def test_Jvec_adjoint_h_ey(self):
            self.JvecVsJtvecTest('ey')


class DerivAdjoint_J(Base_DerivAdjoint_Test):

    formulation = 'j'

    if testDeriv:
        def test_Jvec_j_jy(self):
            self.JvecTest('jy')

        def test_Jvec_j_dhdtx(self):
            self.JvecTest('dhdtx')

        def test_Jvec_j_dhdtz(self):
            self.JvecTest('dhdtz')

        def test_Jvec_j_ey(self):
            self.JvecTest('ey')

        def test_Jvec_j_dbdtx(self):
            self.JvecTest('dbdtx')

        def test_Jvec_j_dbdtz(self):
            self.JvecTest('dbdtz')

    if testAdjoint:
        def test_Jvec_adjoint_j_jy(self):
            self.JvecVsJtvecTest('jy')

        def test_Jvec_adjoint_j_dhdtx(self):
            self.JvecVsJtvecTest('dhdtx')

        def test_Jvec_adjoint_j_dhdtz(self):
            self.JvecVsJtvecTest('dhdtz')

        def test_Jvec_adjoint_j_ey(self):
            self.JvecVsJtvecTest('ey')

        def test_Jvec_adjoint_j_dbdtx(self):
            self.JvecVsJtvecTest('dbdtx')

        def test_Jvec_adjoint_j_dbdtz(self):
            self.JvecVsJtvecTest('dbdtz')

if __name__ == '__main__':
    unittest.main()
