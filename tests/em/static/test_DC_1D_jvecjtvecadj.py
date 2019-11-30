from __future__ import print_function
import unittest
import numpy as np
import discretize
from SimPEG import (
    maps, data_misfit, regularization,
    inversion, optimization, inverse_problem, tests, utils
)
from SimPEG.utils import mkvc
from SimPEG.electromagnetics import resistivity as dc
from pymatsolver import Pardiso

np.random.seed(40)

TOL = 1e-5
FLR = 1e-20 # "zero", so if residual below this --> pass regardless of order

class DC1DSimulation(unittest.TestCase):

    def setUp(self):

        ntx = 31
        xtemp_txP = np.logspace(1, 3, ntx)
        xtemp_txN = -xtemp_txP
        ytemp_tx = np.zeros(ntx)
        xtemp_rxP = -5
        xtemp_rxN = 5
        ytemp_rx = 0.
        abhalf = abs(xtemp_txP-xtemp_txN)*0.5
        a = xtemp_rxN-xtemp_rxP
        b = ((xtemp_txN-xtemp_txP)-a)*0.5

        # We generate tx and rx lists:
        srclist = []
        for i in range(ntx):
            rx = dc.Rx.Dipole(np.r_[xtemp_rxP, ytemp_rx, -12.5], np.r_[xtemp_rxN, ytemp_rx, -12.5])
            locA = np.r_[xtemp_txP[i], ytemp_tx[i], -12.5]
            locB = np.r_[xtemp_txN[i], ytemp_tx[i], -12.5]
            src = dc.Src.Dipole([rx], locA, locB)
            srclist.append(src)
        survey = dc.Survey(srclist)

        rho = np.r_[10, 10, 10]
        dummy_hz = 100.
        hz = np.r_[10, 10, dummy_hz]
        mesh = discretize.TensorMesh([hz])

        problem = dc.DCSimulation_1D(
            mesh=mesh,
            rhoMap=maps.ExpMap(mesh),
            t=hz[:-1],
            survey=survey,
            data_type='apparent_resistivity'
        )
        problem.dpred(np.log(rho))

        mSynth = np.log(rho)
        dobs = problem.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(simulation=problem, data=dobs)
        reg = regularization.Tikhonov(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0.)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis
        self.dobs = dobs

    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: [
                self.p.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(mkvc(self.dobs).shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
