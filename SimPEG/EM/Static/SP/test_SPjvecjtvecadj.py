from __future__ import print_function
import unittest
import numpy as np
from SimPEG import (Mesh, Maps, DataMisfit, Regularization, Inversion,
                    Optimization, InvProblem, Tests, Utils)
import simpegSP as SP
from pymatsolver import PardisoSolver

np.random.seed(40)

class SPProblemTestsCC_CurrentSource(unittest.TestCase):

    def setUp(self):

        mesh = Mesh.TensorMesh([20, 20, 20], "CCN")
        sigma = np.ones(mesh.nC)*1./100.
        actind = mesh.gridCC[:, 2] < -0.2
        # actMap = Maps.InjectActiveCells(mesh, actind, 0.)

        xyzM = Utils.ndgrid(np.ones_like(mesh.vectorCCx[:-1])*-0.4, np.ones_like(mesh.vectorCCy)*-0.4, np.r_[-0.3])
        xyzN = Utils.ndgrid(mesh.vectorCCx[1:], mesh.vectorCCy, np.r_[-0.3])

        problem = SP.Problem_CC(mesh, sigma=sigma, qMap=Maps.IdentityMap(mesh), Solver=PardisoSolver)
        rx = SP.Rx.Dipole(xyzN, xyzM)
        src = SP.Src.StreamingCurrents([rx], L=np.ones(mesh.nC), mesh=mesh,
                                       modelType="CurrentSource")
        survey = SP.Survey([src])
        survey.pair(problem)

        q = np.zeros(mesh.nC)
        inda = Utils.closestPoints(mesh, np.r_[-0.5, 0., -0.8])
        indb = Utils.closestPoints(mesh, np.r_[0.5, 0., -0.8])
        q[inda] = 1.
        q[indb] = -1.

        mSynth = q.copy()
        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Simple(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e-2)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3,
            dx=self.m0*0.1,
            eps = 1e-8
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        v = self.m0
        w = self.survey.dobs
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 2e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis.eval(m), self.dmis.evalDeriv(m)],
            self.m0,
            plotIt=False,
            num=3,
            dx=self.m0*2
        )
        self.assertTrue(passed)



if __name__ == '__main__':
    unittest.main()
