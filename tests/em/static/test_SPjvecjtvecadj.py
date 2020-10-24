from __future__ import print_function
import unittest
import numpy as np

import discretize

from SimPEG import (
    maps, data_misfit, regularization, inversion,
    optimization, inverse_problem, tests, utils
)
from SimPEG.electromagnetics.static import spontaneous_potential as sp
from pymatsolver import PardisoSolver

np.random.seed(40)


class SpontaneousPotential_CellCenters_CurrentSource(unittest.TestCase):

    def setUp(self):

        mesh = discretize.TensorMesh([20, 20, 20], "CCN")
        sigma = np.ones(mesh.nC)*1./100.
        actind = mesh.gridCC[:, 2] < -0.2
        # actMap = maps.InjectActiveCells(mesh, actind, 0.)

        xyzM = utils.ndgrid(np.ones_like(mesh.vectorCCx[:-1])*-0.4, np.ones_like(mesh.vectorCCy)*-0.4, np.r_[-0.3])
        xyzN = utils.ndgrid(mesh.vectorCCx[1:], mesh.vectorCCy, np.r_[-0.3])

        rx = sp.receivers.Dipole(xyzN, xyzM)
        src = sp.sources.SpontaneousPotentialSource([rx])
        survey = sp.survey.Survey([src])

        simulation = sp.simulation.SimulationCurrentSourceCellCenters(
            mesh=mesh, survey=survey, sigma=sigma, qsMap=maps.IdentityMap(mesh), Solver=PardisoSolver
           )

        qs = np.zeros(mesh.nC)
        inda = utils.closestPoints(mesh, np.r_[-0.5, 0., -0.8])
        indb = utils.closestPoints(mesh, np.r_[0.5, 0., -0.8])
        qs[inda] = 1.
        qs[indb] = -1.

        mSynth = qs.copy()
        dobs = simulation.make_synthetic_data(mSynth, add_noise=True)

        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=simulation)
        reg = regularization.Simple(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e-2)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.sim = simulation
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis
        self.dobs = dobs

    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: [
                self.sim.dpred(m), lambda mx: self.sim.Jvec(self.m0, mx)
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
        w = self.dobs.dobs
        wtJv = w.dot(self.sim.Jvec(self.m0, v))
        vtJtw = v.dot(self.sim.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 2e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3,
            dx=self.m0*2
        )
        self.assertTrue(passed)


class SpontaneousPotential_CellCenters_CurrentDensity(unittest.TestCase):

    def setUp(self):

        mesh = discretize.TensorMesh([20, 20, 20], "CCN")
        sigma = np.ones(mesh.nC)*1./100.
        actind = mesh.gridCC[:, 2] < -0.2
        # actMap = maps.InjectActiveCells(mesh, actind, 0.)

        xyzM = utils.ndgrid(np.ones_like(mesh.vectorCCx[:-1])*-0.4, np.ones_like(mesh.vectorCCy)*-0.4, np.r_[-0.3])
        xyzN = utils.ndgrid(mesh.vectorCCx[1:], mesh.vectorCCy, np.r_[-0.3])

        rx = sp.receivers.Dipole(xyzN, xyzM)
        src = sp.sources.SpontaneousPotentialSource([rx])
        survey = sp.survey.Survey([src])

        simulation = sp.simulation.SimulationCurrentDensityCellCenters(
            mesh=mesh, survey=survey, sigma=sigma, jsMap=maps.IdentityMap(nP=3*mesh.nC), Solver=PardisoSolver
           )
        
        x1, y1, z1 = 0.5, 0, 0
        x2, y2, z2 = -0.5, 0, 0
        xc, yc, zc = mesh.grid_cell_centers[:,0], mesh.grid_cell_centers[:,1], mesh.grid_cell_centers[:,2]
        R1 = np.sqrt((xc-x1)**2 + (yc-y1)**2 + (zc-z1)**2)
        R2 = np.sqrt((xc-x2)**2 + (yc-y2)**2 + (zc-z2)**2)
        
        jsx1 = (1/(4*np.pi)) * (xc-x1)/R1**3
        jsy1 = (1/(4*np.pi)) * (yc-y1)/R1**3
        jsz1 = (1/(4*np.pi)) * (zc-z1)/R1**3
        
        jsx2 = -(1/(4*np.pi)) * (xc-x2)/R2**3
        jsy2 = -(1/(4*np.pi)) * (yc-y2)/R2**3
        jsz2 = -(1/(4*np.pi)) * (zc-z2)/R2**3
        
        js = np.r_[jsx1, jsy1, jsz1] + np.r_[jsx2, jsy2, jsz2]

        mSynth = js.copy()
        dobs = simulation.make_synthetic_data(mSynth, add_noise=True)

        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=simulation)
        reg = regularization.Simple(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e-2)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.sim = simulation
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis
        self.dobs = dobs

    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: [
                self.sim.dpred(m), lambda mx: self.sim.Jvec(self.m0, mx)
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
        w = self.dobs.dobs
        wtJv = w.dot(self.sim.Jvec(self.m0, v))
        vtJtw = v.dot(self.sim.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 2e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3,
            dx=self.m0*2
        )
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
