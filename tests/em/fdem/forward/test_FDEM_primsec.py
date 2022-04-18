# import matplotlib
# matplotlib.use('Agg')

import discretize
from scipy.constants import mu_0
from SimPEG import maps, tests, utils
from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics.utils import omega

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import Solver as SolverLU

import time
import os
import numpy as np
import unittest

# This could be reduced if we refine the meshes
TOL_FWD = 5e-1  # relative tolerance for prim-sec comparison

TOL_JT = 1e-10
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order

np.random.seed(2016)

# To test the primary secondary-source, we look at make sure doing primary
# secondary for a simple model gives comprable results to just solving a 3D
# problem

# Also run a sensitivity test, adjoint test

# physical properties
sigmaback = 1e-1
sigmablock = 5e-1

block_x = np.r_[125.0, 225.0]
block_y = np.r_[-50.0, 50.0]
block_z = np.r_[-50.0, 50.0]

# model
model = np.r_[
    np.log(sigmaback),
    np.log(sigmablock),
    np.mean(block_z),
    np.diff(block_z),
    np.mean(block_x),
    np.diff(block_x),
    np.mean(block_y),
    np.diff(block_y),
]

# source
src_loc = np.r_[0.0, 0.0, 0.0]
freq = 10

# receivers
rx_x = np.linspace(-175.0, 175.0, 8)
rx_y = rx_x.copy()
rx_z = np.r_[175.0]
rx_locs = utils.ndgrid(rx_x, rx_y, rx_z)

# mesh
csx, ncx, npadx = 25.0, 16, 10
csz, ncz, npadz = 25.0, 8, 10
pf = 1.5

# primary mesh
hx = [(csx, ncx), (csx, npadx, pf)]
hz = [(csz, npadz, -pf), (csz, ncz), (csz, npadz, pf)]
meshp = discretize.CylMesh([hx, 1.0, hz], x0="0CC")

# secondary mesh
h = [(csz, npadz - 4, -pf), (csz, ncz), (csz, npadz - 4, pf)]
meshs = discretize.TensorMesh(3 * [h], x0="CCC")

# mappings
primaryMapping = (
    maps.ExpMap(meshp) * maps.SurjectFull(meshp) * maps.Projection(nP=8, index=[0])
)

mapping = (
    maps.ExpMap(meshs)
    * maps.ParametricBlockInLayer(meshs)
    * maps.Projection(nP=8, index=np.hstack([np.r_[0], np.arange(0, 8)]))
)

primaryMap2Meshs = (
    maps.ExpMap(meshs) * maps.SurjectFull(meshs) * maps.Projection(nP=8, index=[0])
)


class PrimSecFDEMTest(object):

    # --------------------- Run some tests! --------------------- #
    def DataTest(self):
        print("\nTesting Data")
        dpred_primsec = self.secondarySimulation.dpred(model, f=self.fields_primsec)
        dpred_3D = self.simulation3D.dpred(model, f=self.fields_3D)

        nrx_locs = rx_locs.shape[0]
        dpred_primsec = dpred_primsec.reshape(nrx_locs, len(self.rxlist))
        dpred_3D = dpred_3D.reshape(nrx_locs, len(self.rxlist))

        for i in range(len(self.rxlist)):
            rx = self.rxlist[i]
            normps = np.linalg.norm(dpred_primsec[:, i])
            norm3D = np.linalg.norm(dpred_3D[:, i])
            normdiff = np.linalg.norm(dpred_primsec[:, i] - dpred_3D[:, i])
            passed = normdiff < TOL_FWD * np.mean([normps, norm3D])
            print(
                "  Testing {rxfield}{rxorient} {rxcomp}...   "
                "prim-sec: {normps:10.5e}, 3D: {norm3D:10.5e}, "
                "diff: {diff:10.5e}, passed? {passed}".format(
                    rxfield=rx.projField,
                    rxorient=rx.orientation,
                    rxcomp=rx.component,
                    normps=normps,
                    norm3D=norm3D,
                    diff=normdiff,
                    passed=passed,
                )
            )

            self.assertTrue(passed)
        return True

    def JvecTest(self):
        print("\nTesting Jvec")
        x0 = model

        def fun(x):
            return [
                self.secondarySimulation.dpred(x),
                lambda x: self.secondarySimulation.Jvec(x0, x, f=self.fields_primsec),
            ]

        return tests.checkDerivative(fun, x0, num=2, plotIt=False)

    def AdjointTest(self):
        print("\nTesting adjoint")

        m = model
        f = self.fields_primsec
        v = np.random.rand(self.secondarySurvey.nD)
        w = np.random.rand(self.secondarySimulation.sigmaMap.nP)

        vJw = v.dot(self.secondarySimulation.Jvec(m, w, f))
        wJtv = w.dot(self.secondarySimulation.Jtvec(m, v, f))
        tol = np.max([TOL_JT * (10 ** int(np.log10(np.abs(vJw)))), FLR])
        passed = np.abs(vJw - wJtv) < tol
        print(
            " J: {}, JT: {}, diff: {}, tol: {}, passed? {}".format(
                vJw, wJtv, vJw - wJtv, tol, passed
            )
        )

        return passed


class PrimSecFDEMSrcTest_Cyl2Cart_EB_EB(unittest.TestCase, PrimSecFDEMTest):
    @classmethod
    def setUpClass(self):

        print("\n------- Testing Primary Secondary Source EB -> EB --------\n")
        # receivers
        self.rxlist = []
        for rxtype in ["MagneticFluxDensity", "ElectricField"]:
            rx = getattr(fdem.Rx, "Point{}".format(rxtype))
            for orientation in ["x", "y", "z"]:
                for comp in ["real", "imag"]:
                    self.rxlist.append(
                        rx(rx_locs, component=comp, orientation=orientation)
                    )

        # primary
        self.primarySimulation = fdem.Simulation3DMagneticFluxDensity(
            meshp, sigmaMap=primaryMapping
        )
        self.primarySimulation.solver = Solver
        primarySrc = fdem.Src.MagDipole(self.rxlist, frequency=freq, location=src_loc)
        self.primarySurvey = fdem.Survey([primarySrc])

        self.secondarySrc = fdem.Src.PrimSecMappedSigma(
            self.rxlist,
            freq,
            self.primarySimulation,
            self.primarySurvey,
            primaryMap2Meshs,
        )
        self.secondarySurvey = fdem.Survey([self.secondarySrc])
        # Secondary Problem
        self.secondarySimulation = fdem.Simulation3DMagneticFluxDensity(
            meshs, survey=self.secondarySurvey, sigmaMap=mapping
        )
        self.secondarySimulation.solver = Solver

        # Full 3D problem to compare with
        self.survey3D = fdem.Survey([primarySrc])

        self.simulation3D = fdem.Simulation3DMagneticFluxDensity(
            meshs, survey=self.survey3D, sigmaMap=mapping
        )
        self.simulation3D.solver = Solver

        # solve and store fields
        print("   solving primary - secondary")
        self.fields_primsec = self.secondarySimulation.fields(model)
        print("     ... done")

        self.fields_primsec = self.secondarySimulation.fields(model)
        print("   solving 3D")
        self.fields_3D = self.simulation3D.fields(model)
        print("     ... done")

        return None

    # --------------------- Run some tests! --------------------- #
    def test_data_EB(self):
        self.DataTest()

    def test_Jvec_EB(self):
        self.JvecTest()

    def test_Jadjoint_EB(self):
        self.AdjointTest()


class PrimSecFDEMSrcTest_Cyl2Cart_HJ_EB(unittest.TestCase, PrimSecFDEMTest):
    @classmethod
    def setUpClass(self):

        print("\n------- Testing Primary Secondary Source HJ -> EB --------\n")
        # receivers
        self.rxlist = []
        for rxtype in ["MagneticFluxDensity", "ElectricField"]:
            rx = getattr(fdem.Rx, "Point{}".format(rxtype))
            for orientation in ["x", "y", "z"]:
                for comp in ["real", "imag"]:
                    self.rxlist.append(
                        rx(rx_locs, component=comp, orientation=orientation)
                    )

        # primary
        self.primarySimulation = fdem.Simulation3DCurrentDensity(
            meshp, sigmaMap=primaryMapping
        )
        self.primarySimulation.solver = Solver
        s_e = np.zeros(meshp.nF)
        inds = meshp.nFx + utils.closestPoints(meshp, src_loc, gridLoc="Fz")
        s_e[inds] = 1.0 / csz
        primarySrc = fdem.Src.RawVec_e(
            self.rxlist, frequency=freq, s_e=s_e / meshp.area
        )
        self.primarySurvey = fdem.Survey([primarySrc])

        # Secondary Problem
        self.secondarySrc = fdem.Src.PrimSecMappedSigma(
            self.rxlist,
            freq,
            self.primarySimulation,
            self.primarySurvey,
            primaryMap2Meshs,
        )
        self.secondarySurvey = fdem.Survey([self.secondarySrc])

        self.secondarySimulation = fdem.Simulation3DElectricField(
            meshs,
            survey=self.secondarySurvey,
            sigmaMap=mapping,
        )
        self.secondarySimulation.solver = Solver

        # Full 3D problem to compare with

        s_e3D = np.zeros(meshs.nE)
        inds = meshs.nEx + meshs.nEy + utils.closestPoints(meshs, src_loc, gridLoc="Ez")
        s_e3D[inds] = [1.0 / (len(inds))] * len(inds)

        src3D = fdem.Src.RawVec_e(self.rxlist, frequency=freq, s_e=s_e3D)
        self.survey3D = fdem.Survey([src3D])

        self.simulation3D = fdem.Simulation3DElectricField(
            meshs, survey=self.survey3D, sigmaMap=mapping
        )
        self.simulation3D.solver = Solver
        self.simulation3D.model = model

        # solve and store fields
        print("   solving primary - secondary")
        self.fields_primsec = self.secondarySimulation.fields(model)
        print("     ... done")

        self.fields_primsec = self.secondarySimulation.fields(model)
        print("   solving 3D")
        self.fields_3D = self.simulation3D.fields(model)
        print("     ... done")

        return None

    # --------------------- Run some tests! --------------------- #
    def test_data_HJ(self):
        self.DataTest()

    def test_Jvec_HJ(self):
        self.JvecTest()

    def test_Jadjoint_HJ(self):
        self.AdjointTest()
