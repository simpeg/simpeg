from __future__ import print_function

import unittest
import warnings

import discretize
import numpy as np
from geoana.em.static import MagneticDipoleWholeSpace
from scipy.constants import mu_0
from SimPEG import maps, utils
from SimPEG.electromagnetics import frequency_domain as fdem

TOL = 0.5  # relative tolerance (to norm of soln)
plotIt = False

import matplotlib.pyplot as plt


class TestSimpleSourcePropertiesTensor(unittest.TestCase):
    def setUp(self):
        cs = 10.0
        ncx, ncy, ncz = 30.0, 30.0, 30.0
        npad = 10.0
        hx = [(cs, npad, -1.5), (cs, ncx), (cs, npad, 1.5)]
        hy = [(cs, npad, -1.5), (cs, ncy), (cs, npad, 1.5)]
        hz = [(cs, npad, -1.5), (cs, ncz), (cs, npad, 1.5)]
        self.mesh = discretize.TensorMesh([hx, hy, hz], "CCC")
        mapping = maps.ExpMap(self.mesh)

        self.frequency = 1.0

        self.prob_e = fdem.Simulation3DElectricField(self.mesh, sigmaMap=mapping)
        self.prob_b = fdem.Simulation3DMagneticFluxDensity(self.mesh, sigmaMap=mapping)
        self.prob_h = fdem.Simulation3DMagneticField(self.mesh, sigmaMap=mapping)
        self.prob_j = fdem.Simulation3DCurrentDensity(self.mesh, sigmaMap=mapping)

        loc = np.r_[0.0, 0.0, 0.0]
        self.location = utils.mkvc(
            self.mesh.gridCC[utils.closestPoints(self.mesh, loc, "CC"), :]
        )

    def test_MagDipole(self):

        print("\ntesting MagDipole assignments")

        for orient in ["x", "y", "z", "X", "Y", "Z"]:
            src = fdem.sources.MagDipole(
                [],
                frequency=self.frequency,
                location=np.r_[0.0, 0.0, 0.0],
                orientation=orient,
            )
            # test assignments
            assert np.all(src.location == np.r_[0.0, 0.0, 0.0])
            assert src.frequency == self.frequency
            assert src.moment == 1.0

            if orient.upper() == "X":
                orient_vec = np.r_[1.0, 0.0, 0.0]
            elif orient.upper() == "Y":
                orient_vec = np.r_[0.0, 1.0, 0.0]
            elif orient.upper() == "Z":
                orient_vec = np.r_[0.0, 0.0, 1.0]

            print(
                " {0} component. src: {1}, expected: {2}".format(
                    orient, src.orientation, orient_vec
                )
            )
            assert np.all(src.orientation == orient_vec)

    def test_MagDipoleSimpleFail(self):

        print("\ntesting MagDipole error handling")

        with warnings.catch_warnings(record=True):
            fdem.sources.MagDipole(
                [],
                frequency=self.frequency,
                location=np.r_[0.0, 0.0, 0.0],
                orientation=np.r_[1.0, 1.0, 0.0],
            )

    def bPrimaryTest(self, src, probType):
        passed = True
        print(
            "\ntesting bPrimary {}, problem {}, mu {}".format(
                src.__class__.__name__, probType, src.mu / mu_0
            )
        )
        prob = getattr(self, "prob_{}".format(probType))

        bPrimary = src.bPrimary(prob)

        def ana_sol(XYZ):
            return MagneticDipoleWholeSpace(
                location=src.location,
                moment=1.0,
                orientation=src.orientation,
                mu=src.mu,
            ).magnetic_flux_density(XYZ)

        if probType in ["e", "b"]:
            # TODO: clean up how we call analytics
            bx = ana_sol(self.mesh.gridFx)[:, 0]
            by = ana_sol(self.mesh.gridFy)[:, 1]
            bz = ana_sol(self.mesh.gridFz)[:, 2]

            # remove the z faces right next to the source
            ignore_these = (
                (-self.mesh.hx.min() + src.location[0] <= self.mesh.gridFz[:, 0])
                & (self.mesh.gridFz[:, 0] <= self.mesh.hx.min() + src.location[0])
                & (-self.mesh.hy.min() + src.location[1] <= self.mesh.gridFz[:, 1])
                & (self.mesh.gridFz[:, 1] <= self.mesh.hy.min() + src.location[1])
                & (-self.mesh.hz.min() + src.location[2] <= self.mesh.gridFz[:, 2])
                & (self.mesh.gridFz[:, 2] <= self.mesh.hz.min() + src.location[2])
            )

            look_at_these = np.ones(self.mesh.nFx + self.mesh.nFy, dtype=bool)

        elif probType in ["h", "j"]:
            # TODO: clean up how we call analytics
            bx = ana_sol(self.mesh.gridEx)[:, 0]
            by = ana_sol(self.mesh.gridEy)[:, 1]
            bz = ana_sol(self.mesh.gridEz)[:, 2]

            # remove the z faces right next to the source
            ignore_these = (
                (-self.mesh.hx.min() + src.location[0] <= self.mesh.gridEz[:, 0])
                & (self.mesh.gridEz[:, 0] <= self.mesh.hx.min() + src.location[0])
                & (-self.mesh.hy.min() + src.location[1] <= self.mesh.gridEz[:, 1])
                & (self.mesh.gridEz[:, 1] <= self.mesh.hy.min() + src.location[1])
                & (-self.mesh.hz.min() + src.location[2] <= self.mesh.gridEz[:, 2])
                & (self.mesh.gridEz[:, 2] <= self.mesh.hz.min() + src.location[2])
            )

            look_at_these = np.ones(self.mesh.nEx + self.mesh.nEy, dtype=bool)

        look_at_these = np.hstack(
            [look_at_these, np.array(ignore_these == False, dtype=bool)]
        )

        bPrimary_ana = np.hstack([bx, by, bz])

        check = np.linalg.norm(bPrimary[look_at_these] - bPrimary_ana[look_at_these])
        tol = np.linalg.norm(bPrimary[look_at_these]) * TOL
        passed = check < tol

        print(
            "  {}, num: {}, ana: {}, num/ana: {}, num - ana: {}, < {} ? {}".format(
                probType,
                np.linalg.norm(bPrimary[look_at_these]),
                np.linalg.norm(bPrimary_ana[look_at_these]),
                np.linalg.norm(bPrimary[look_at_these])
                / np.linalg.norm(bPrimary_ana[look_at_these]),
                check,
                tol,
                passed,
            )
        )

        if plotIt is True:

            print(self.mesh.vnF)

            fig, ax = plt.subplots(1, 2)
            ax[0].semilogy(np.absolute(bPrimary_ana), linewidth=2.0)
            ax[0].semilogy(np.absolute(bPrimary))
            ax[0].legend(["|num|", "|ana|"])
            ax[0].set_ylim([tol, bPrimary.max() * 2])

            ax[1].semilogy(np.absolute(bPrimary - bPrimary_ana))
            ax[1].legend(["num - ana"])
            ax[1].set_ylim([tol, np.absolute(bPrimary - bPrimary_ana).max() * 2])

            plt.show()

        return passed

    # ------------- GENERAL ------------------ #

    def test_integrate_source_failure(self):
        self.assertRaises(
            TypeError,
            fdem.sources.BaseFDEMSrc,
            [],
            frequency=self.frequency,
            location=self.location,
            integrate=4.0,
        )

    # ------------- TEST MAG DIPOLE ------------------ #

    def test_MagDipole_bPrimaryMu0_e(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "e")

    def test_MagDipole_bPrimaryMu50_e(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "e")

    def test_MagDipole_bPrimaryMu0_b(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "b")

    def test_MagDipole_bPrimaryMu50_b(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "b")

    def test_MagDipole_bPrimaryMu0_h(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "h")

    def test_MagDipole_bPrimaryMu50_h(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "h")

    def test_MagDipole_bPrimaryMu0_h(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "j")

    def test_MagDipole_bPrimaryMu50_h(self):
        src = fdem.sources.MagDipole(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "j")

    # ------------- TEST MAG DIPOLE B FIELD ------------------ #

    def test_MagDipole_Bfield_bPrimaryMu0_e(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "e")

    def test_MagDipole_Bfield_bPrimaryMu50_e(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "e")

    def test_MagDipole_Bfield_bPrimaryMu0_b(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "b")

    def test_MagDipole_Bfield_bPrimaryMu50_b(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "b")

    def test_MagDipole_Bfield_bPrimaryMu0_h(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "h")

    def test_MagDipole_Bfield_bPrimaryMu50_h(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "h")

    def test_MagDipole_Bfield_bPrimaryMu0_h(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "j")

    def test_MagDipole_Bfield_bPrimaryMu50_h(self):
        src = fdem.sources.MagDipole_Bfield(
            [],
            frequency=self.frequency,
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "j")

    # ------------- TEST MAG DIPOLE CIRCULAR LOOP ------------------ #

    def test_CircularLoop_bPrimaryMu0_e(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "e")

    def test_CircularLoop_bPrimaryMu50_e(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "e")

    def test_CircularLoop_bPrimaryMu0_b(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "b")

    def test_CircularLoop_bPrimaryMu50_b(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "b")

    def test_CircularLoop_bPrimaryMu0_h(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "h")

    def test_CircularLoop_bPrimaryMu50_h(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "h")

    def test_CircularLoop_bPrimaryMu0_h(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=mu_0,
        )
        assert self.bPrimaryTest(src, "j")

    def test_CircularLoop_bPrimaryMu50_h(self):
        src = fdem.sources.CircularLoop(
            [],
            frequency=self.frequency,
            radius=np.sqrt(1 / np.pi),
            location=self.location,
            orientation="Z",
            mu=50.0 * mu_0,
        )
        assert self.bPrimaryTest(src, "j")


def test_CircularLoop_test_N_assign():
    """
    Test depreciation of the N argument (now n_turns)
    """
    src = fdem.sources.CircularLoop(
        [],
        frequency=1e-3,
        radius=np.sqrt(1 / np.pi),
        location=[0, 0, 0],
        orientation="Z",
        mu=mu_0,
        current=0.5,
        N=2,
    )
    assert src.n_turns == 2
