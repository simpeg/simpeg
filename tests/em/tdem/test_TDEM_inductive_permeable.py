from __future__ import division, print_function
import unittest

import discretize
from discretize import utils
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import mu_0, inch, foot
import time

from SimPEG.electromagnetics import time_domain as tdem
from SimPEG import utils, maps
from SimPEG.utils import Zero

from pymatsolver import Pardiso

plotIt = False
TOL = 1e-4


class TestInductiveSourcesPermeability(unittest.TestCase):
    def setUp(self):
        target_mur = [1, 50, 100, 200]
        target_l = 500
        target_r = 50
        sigma_back = 1e-5
        radius_loop = 100

        model_names = ["target_{}".format(mur) for mur in target_mur]

        # Set up a Cyl mesh
        csx = 5.0  # cell size in the x-direction
        csz = 5.0  # cell size in the z-direction
        domainx = 100  # go out 500m from the well

        # padding parameters
        npadx, npadz = 15, 15  # number of padding cells
        pfx = 1.4  # expansion factor for the padding to infinity
        pfz = 1.4

        ncz = int(target_l / csz)
        mesh = discretize.CylMesh(
            [
                [(csx, int(domainx / csx)), (csx, npadx, pfx)],
                1,
                [(csz, npadz, -pfz), (csz, ncz), (csz, npadz, pfz)],
            ]
        )
        mesh.x0 = [0, 0, -mesh.hz[: npadz + ncz].sum()]

        # Plot the mesh
        if plotIt:
            mesh.plotGrid()
            plt.show()

        self.radius_loop = radius_loop
        self.target_mur = target_mur
        self.target_l = target_l
        self.target_r = target_r
        self.sigma_back = sigma_back
        self.model_names = model_names
        self.mesh = mesh

    def test_permeable_sources(self):

        target_mur = self.target_mur
        target_l = self.target_l
        target_r = self.target_r
        sigma_back = self.sigma_back
        model_names = self.model_names
        mesh = self.mesh
        radius_loop = self.radius_loop

        # Assign physical properties on the mesh
        def populate_target(mur):
            mu_model = np.ones(mesh.nC)
            x_inds = mesh.gridCC[:, 0] < target_r
            z_inds = (mesh.gridCC[:, 2] <= 0) & (mesh.gridCC[:, 2] >= -target_l)
            mu_model[x_inds & z_inds] = mur
            return mu_0 * mu_model

        mu_dict = {key: populate_target(mu) for key, mu in zip(model_names, target_mur)}
        sigma = np.ones(mesh.nC) * sigma_back

        # Plot the models
        if plotIt:
            xlim = np.r_[-200, 200]  # x-limits in meters
            zlim = np.r_[-1.5 * target_l, 10.0]  # z-limits in meters. (z-positive up)

            fig, ax = plt.subplots(
                1, len(model_names), figsize=(6 * len(model_names), 5)
            )
            if len(model_names) == 1:
                ax = [ax]

            for a, key in zip(ax, model_names):
                plt.colorbar(
                    mesh.plotImage(
                        mu_dict[key],
                        ax=a,
                        pcolorOpts={"norm": LogNorm()},  # plot on a log-scale
                        mirror=True,
                    )[0],
                    ax=a,
                )
                a.set_title("{}".format(key), fontsize=13)
                #     cylMeshGen.mesh.plotGrid(ax=a, slice='theta') # uncomment to plot the mesh on top of this
                a.set_xlim(xlim)
                a.set_ylim(zlim)
            plt.tight_layout()
            plt.show()

        ramp = [
            (1e-5, 20),
            (1e-4, 20),
            (3e-4, 20),
            (1e-3, 20),
            (3e-3, 20),
            (1e-2, 20),
            (3e-2, 20),
            (1e-1, 20),
            (3e-1, 20),
            (1, 50),
        ]
        time_steps = ramp

        time_mesh = discretize.TensorMesh([ramp])
        offTime = 10000
        waveform = tdem.Src.QuarterSineRampOnWaveform(
            ramp_on=np.r_[1e-4, 20], ramp_off=offTime - np.r_[1e-4, 0]
        )

        if plotIt:
            wave = np.r_[[waveform.eval(t) for t in time_mesh.gridN]]
            plt.plot(time_mesh.gridN, wave)
            plt.plot(time_mesh.gridN, np.zeros(time_mesh.nN), "-|", color="k")
            plt.show()

        src_magnetostatic = tdem.Src.CircularLoop(
            [],
            location=np.r_[0.0, 0.0, 0.0],
            orientation="z",
            radius=100,
        )

        src_ramp_on = tdem.Src.CircularLoop(
            [],
            location=np.r_[0.0, 0.0, 0.0],
            orientation="z",
            radius=100,
            waveform=waveform,
        )

        src_list = [src_magnetostatic]
        src_list_late_ontime = [src_ramp_on]

        survey = tdem.Survey(source_list=src_list)
        survey_late_ontime = tdem.Survey(src_list_late_ontime)

        prob = tdem.Simulation3DMagneticFluxDensity(
            mesh=mesh,
            survey=survey,
            time_steps=time_steps,
            sigmaMap=maps.IdentityMap(mesh),
            solver=Pardiso,
        )
        prob_late_ontime = tdem.Simulation3DMagneticFluxDensity(
            mesh=mesh,
            survey=survey_late_ontime,
            time_steps=time_steps,
            sigmaMap=maps.IdentityMap(mesh),
            solver=Pardiso,
        )

        fields_dict = {}

        for key in model_names:
            t = time.time()
            print("--- Running {} ---".format(key))

            prob_late_ontime.mu = mu_dict[key]
            fields_dict[key] = prob_late_ontime.fields(sigma)

            print(" ... done. Elapsed time {}".format(time.time() - t))
            print("\n")

            b_magnetostatic = {}
            b_late_ontime = {}

        for key in model_names:
            prob.mu = mu_dict[key]
            prob.sigma = sigma
            b_magnetostatic[key] = src_magnetostatic.bInitial(prob)

            prob_late_ontime.mu = mu_dict[key]
            b_late_ontime[key] = utils.mkvc(fields_dict[key][:, "b", -1])

        if plotIt:
            fig, ax = plt.subplots(
                len(model_names), 2, figsize=(3 * len(model_names), 5)
            )

            for i, key in enumerate(model_names):
                ax[i][0].semilogy(
                    np.absolute(b_magnetostatic[key]), label="magnetostatic"
                )
                ax[i][0].semilogy(np.absolute(b_late_ontime[key]), label="late on-time")
                ax[i][0].legend()

                ax[i][1].semilogy(
                    np.absolute(b_magnetostatic[key] - b_late_ontime[key])
                )
            plt.tight_layout()
            plt.show()

        print("Testing TDEM with permeable targets")
        passed = []
        for key in model_names:
            norm_magneotstatic = np.linalg.norm(b_magnetostatic[key])
            norm_late_ontime = np.linalg.norm(b_late_ontime[key])
            norm_diff = np.linalg.norm(b_magnetostatic[key] - b_late_ontime[key])
            passed_test = (
                norm_diff / (0.5 * (norm_late_ontime + norm_magneotstatic)) < TOL
            )
            print("\n{}".format(key))
            print(
                "||magnetostatic||: {:1.2e}, "
                "||late on-time||: {:1.2e}, "
                "||difference||: {:1.2e} passed?: {}".format(
                    norm_magneotstatic, norm_late_ontime, norm_diff, passed_test
                )
            )

            passed += [passed_test]

        assert all(passed)

        prob.sigma = 1e-4 * np.ones(mesh.nC)
        v = utils.mkvc(np.random.rand(mesh.nE))
        w = utils.mkvc(np.random.rand(mesh.nF))
        assert np.all(
            mesh.getEdgeInnerProduct(1e-4 * np.ones(mesh.nC)) * v == prob.MeSigma * v
        )

        assert np.all(
            mesh.getEdgeInnerProduct(1e-4 * np.ones(mesh.nC), invMat=True) * v
            == prob.MeSigmaI * v
        )
        assert np.all(
            mesh.getFaceInnerProduct(1.0 / 1e-4 * np.ones(mesh.nC)) * w
            == prob.MfRho * w
        )

        assert np.all(
            mesh.getFaceInnerProduct(1.0 / 1e-4 * np.ones(mesh.nC), invMat=True) * w
            == prob.MfRhoI * w
        )

        prob.rho = 1.0 / 1e-3 * np.ones(mesh.nC)
        v = utils.mkvc(np.random.rand(mesh.nE))
        w = utils.mkvc(np.random.rand(mesh.nF))

        np.testing.assert_allclose(
            mesh.getEdgeInnerProduct(1e-3 * np.ones(mesh.nC)) * v, prob.MeSigma * v
        )

        np.testing.assert_allclose(
            mesh.getEdgeInnerProduct(1e-3 * np.ones(mesh.nC), invMat=True) * v,
            prob.MeSigmaI * v,
        )

        np.testing.assert_allclose(
            mesh.getFaceInnerProduct(1.0 / 1e-3 * np.ones(mesh.nC)) * w, prob.MfRho * w
        )

        np.testing.assert_allclose(
            mesh.getFaceInnerProduct(1.0 / 1e-3 * np.ones(mesh.nC), invMat=True) * w,
            prob.MfRhoI * w,
        )
