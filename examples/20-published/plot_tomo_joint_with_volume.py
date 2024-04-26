"""
Straight Ray with Volume Data Misfit Term
=========================================

Based on the SEG abstract Heagy, Cockett and Oldenburg, 2014.

Heagy, L. J., Cockett, A. R., & Oldenburg, D. W. (2014, August 5).
Parametrized Inversion Framework for Proppant Volume in a Hydraulically
Fractured Reservoir. SEG Technical Program Expanded Abstracts 2014.
Society of Exploration Geophysicists. doi:10.1190/segam2014-1639.1

This example is a simple joint inversion that consists of a

    - data misfit for the tomography problem
    - data misfit for the volume of the inclusions
      (uses the effective medium theory mapping)
    - model regularization

"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from simpeg.seismic import straight_ray_tomography as tomo
import discretize
from simpeg import (
    maps,
    utils,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    data_misfit,
    objective_function,
)


class Volume(objective_function.BaseObjectiveFunction):
    r"""
    A regularization on the volume integral of the model

    .. math::

        \phi_v = || \int_V m dV - \text{knownVolume} ||^2
    """

    def __init__(self, mesh, knownVolume=0.0, **kwargs):
        self.mesh = mesh
        self.knownVolume = knownVolume
        super().__init__(**kwargs)

    @property
    def knownVolume(self):
        """known volume"""
        return self._knownVolume

    @knownVolume.setter
    def knownVolume(self, value):
        self._knownVolume = utils.validate_float("knownVolume", value, min_val=0.0)

    def __call__(self, m):
        return (self.estVol(m) - self.knownVolume) ** 2

    def estVol(self, m):
        return np.inner(self.mesh.cell_volumes, m)

    def deriv(self, m):
        # return (self.mesh.cell_volumes * np.inner(self.mesh.cell_volumes, m))
        return (
            2
            * self.mesh.cell_volumes
            * (self.knownVolume - np.inner(self.mesh.cell_volumes, m))
        )  # factor of 2 from deriv of ||estVol - knownVol||^2

    def deriv2(self, m, v=None):
        if v is not None:
            return 2 * utils.mkvc(
                self.mesh.cell_volumes * np.inner(self.mesh.cell_volumes, v)
            )
        else:
            # TODO: this is inefficent. It is a fully dense matrix
            return 2 * sp.csc_matrix(
                np.outer(self.mesh.cell_volumes, self.mesh.cell_volumes)
            )


def run(plotIt=True):
    nC = 40
    de = 1.0
    h = np.ones(nC) * de / nC
    M = discretize.TensorMesh([h, h])

    y = np.linspace(M.cell_centers_y[0], M.cell_centers_x[-1], int(np.floor(nC / 4)))
    rlocs = np.c_[0 * y + M.cell_centers_x[-1], y]
    rx = tomo.Rx(rlocs)

    source_list = [
        tomo.Src(location=np.r_[M.cell_centers_x[0], yi], receiver_list=[rx])
        for yi in y
    ]

    # phi model
    phi0 = 0
    phi1 = 0.65
    phitrue = utils.model_builder.create_block_in_wholespace(
        M.gridCC, [0.4, 0.6], [0.6, 0.4], [phi1, phi0]
    )

    knownVolume = np.sum(phitrue * M.cell_volumes)
    print("True Volume: {}".format(knownVolume))

    # Set up true conductivity model and plot the model transform
    sigma0 = np.exp(1)
    sigma1 = 1e4

    if plotIt:
        fig, ax = plt.subplots(1, 1)
        sigmaMapTest = maps.SelfConsistentEffectiveMedium(
            nP=1000, sigma0=sigma0, sigma1=sigma1, rel_tol=1e-1, maxIter=150
        )
        testphis = np.linspace(0.0, 1.0, 1000)

        sigetest = sigmaMapTest * testphis
        ax.semilogy(testphis, sigetest)
        ax.set_title("Model Transform")
        ax.set_xlabel(r"$\varphi$")
        ax.set_ylabel(r"$\sigma$")

    sigmaMap = maps.SelfConsistentEffectiveMedium(M, sigma0=sigma0, sigma1=sigma1)

    # scale the slowness so it is on a ~linear scale
    slownessMap = maps.LogMap(M) * sigmaMap

    # set up the problem and survey
    survey = tomo.Survey(source_list)
    problem = tomo.Simulation(M, survey=survey, slownessMap=slownessMap)

    if plotIt:
        _, ax = plt.subplots(1, 1)
        cb = plt.colorbar(M.plot_image(phitrue, ax=ax)[0], ax=ax)
        survey.plot(ax=ax)
        cb.set_label(r"$\varphi$")

    # get observed data
    data = problem.make_synthetic_data(phitrue, relative_error=0.03, add_noise=True)
    dpred = problem.dpred(np.zeros(M.nC))

    # objective function pieces
    reg = regularization.WeightedLeastSquares(M)
    dmis = data_misfit.L2DataMisfit(simulation=problem, data=data)
    dmisVol = Volume(mesh=M, knownVolume=knownVolume)
    beta = 0.25
    maxIter = 15

    # without the volume regularization
    opt = optimization.ProjectedGNCG(maxIter=maxIter, lower=0.0, upper=1.0)
    opt.remember("xc")
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=beta)
    inv = inversion.BaseInversion(invProb)

    mopt1 = inv.run(np.zeros(M.nC) + 1e-16)
    print(
        "\nTotal recovered volume (no vol misfit term in inversion): "
        "{}".format(dmisVol(mopt1))
    )

    # with the volume regularization
    vol_multiplier = 9e4
    reg2 = reg
    dmis2 = dmis + vol_multiplier * dmisVol
    opt2 = optimization.ProjectedGNCG(maxIter=maxIter, lower=0.0, upper=1.0)
    opt2.remember("xc")
    invProb2 = inverse_problem.BaseInvProblem(dmis2, reg2, opt2, beta=beta)
    inv2 = inversion.BaseInversion(invProb2)

    mopt2 = inv2.run(np.zeros(M.nC) + 1e-16)
    print("\nTotal volume (vol misfit term in inversion): {}".format(dmisVol(mopt2)))

    # plot results

    if plotIt:
        fig, ax = plt.subplots(1, 1)
        ax.plot(data.dobs)
        ax.plot(dpred)
        ax.plot(problem.dpred(mopt1), "o")
        ax.plot(problem.dpred(mopt2), "s")
        ax.legend(["dobs", "dpred0", "dpred w/o Vol", "dpred with Vol"])

        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        im0 = M.plot_image(phitrue, ax=ax[0])[0]
        im1 = M.plot_image(mopt1, ax=ax[1])[0]
        im2 = M.plot_image(mopt2, ax=ax[2])[0]

        for im in [im0, im1, im2]:
            im.set_clim([0.0, phi1])

        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1])
        plt.colorbar(im2, ax=ax[2])

        ax[0].set_title("true, vol: {:1.3e}".format(knownVolume))
        ax[1].set_title(
            "recovered(no Volume term), vol: {:1.3e} ".format(dmisVol(mopt1))
        )
        ax[2].set_title(
            "recovered(with Volume term), vol: {:1.3e} ".format(dmisVol(mopt2))
        )

        plt.tight_layout()


if __name__ == "__main__":
    run()
    plt.show()
