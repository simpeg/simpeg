"""
FLOW: Richards: 1D: Inversion
=============================

The example shows an inversion of Richards equation in 1D with a
heterogeneous hydraulic conductivity function.

The haverkamp model is used with the same parameters as Celia1990_
the boundary and initial conditions are also the same. The simulation
domain is 40cm deep and is run for an hour with an exponentially
increasing time step that has a maximum of one minute. The general
setup of the experiment is an infiltration front that advances
downward through the model over time.

The model chosen is the saturated hydraulic conductivity inside
the hydraulic conductivity function (using haverkamp). The initial
model is chosen to be the background (1e-3 cm/s). The saturation data
has 2% random Gaussian noise added.

The figure shows the recovered saturated hydraulic conductivity
next to the true model. The other two figures show the saturation
field for the entire simulation for the true and recovered models.

Rowan Cockett - 21/12/2016

.. _Celia1990: http://www.webpages.uidaho.edu/ch/papers/Celia.pdf
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import discretize
from SimPEG import maps
from SimPEG import regularization
from SimPEG import data_misfit
from SimPEG import optimization
from SimPEG import inverse_problem
from SimPEG import directives
from SimPEG import inversion

from SimPEG.flow import richards


def run(plotIt=True):

    M = discretize.TensorMesh([np.ones(40)], x0="N")
    M.setCellGradBC("dirichlet")
    # We will use the haverkamp empirical model with parameters from Celia1990
    k_fun, theta_fun = richards.empirical.haverkamp(
        M,
        A=1.1750e06,
        gamma=4.74,
        alpha=1.6110e06,
        theta_s=0.287,
        theta_r=0.075,
        beta=3.96,
    )

    # Here we are making saturated hydraulic conductivity
    # an exponential mapping to the model (defined below)
    k_fun.KsMap = maps.ExpMap(nP=M.nC)

    # Setup the boundary and initial conditions
    bc = np.array([-61.5, -20.7])
    h = np.zeros(M.nC) + bc[0]
    prob = richards.SimulationNDCellCentered(
        M,
        hydraulic_conductivity=k_fun,
        water_retention=theta_fun,
        boundary_conditions=bc,
        initial_conditions=h,
        do_newton=False,
        method="mixed",
        debug=False,
    )
    prob.time_steps = [(5, 25, 1.1), (60, 40)]

    # Create the survey
    locs = -np.arange(2, 38, 4.0).reshape(-1, 1)
    times = np.arange(30, prob.time_mesh.vectorCCx[-1], 60)
    rxSat = richards.receivers.Saturation(locs, times)
    survey = richards.Survey([rxSat])
    prob.survey = survey

    # Create a simple model for Ks
    Ks = 1e-3
    mtrue = np.ones(M.nC) * np.log(Ks)
    mtrue[15:20] = np.log(5e-2)
    mtrue[20:35] = np.log(3e-3)
    mtrue[35:40] = np.log(1e-2)
    m0 = np.ones(M.nC) * np.log(Ks)

    # Create some synthetic data and fields
    relative = 0.02  # The standard deviation for the noise
    Hs = prob.fields(mtrue)
    data = prob.make_synthetic_data(
        mtrue, relative_error=relative, f=Hs, add_noise=True
    )

    # Setup a pretty standard inversion
    reg = regularization.WeightedLeastSquares(M, alpha_s=1e-1)
    dmis = data_misfit.L2DataMisfit(simulation=prob, data=data)
    opt = optimization.InexactGaussNewton(maxIter=20, maxIterCG=10)
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    beta = directives.BetaSchedule(coolingFactor=4)
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e2)
    target = directives.TargetMisfit()
    dir_list = [beta, betaest, target]
    inv = inversion.BaseInversion(invProb, directiveList=dir_list)

    mopt = inv.run(m0)

    Hs_opt = prob.fields(mopt)

    if plotIt:
        plt.figure(figsize=(14, 9))

        ax = plt.subplot(121)
        plt.semilogx(np.exp(np.c_[mopt, mtrue]), M.gridCC)
        plt.xlabel("Saturated Hydraulic Conductivity, $K_s$")
        plt.ylabel("Depth, cm")
        plt.semilogx([10 ** -3.9] * len(locs), locs, "ro")
        plt.legend(("$m_{rec}$", "$m_{true}$", "Data locations"), loc=4)

        ax = plt.subplot(222)
        mesh2d = discretize.TensorMesh([prob.time_mesh.hx / 60, prob.mesh.hx], "0N")
        sats = [theta_fun(_) for _ in Hs]
        clr = mesh2d.plotImage(np.c_[sats][1:, :], ax=ax)
        cmap0 = matplotlib.cm.RdYlBu_r
        clr[0].set_cmap(cmap0)
        c = plt.colorbar(clr[0])
        c.set_label("Saturation $\\theta$")
        plt.xlabel("Time, minutes")
        plt.ylabel("Depth, cm")
        plt.title("True saturation over time")

        ax = plt.subplot(224)
        mesh2d = discretize.TensorMesh([prob.time_mesh.hx / 60, prob.mesh.hx], "0N")
        sats = [theta_fun(_) for _ in Hs_opt]
        clr = mesh2d.plotImage(np.c_[sats][1:, :], ax=ax)
        cmap0 = matplotlib.cm.RdYlBu_r
        clr[0].set_cmap(cmap0)
        c = plt.colorbar(clr[0])
        c.set_label("Saturation $\\theta$")
        plt.xlabel("Time, minutes")
        plt.ylabel("Depth, cm")
        plt.title("Recovered saturation over time")

        plt.tight_layout()


if __name__ == "__main__":
    run()
    plt.show()
