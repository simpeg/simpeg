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

from SimPEG import Mesh
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion

from SimPEG.FLOW import Richards


def run(plotIt=True):

    M = Mesh.TensorMesh([np.ones(40)], x0='N')
    M.setCellGradBC('dirichlet')
    # We will use the haverkamp empirical model with parameters from Celia1990
    k_fun, theta_fun = Richards.Empirical.haverkamp(
        M, A=1.1750e+06, gamma=4.74, alpha=1.6110e+06,
        theta_s=0.287, theta_r=0.075, beta=3.96
    )

    # Here we are making saturated hydraulic conductivity
    # an exponential mapping to the model (defined below)
    k_fun.KsMap = Maps.ExpMap(nP=M.nC)

    # Setup the boundary and initial conditions
    bc = np.array([-61.5, -20.7])
    h = np.zeros(M.nC) + bc[0]
    prob = Richards.RichardsProblem(
        M,
        hydraulic_conductivity=k_fun,
        water_retention=theta_fun,
        boundary_conditions=bc, initial_conditions=h,
        do_newton=False, method='mixed', debug=False
    )
    prob.timeSteps = [(5, 25, 1.1), (60, 40)]

    # Create the survey
    locs = -np.arange(2, 38, 4.)
    times = np.arange(30, prob.timeMesh.vectorCCx[-1], 60)
    rxSat = Richards.SaturationRx(locs, times)
    survey = Richards.RichardsSurvey([rxSat])
    survey.pair(prob)

    # Create a simple model for Ks
    Ks = 1e-3
    mtrue = np.ones(M.nC)*np.log(Ks)
    mtrue[15:20] = np.log(5e-2)
    mtrue[20:35] = np.log(3e-3)
    mtrue[35:40] = np.log(1e-2)
    m0 = np.ones(M.nC)*np.log(Ks)

    # Create some synthetic data and fields
    stdev = 0.02  # The standard deviation for the noise
    Hs = prob.fields(mtrue)
    survey.makeSyntheticData(mtrue, std=stdev, f=Hs, force=True)

    # Setup a pretty standard inversion
    reg = Regularization.Tikhonov(M, alpha_s=1e-1)
    dmis = DataMisfit.l2_DataMisfit(survey)
    opt = Optimization.InexactGaussNewton(maxIter=20, maxIterCG=10)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    beta = Directives.BetaSchedule(coolingFactor=4)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e2)
    target = Directives.TargetMisfit()
    dir_list = [beta, betaest, target]
    inv = Inversion.BaseInversion(invProb, directiveList=dir_list)

    mopt = inv.run(m0)

    Hs_opt = prob.fields(mopt)

    if plotIt:
        plt.figure(figsize=(14, 9))

        ax = plt.subplot(121)
        plt.semilogx(np.exp(np.c_[mopt, mtrue]), M.gridCC)
        plt.xlabel('Saturated Hydraulic Conductivity, $K_s$')
        plt.ylabel('Depth, cm')
        plt.semilogx([10**-3.9]*len(locs), locs, 'ro')
        plt.legend(('$m_{rec}$', '$m_{true}$', 'Data locations'), loc=4)

        ax = plt.subplot(222)
        mesh2d = Mesh.TensorMesh([prob.timeMesh.hx/60, prob.mesh.hx], '0N')
        sats = [theta_fun(_) for _ in Hs]
        clr = mesh2d.plotImage(np.c_[sats][1:, :], ax=ax)
        cmap0 = matplotlib.cm.RdYlBu_r
        clr[0].set_cmap(cmap0)
        c = plt.colorbar(clr[0])
        c.set_label('Saturation $\\theta$')
        plt.xlabel('Time, minutes')
        plt.ylabel('Depth, cm')
        plt.title('True saturation over time')

        ax = plt.subplot(224)
        mesh2d = Mesh.TensorMesh([prob.timeMesh.hx/60, prob.mesh.hx], '0N')
        sats = [theta_fun(_) for _ in Hs_opt]
        clr = mesh2d.plotImage(np.c_[sats][1:, :], ax=ax)
        cmap0 = matplotlib.cm.RdYlBu_r
        clr[0].set_cmap(cmap0)
        c = plt.colorbar(clr[0])
        c.set_label('Saturation $\\theta$')
        plt.xlabel('Time, minutes')
        plt.ylabel('Depth, cm')
        plt.title('Recovered saturation over time')

        plt.tight_layout()

if __name__ == '__main__':
    run()
    plt.show()
