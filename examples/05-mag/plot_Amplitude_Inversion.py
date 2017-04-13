"""
PF: Magnetic Amplitude Inversion
================================

In this example, we invert magnetic field data simulated
from a simple block model affected by remanent magnetization.
The algorithm builtds upon the research done at CSM:

Li, Y., S. E. Shearer, M. M. Haney, and N. Dannemiller, 2010,
Comprehensive approaches to 3D inversion of magnetic data affected by
remanent magnetization:  Geophysics, 75, no. 1, 1-11

The steps are:
1- SETUP: Create a synthetic model and calculate TMI data. This will
simulate the usual magnetic experiment.

2- PROCESSING: Invert for an equivalent source layer to extract
3-component magnetic field data. The components are then used to
calculate amplitude data.

3- INVERSION: Invert for an effective susceptibility model.

For comparison, the same TMI data are first inverted with the usual
induced assumption, both with smooth and compact norms. Note that
the recovered susceptibility model is highly distorted due to the
incorrect magnetization assumption, compared to the more reboust solution
using the amplitude data.

"""
import numpy as np
import matplotlib.pyplot as plt

from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion
from SimPEG import PF


def run(plotIt=True):

    # # STEP 1: Setup and data simulation # #

    # Magnetic inducing field parameter (A,I,D)
    B = [50000, 90, 0]

    # Create a mesh
    dx = 5.

    hxind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 7)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

    # Get index of the center
    midx = int(mesh.nCx/2)
    midy = int(mesh.nCy/2)

    # Lets create a simple flat topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = np.ones_like(xx)*mesh.vectorNz[-1]
    topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

    # Go from topo to actv cells
    actv = Utils.surface2ind_topo(mesh, topo, 'N')
    actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem],
                      dtype=int) - 1

    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)
    nC = int(len(actv))

    # Create and array of observation points
    xr = np.linspace(-30., 30., 20)
    yr = np.linspace(-30., 30., 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = np.ones_like(X) * mesh.vectorNz[-1] + dx

    # Create a MAGsurvey
    rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    rxLoc = PF.BaseMag.RxObs(rxLoc)
    srcField = PF.BaseMag.SrcField([rxLoc], param=(B[0], B[1], B[2]))
    survey = PF.BaseMag.LinearSurvey(srcField)

    # We can now create a susceptibility model and generate data
    # Here a simple block in half-space
    model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
    model[(midx-2):(midx+2), (midy-2):(midy+2), -6:-2] = 0.05
    model = Utils.mkvc(model)
    model = model[actv]

    # We create a magnetization model different than the inducing field
    # to simulate remanent magnetization. Let's do something simple [45,90]
    M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * 45., np.ones(nC) * 90.)

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Create reduced identity map
    idenMap = Maps.IdentityMap(nP=nC)

    # Create the forward problem (forwardOnly)
    prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv,
                                         M=M, silent=True)

    # Pair the survey and problem
    survey.pair(prob)

    # Compute forward model some data
    d = prob.fields(model)

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    d_TMI = d + np.random.randn(len(d))*0.
    wd = np.ones(len(d_TMI))  # Assign flat uncertainties
    survey.dobs = d_TMI
    survey.std = wd

    # For comparison, let's run the inversion assuming an induced response
    M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * B[1], np.ones(nC) * B[2])

    # Reset the magnetization
    prob.M = M
    prob._F = None

    # Create a regularization function, in this case l2l2
    wr = np.sum(prob.F**2., axis=0)**0.5
    wr = (wr/np.max(wr))

    # Create a regularization
    reg_Susc = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg_Susc.cell_weights = wr
    reg_Susc.norms = ([0, 1, 1, 1])
    reg_Susc.eps_p, reg_Susc.eps_q = 1e-3, 1e-3

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1/wd

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                     maxIterLS=20, maxIterCG=10, tolCG=1e-3)
    invProb = InvProblem.BaseInvProblem(dmis, reg_Susc, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    # Use pick a treshold parameter empirically based on the distribution of
    #  model parameters
    IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3)
    update_Jacobi = Directives.UpdatePreCond()
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, IRLS, update_Jacobi])

    # Run the inversion
    m0 = np.ones(nC)*1e-4  # Starting model
    mrec = inv.run(m0)

    # # STEP 2: Equivalent source inversion and amplitude data # #

    # Get the layer of cells directly below topo
    surf = Utils.surface2ind_topo(mesh, topo, 'N', layer=True)
    nC = int(np.sum(surf))  # Number of active cells

    # Create active map to go from reduce set to full
    surfMap = Maps.InjectActiveCells(mesh, surf, -100)

    # Create identity map
    idenMap = Maps.IdentityMap(nP=nC)

    # Create MAG equivalent layer problem
    prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=surf,
                                         equiSourceLayer=True, silent=True)
    prob.solverOpts['accuracyTol'] = 1e-4

    # Pair the survey and problem
    survey.pair(prob)

    # Create a regularization function, in this case l2l2
    reg = Regularization.Simple(mesh, indActive=surf)
    reg.mref = np.zeros(nC)

    # Specify how the optimization will proceed
    opt = Optimization.ProjectedGNCG(maxIter=150, lower=-np.inf,
                                     upper=np.inf, maxIterLS=20,
                                     maxIterCG=20, tolCG=1e-3)

    # Define misfit function (obs-calc)
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./survey.std

    # Create the default L2 inverse problem from the above objects
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

    # Specify how the initial beta is found
    betaest = Directives.BetaEstimate_ByEig()

    # Beta schedule for inversion
    betaSchedule = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

    # Target misfit to stop the inversion
    targetMisfit = Directives.TargetMisfit()
    update_Jacobi = Directives.UpdatePreCond()
    # Put all the parts together
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, betaSchedule,
                                                 update_Jacobi, targetMisfit])

    # Run the equivalent source inversion
    mstart = np.zeros(nC)
    mrec = inv.run(mstart)

    # COMPUTE AMPLITUDE DATA
    # Now that we have an equialent source layer, we can forward model all
    # three components of the field and add them up:
    # |B| = ( Bx**2 + Bx**2 + Bx**2 )**0.5

    # Won't store the sensitivity and output 'xyz' data.
    prob.forwardOnly = True
    pred_x = prob.Intrgl_Fwr_Op(m=mrec, recType='x')
    pred_y = prob.Intrgl_Fwr_Op(m=mrec, recType='y')
    pred_z = prob.Intrgl_Fwr_Op(m=mrec, recType='z')

    ndata = survey.nD

    d_amp = np.sqrt(pred_x**2. +
                    pred_y**2. +
                    pred_z**2.)

    rxLoc = survey.srcField.rxList[0].locs

    # # STEP 3: RUN AMPLITUDE INVERSION ##

    # Now that we have |B| data, we can invert. This is a non-linear inversion,
    # which requires some special care for the sensitivity weighting
    # (see Directives)

    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)
    nC = int(len(actv))

    # Create identity map
    idenMap = Maps.IdentityMap(nP=nC)

    # Create the forward model operator
    prob = PF.Magnetics.MagneticAmplitude(mesh, chiMap=idenMap,
                                          actInd=actv, silent=True)

    # Define starting model
    mstart = np.ones(len(actv))*1e-4
    prob.chi = mstart

    # # Change the survey to xyz components
    # survey.srcField.rxList[0].rxType = 'xyz'

    # Pair the survey and problem
    survey.pair(prob)

    # Re-set the observations to |B|
    survey.dobs = d_amp

    # Create a sparse regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.norms = ([0, 1, 1, 1])
    reg.eps_p, reg.eps_q = 1e-3, 1e-3

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1/survey.std

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                     maxIterLS=20, maxIterCG=10,
                                     tolCG=1e-3)

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

    # Here is the list of directives
    betaest = Directives.BetaEstimate_ByEig()

    # Specify the sparse norms
    IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                                  minGNiter=2,
                                  chifact=0.25)

    # Special directive specific to the mag amplitude problem. The sensitivity
    # weights are update between each iteration.
    update_SensWeight = Directives.UpdateSensWeighting()
    update_Jacobi = Directives.UpdatePreCond()

    # Put all together
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, IRLS, update_SensWeight, update_Jacobi])

    # Invert
    mrec = inv.run(mstart)

    if plotIt:
        # Here is the recovered susceptibility model
        m_l2 = actvMap * reg.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp = actvMap * mrec
        m_lp[m_lp == -100] = np.nan

        m_l2_susc = actvMap * reg_Susc.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp_susc = actvMap * reg_Susc.model
        m_lp[m_lp == -100] = np.nan

        m_true = actvMap * model
        m_true[m_true == -100] = np.nan

        # Plot the data
        fig = plt.figure(figsize=(6, 6))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_TMI, fig=fig, ax=ax1,
                                       title='TMI Data')
        out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_amp, fig=fig, ax=ax2,
                                       title='Amplitude Data')

        plt.figure()

        # Plot L2 model
        vmin = model.min()
        vmax = model.max() / 5.

        ax = plt.subplot(321)
        mesh.plotSlice(m_l2_susc, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax),
                       pcolorOpts={'cmap': 'pink_r', })
        plt.title('Susceptibility l2-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Amplitude model section
        ax = plt.subplot(322)
        dat = mesh.plotSlice(m_l2, ax=ax, normal='Y', ind=midx,
                             grid=True, clim=(vmin, vmax),
                             pcolorOpts={'cmap': 'pink_r', })
        plt.colorbar(dat[0], orientation="vertical",
                    ticks=np.linspace(vmin, vmax, 4))
        plt.title('Amplitude l2-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot Lp model
        vmin = model.min()
        vmax = model.max()

        ax = plt.subplot(323)
        mesh.plotSlice(m_lp_susc, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax),
                       pcolorOpts={'cmap': 'pink_r', })
        plt.title('Susceptibility lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertical section
        ax = plt.subplot(324)
        dat = mesh.plotSlice(m_lp, ax=ax, normal='Y', ind=midx,
                             grid=True, clim=(vmin, vmax),
                             pcolorOpts={'cmap': 'pink_r', })
        plt.colorbar(dat[0], orientation="vertical",
                    ticks=np.linspace(vmin, vmax, 4))
        plt.title('Amplitude lp-model')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot True model
        ax = plt.subplot(326)
        dat = mesh.plotSlice(m_true, ax=ax, normal='Y', ind=midx,
                             grid=True, clim=(vmin, vmax),
                             pcolorOpts={'cmap': 'pink_r', })
        plt.colorbar(dat[0], orientation="vertical",
                     ticks=np.linspace(vmin, vmax, 4))
        plt.title('True model')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')


if __name__ == '__main__':
    run()
    plt.show()
