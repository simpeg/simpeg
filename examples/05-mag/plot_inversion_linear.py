"""
PF: Magnetic: Inversion Linear
===============================

In this example, we showcase the magnetic susceptibility
inversion as introduced in the classic paper of:

Li Y., and D. W. Oldenburg, 1996, 3-D inversion of magnetic data:
Geophysics, 61, no. 2, 394-408.

We first create a synthetic block model and invert
with both smooth and compact norms for comparison.
A solution to the compact norms is found using the Scaled-IRLS
method presented in:

Fournier, D. 2015, A cooperative magnetic inversion method with Lp-norm
regularization. MSc. Thesis, UBC - EOAS Department

"""
import matplotlib.pyplot as plt
import numpy as np

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

    # Define the inducing field parameter
    H0 = (50000, 90, 0)

    # Create a mesh
    dx = 5.

    hxind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 10)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

    # Get index of the center
    midx = int(mesh.nCx/2)
    midy = int(mesh.nCy/2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.vectorNz[-1]

    # We would usually load a topofile
    topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

    # Go from topo to actv cells
    actv = Utils.surface2ind_topo(mesh, topo, 'N')
    actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem],
                      dtype=int) - 1

    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)
    nC = len(actv)

    # Create and array of observation points
    xr = np.linspace(-20., 20., 20)
    yr = np.linspace(-20., 20., 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.vectorNz[-1] + 5.

    # Create a MAGsurvey
    rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    rxLoc = PF.BaseMag.RxObs(rxLoc)
    srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
    survey = PF.BaseMag.LinearSurvey(srcField)

    # We can now create a susceptibility model and generate data
    # Here a simple block in half-space
    model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
    model[(midx-2):(midx+2), (midy-2):(midy+2), -6:-2] = 0.02
    model = Utils.mkvc(model)
    model = model[actv]

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP=nC)

    # Create the forward model operator
    prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv,
                                         silent=True)

    # Pair the survey and problem
    survey.pair(prob)

    # Create a static sensitivity weighting function
    wr = np.sum(prob.F**2., axis=0)**0.5
    wr = (wr/np.max(wr))
    
    # Compute linear forward operator and compute some data
    d = prob.fields(model)

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    data = d #+ np.random.randn(len(d))
    wd = np.ones(len(data))*1.  # Assign flat uncertainties

    survey.dobs = data
    survey.std = wd
    survey.mtrue = model

    # Create sensitivity weights from our linear forward operator
    rxLoc = survey.srcField.rxList[0].locs

    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.norms = [0, 1, 1, 1]
    reg.eps_p, reg.eps_q = 1e-3, 1e-3
    reg.cell_weights = wr

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./wd

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                     maxIterLS=20, maxIterCG=10, tolCG=1e-3)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied

    # Use pick a treshold parameter empirically based on the distribution of
    #  model parameters

    IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3)

    update_Jacobi = Directives.UpdatePreCond()
    update_SensWeight = Directives.UpdateSensWeighting()
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, IRLS, update_Jacobi])

    # Run the inversion
    m0 = np.ones(nC)*1e-4  # Starting model
    mrec = inv.run(m0)

    if plotIt:
        # Here is the recovered susceptibility model
        ypanel = midx
        zpanel = -5
        
        m_l2 = actvMap * invProb.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp = actvMap * mrec
        m_lp[m_lp == -100] = np.nan

        m_true = actvMap * model
        m_true[m_true == -100] = np.nan

        # Plot the data
        fig = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        PF.Magnetics.plot_obs_2D(rxLoc, d=d, fig=fig, ax=ax1,
                                 title='TMI Data')
        PF.Magnetics.plot_obs_2D(rxLoc, d=invProb.dpred, fig=fig, ax=ax2,
                                 title='Predicted Data')

        plt.figure(figsize=(5, 8))

        # Plot L2 model
        ax = plt.subplot(321)
        mesh.plotSlice(m_l2, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(model.min(), model.max()),
                       pcolorOpts={'cmap': 'magma_r', })
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan l2-model.')
        plt.gca().set_aspect('equal')
        plt.ylabel('y')
        ax.xaxis.set_visible(False)
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertica section
        ax = plt.subplot(322)
        mesh.plotSlice(m_l2, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(model.min(), model.max()),
                       pcolorOpts={'cmap': 'magma_r', })
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W l2-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot Lp model
        ax = plt.subplot(323)
        mesh.plotSlice(m_lp, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(model.min(), model.max()),
                       pcolorOpts={'cmap': 'magma_r', })
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertical section
        ax = plt.subplot(324)
        mesh.plotSlice(m_lp, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(model.min(), model.max()),
                       pcolorOpts={'cmap': 'magma_r', })
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot True model
        ax = plt.subplot(325)
        mesh.plotSlice(m_true, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(model.min(), model.max()),
                       pcolorOpts={'cmap': 'magma_r', })
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertical section
        ax = plt.subplot(326)
        mesh.plotSlice(m_true, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(model.min(), model.max()),
                       pcolorOpts={'cmap': 'magma_r', })
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

if __name__ == '__main__':
    run()
    plt.show()
