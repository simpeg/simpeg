"""
PF: Tiled magnetic Inversion
============================

In this example we run a magnetic example
with a tiling strategy to reduce the computational
cost of the integral equation problem.

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
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle


def run(plotIt=True):

    # Define the inducing field parameter
    H0 = (50000, 90, 0)

    # Create a base mesh
    dx = 5.

    hxind = [(dx, 5, -1.3), (dx, 20), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 20), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 10)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

    # Get index of the center
    midx = int(mesh.nCx/2)
    midy = int(mesh.nCy/2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.vectorNz[-1]

    # Create an array for topography
    topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

    # Go from topo to actv cells
    actv = Utils.surface2ind_topo(mesh, topo, 'N')
    actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem],
                      dtype=int) - 1
    nC = len(actv)
    # Create KDTree for active cells of out mesh
    tree = cKDTree(np.c_[mesh.gridCC[actv, 0],
                         mesh.gridCC[actv, 1],
                         mesh.gridCC[actv, 2]])

    # Create a survey made up two lines, we will set the lines far enough
    # apart so that the tiling strategy is obvious
    xr = np.linspace(-50., 50., 20)
    yr = np.c_[-30, 0, 30]
    X1, Y1 = np.meshgrid(xr, yr)

    yr = np.linspace(-50., 50., 20)
    xr = np.c_[-30, 30]
    X2, Y2 = np.meshgrid(xr, yr)

    X = np.r_[Utils.mkvc(X1), Utils.mkvc(X2)]
    Y = np.r_[Utils.mkvc(Y1), Utils.mkvc(Y2)]

    # Move the observation points 5m above the topo
    Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.vectorNz[-1] + dx

    # Create a MAGsurvey
    xyzLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    rxLoc = PF.BaseMag.RxObs(xyzLoc)
    srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
    survey = PF.BaseMag.LinearSurvey(srcField)

    # Design subsets of data for the tiling strategy
    # # TILE THE PROBLEM
    maxNpoints = 4
    tiles = Utils.modelutils.tileSurveyPoints(xyzLoc, maxNpoints)

    X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
    X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

    # We can now create a susceptibility model and generate data
    # Here a simple block in half-space
    model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
    offx, offy, offz = 0, 0, 4
    nx, ny, nz = 2, 2, 2

    model[(midx-offx-nx):(midx-offx+nx),
          (midy-offy-ny):(midy-offy+ny), -offz-nz:-offz+nz] = 0.02
    model = Utils.mkvc(model)
    model = model[actv]

    # Create data from the global problem
    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP=nC)
    prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv)

    # Pair the survey and problem
    survey.pair(prob)

    # Compute linear forward operator and compute some data
    d = prob.fields(model)

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    data = d + np.random.randn(len(d))
    wd = np.ones(len(data))*1.  # Assign flat uncertainties

    survey.dobs = data
    survey.std = wd

    # Estimate the size for comparison
    fullSize = prob.G.shape[0] * prob.G.shape[1] * 32  # bytes

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # LOOP THROUGH TILES
    expf = 1.3
    dx = [mesh.hx.min(), mesh.hy.min()]
    surveyMask = np.ones(survey.nD, dtype='bool')
    # Going through all problems:
    # 1- Pair the survey and problem
    # 2- Add up sensitivity weights
    # 3- Add to the ComboMisfit
    wrGlobal = np.zeros(nC)
    probSize = 0
    for tt in range(X1.shape[0]):

        # Grab the data for current tile
        ind_t = np.all([xyzLoc[:, 0] >= X1[tt], xyzLoc[:, 0] <= X2[tt],
                        xyzLoc[:, 1] >= Y1[tt], xyzLoc[:, 1] <= Y2[tt],
                        surveyMask], axis=0)

        # Remember selected data in case of tile overlap
        surveyMask[ind_t] = False

        # Create new survey
        xyzLoc_t = PF.BaseMag.RxObs(xyzLoc[ind_t, :])
        srcField = PF.BaseMag.SrcField([xyzLoc_t], param=survey.srcField.param)
        survey_t = PF.BaseMag.LinearSurvey(srcField)
        survey_t.dobs = survey.dobs[ind_t]
        survey_t.std = survey.std[ind_t]
        survey_t.ind = ind_t

        if ind_t.sum() == 0:
            continue

        padDist = np.r_[np.c_[50, 50], np.c_[50, 50], np.c_[50, 0]]
        mesh_t = Utils.modelutils.meshBuilder(xyzLoc[ind_t, :],
                                              np.r_[dx, dx, dx],
                                              padDist, meshGlobal=mesh)

        # Extract model from global to local mesh
        actv_t = Utils.surface2ind_topo(mesh_t, topo, 'N')

        # Creat reduced identity map
        tileMap = Maps.Tile((mesh, actv), (mesh_t, actv_t), tree=tree)

        # Create the forward model operator
        prob = PF.Magnetics.MagneticIntegral(mesh_t, chiMap=tileMap, actInd=actv_t)
        survey_t.pair(prob)

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey_t)
        dmis.W = 1./survey_t.std

        wr = np.sum(prob.G**2., axis=0)

        wrGlobal += prob.chiMap.deriv(0).T*wr

        # Create combo misfit function
        if tt == 0:
            ComboMisfit = dmis

        else:
            ComboMisfit += dmis

        # Add problem size
        probSize += prob.G.shape[0] * prob.G.shape[1] * 32

    # Scale global weights for regularization
    wrGlobal = wrGlobal**0.5
    wrGlobal = (wrGlobal/np.max(wrGlobal))

    # Create a regularization
    idenMap = Maps.IdentityMap(nP=nC)
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.norms = [0, 1, 1, 1]
    reg.eps_p = 1e-3
    reg.eps_q = 1e-3

    reg.cell_weights = wrGlobal
    reg.mref = np.zeros(mesh.nC)[actv]

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=30, lower=0., upper=10.,
                                     maxIterLS=20, maxIterCG=10, tolCG=1e-4)
    invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    # Use pick a treshold parameter empirically based on the distribution of
    #  model parameters
    IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3,
                                  maxIRLSiter=10)

    IRLS.target = survey.nD
    update_Jacobi = Directives.UpdateJacobiPrecond()

    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, IRLS, update_Jacobi])

    # Run the inversion
    m0 = np.ones(mesh.nC)[actv]*1e-4  # Starting model
    mrec = inv.run(m0)

    print('Size of all sub problems:' + str(np.round(probSize*1e-6)) + ' Mb')
    print('Size of global problems:' + str(np.round(fullSize*1e-6)) + ' Mb')

    if plotIt:
        # Here is the recovered susceptibility model
        ypanel = midy
        zpanel = -offz
        m_l2 = actvMap * IRLS.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp = actvMap * mrec
        m_lp[m_lp == -100] = np.nan

        m_true = actvMap * model
        m_true[m_true == -100] = np.nan

        # Plot the data
        fig, ax1 = plt.figure(), plt.subplot()
        PF.Magnetics.plot_obs_2D(xyzLoc, d=data, ax=ax1, fig=fig)
        for ii in range(X1.shape[0]):
            ax1.add_patch(Rectangle((X1[ii], Y1[ii]),
                                    X2[ii]-X1[ii],
                                    Y2[ii]-Y1[ii],
                                    facecolor='none', edgecolor='k'))
        # PF.Gravity.plot_obs_2D(rxLoc, d=invProb.dpred)
        plt.figure(figsize=(6, 8))

        # Plot L2 model
        ax = plt.subplot(321)
        mesh.plotSlice(m_l2, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(0, 0.02))
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
                       grid=True, clim=(0, 0.02))
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
                       grid=True, clim=(0, 0.02))
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
                       grid=True, clim=(0, 0.02))
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
                       grid=True, clim=(0, 0.02))
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
                       grid=True, clim=(0, 0.02))
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
