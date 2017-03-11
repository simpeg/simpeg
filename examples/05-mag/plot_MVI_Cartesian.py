"""
PF: Magnetics Vector Inversion - Cartesian
==========================================

In this example, we invert for the 3-component magnetization vector
with the Cartesian formulation (MVI-C). The code is used to invert magnetic
data affected by remanent magnetization and makes no induced
assumption. The inverse problem is three times larger than the usual
susceptibility inversion and depends strongly on the regularization.
The algorithm builds upon the research done at UBC:

Lelievre, G.P., D.W. Oldenburg, 2009, A 3D total magnetization
inversion applicable when significant, complicated remance is present.
Geophysics, 74, no.3: 21-30

The steps are:

1- **SETUP**: Create a synthetic model and calculate TMI data. This will
 simulate the usual magnetic experiment.

2- **INVERSION**: Invert for the magnetization vector.

The MVI-C formulation is easy to solve and has been used by other commercial
 codes such as VOXI.
The MVI-C formulation suffers however being highly non-unique, resulting in an
overly complex and smooth solution. Please visit the MVI-Spherical page for a
neat improvement to this problem.

"""

import numpy as np
import scipy.sparse as sp
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
from SimPEG import mkvc


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
    rxObj = PF.BaseMag.RxObs(rxLoc)
    srcField = PF.BaseMag.SrcField([rxObj], param=(B[0], B[1], B[2]))
    survey = PF.BaseMag.LinearSurvey(srcField)

    # We can now create a susceptibility model and generate data
    # Here a simple block in half-space
    model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
    model[(midx-2):(midx+2), (midy-2):(midy+2), -6:-2] = 0.05
    model = Utils.mkvc(model)
    model = model[actv]

    # We create a magnetization model different than the inducing field
    # to simulate remanent magnetization. Let's do something simple,
    # reversely magnetized [45,90]
    M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * 45., np.ones(nC) * 90.)

    # Multiply the orientation with the effective susceptibility
    # and reshape as [mx,my,mz] vector
    m = mkvc(sp.diags(model, 0) * M)

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Create reduced identity map
    idenMap = Maps.IdentityMap(nP=3*nC)

    # Create wires to link the regularization to each model blocks
    wires = Maps.Wires(('prim', mesh.nC),
                       ('second', mesh.nC),
                       ('third', mesh.nC))

    # Create identity map
    idenMap = Maps.IdentityMap(nP=3*nC)

    # Create the forward model operator
    prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                         actInd=actv)
    # Pair the survey and problem
    survey.pair(prob)



    # Compute forward model some data
    d = prob.fields(m)

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    d_TMI = d + np.random.randn(len(d))*0.
    wd = np.ones(len(d_TMI))  # Assign flat uncertainties
    survey.dobs = d_TMI
    survey.std = wd

    # Create a static sensitivity weighting function
    wr = np.sum(prob.F**2., axis=0)**0.5
    wr = (wr/np.max(wr))

    # Create a regularization
    reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.prim)
    reg_p.cell_weights = wires.prim * wr

    reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.second)
    reg_s.cell_weights = wires.second * wr

    reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.third)
    reg_t.cell_weights = wires.third * wr

    reg = reg_p + reg_s + reg_t

    reg.mref = np.zeros(3*nC)

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./survey.std

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=10, lower=-10., upper=10.,
                                     maxIterCG=20, tolCG=1e-3)

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                                  minGNiter=3, beta_tol=1e-2)

    update_Jacobi = Directives.Update_lin_PreCond()

    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[update_Jacobi, IRLS, betaest])

    mstart = np.ones(3*nC)*1e-4
    mrec = inv.run(mstart)

    if plotIt:
        # Here is the recovered susceptibility model
        ypanel = midx
        zpanel = -4

        vmin = model.min()
        vmax = model.max()/10.

        m_lpx = actvMap * mrec[0:nC]
        m_lpy = actvMap * mrec[nC:2*nC]
        m_lpz = actvMap * -mrec[2*nC:]

        m_lpx[m_lpx == -100] = np.nan
        m_lpy[m_lpy == -100] = np.nan
        m_lpz[m_lpz == -100] = np.nan

        amp = np.sqrt(m_lpx**2. + m_lpy**2. + m_lpz**2.)

        m_lpx = (m_lpx/amp).reshape(mesh.vnC, order='F')
        m_lpy = (m_lpy/amp).reshape(mesh.vnC, order='F')
        m_lpz = (m_lpz/amp).reshape(mesh.vnC, order='F')
        amp = amp.reshape(mesh.vnC, order='F')

        sub = 2

        xx = mesh.gridCC[:, 0].reshape(mesh.vnC, order="F")
        zz = mesh.gridCC[:, 2].reshape(mesh.vnC, order="F")
        yy = mesh.gridCC[:, 1].reshape(mesh.vnC, order="F")

        fig = plt.figure(figsize=(8, 8))
        ax2 = plt.subplot(212)
        im2 = ax2.contourf(xx[:, ypanel, :].T, zz[:, ypanel, :].T,
                           amp[:, ypanel, :].T, 40,
                           vmin=vmin, vmax=vmax, clim=[vmin, vmax],
                           cmap='magma_r')

        ax2.quiver(mkvc(xx[::sub, ypanel, ::sub].T),
                   mkvc(zz[::sub, ypanel, ::sub].T),
                   mkvc(m_lpx[::sub, ypanel, ::sub].T),
                   mkvc(m_lpz[::sub, ypanel, ::sub].T), pivot='mid',
                   units="xy", scale=0.2, linewidths=(1,), edgecolors=('k'),
                   headaxislength=0.1, headwidth=10, headlength=30)
        plt.colorbar(im2, orientation="vertical", ax=ax2,
                     ticks=np.linspace(im2.vmin, im2.vmax, 4),
                     format="${%.3f}$")

        ax2.set_aspect('equal')
        ax2.set_xlim(-50, 50)
        ax2.set_ylim(mesh.vectorNz[3], mesh.vectorNz[-1]+dx)
        ax2.set_title('EW - Recovered')
        ax2.set_xlabel('Easting (m)', size=14)
        ax2.set_ylabel('Elevation (m)', size=14)

        # plot true model
        vmin = model.min()
        vmax = model.max()*1.5

        m_lpx = (M[:, 0]).reshape(mesh.vnC, order='F')
        m_lpy = (M[:, 1]).reshape(mesh.vnC, order='F')
        m_lpz = -(M[:, 2]).reshape(mesh.vnC, order='F')
        amp = model.reshape(mesh.vnC, order='F')

        ax3 = plt.subplot(211)
        im2 = ax3.contourf(xx[:, ypanel, :].T, zz[:, ypanel, :].T,
                           amp[:, ypanel, :].T, 40,
                           vmin=vmin, vmax=vmax, clim=[vmin, vmax],
                           cmap='magma_r')

        ind = mkvc(amp[::sub, ypanel, ::sub].T) > 0
        ax3.quiver(mkvc(xx[::sub, ypanel, ::sub].T)[ind],
                   mkvc(zz[::sub, ypanel, ::sub].T)[ind],
                   mkvc(m_lpx[::sub, ypanel, ::sub].T)[ind],
                   mkvc(m_lpz[::sub, ypanel, ::sub].T)[ind], pivot='mid',
                   units="xy", scale=0.2, linewidths=(1,), edgecolors=('k'),
                   headaxislength=0.1, headwidth=10, headlength=30)
        plt.colorbar(im2, orientation="vertical", ax=ax3,
                     ticks=np.linspace(im2.vmin, im2.vmax, 4),
                     format="${%.3f}$")
        ax3.set_aspect('equal')
        ax3.set_xlim(-50, 50)
        ax3.set_ylim(mesh.vectorNz[3], mesh.vectorNz[-1]+dx)
        ax3.set_title('EW - True')
        ax3.xaxis.set_visible(False)
        ax3.set_ylabel('Elevation (m)', size=14)

        # Plot the data
        fig = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        PF.Magnetics.plot_obs_2D(rxLoc, d=d_TMI, fig=fig, ax=ax1,
                                 title='TMI Data')
        PF.Magnetics.plot_obs_2D(rxLoc, d=invProb.dpred, fig=fig, ax=ax2,
                                 title='Predicted Data')
if __name__ == '__main__':
    run()
    plt.show()
