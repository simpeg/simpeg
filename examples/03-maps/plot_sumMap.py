"""
Maps: ComboMaps
===============

Invert synthetic magnetic data with variable background values
and a single block anomaly buried at depth. We will use the Sum Map
to invert for both the background values and an heterogeneous susceptibiilty
model.

.. code-block:: python
    :linenos:


"""
from SimPEG import (
    Mesh, Utils, Maps, Regularization,
    DataMisfit, Optimization, InvProblem,
    Directives, Inversion, PF
)
import numpy as np
import matplotlib.pyplot as plt


def run(plotIt=True):

    H0 = (50000., 90., 0.)

    # Create a mesh
    dx = 5.

    hxind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 10)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.vectorNz[-1]

    # We would usually load a topofile
    topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

    # Go from topo to array of indices of active cells
    actv = Utils.surface2ind_topo(mesh, topo, 'N')
    actv = np.where(actv)[0]

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
    model = np.zeros(mesh.nC)

    # Change values in half the domain
    model[mesh.gridCC[:,0] < 0] = 0.01

    # Add a block in half-space
    model = Utils.ModelBuilder.addBlock(mesh.gridCC, model, np.r_[-10,-10,20], np.r_[10,10,40], 0.05)

    model = Utils.mkvc(model)
    model = model[actv]


    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, np.nan)

    # Create reduced identity map
    idenMap = Maps.IdentityMap(nP=len(actv))

    # Create the forward model operator
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
    survey.mtrue = model

    # Plot the data
    rxLoc = survey.srcField.rxList[0].locs

    # Create a homogenous maps for the two domains
    domains = [mesh.gridCC[actv,0] < 0, mesh.gridCC[actv,0] >= 0]
    homogMap = Maps.SurjectUnits(domains)

    # Create a wire map for a second model space, voxel based
    wires = Maps.Wires(('homo', len(domains)), ('hetero', len(actv)))

    # Create Sum map
    sumMap = Maps.SumMap([homogMap*wires.homo, wires.hetero])

    # Create the forward model operator
    prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=sumMap, actInd=actv)

    # Pair the survey and problem
    survey.unpair()
    survey.pair(prob)

    # Make depth weighting
    wr = np.zeros(sumMap.shape[1])

    # Take the cell number out of the scaling.
    # Want to keep high sens for large volumes
    scale = Utils.sdiag(np.r_[Utils.mkvc(1./homogMap.P.sum(axis=0)),np.ones_like(actv)])

    for ii in range(survey.nD):
        wr += ((prob.G[ii, :]*prob.chiMap.deriv(np.ones(sumMap.shape[1])*1e-4)*scale)/survey.std[ii])**2.

    # Scale the model spaces independently
    wr[wires.homo.index] /= (np.max((wires.homo*wr)))
    wr[wires.hetero.index] /= (np.max(wires.hetero*wr))
    wr = wr**0.5

    ## Create a regularization
    # For the homogeneous model
    regMesh = Mesh.TensorMesh([len(domains)])

    reg_m1 = Regularization.Sparse(regMesh, mapping=wires.homo)
    reg_m1.cell_weights = wires.homo*wr
    reg_m1.norms = np.c_[0, 2, 2, 2]
    reg_m1.mref = np.zeros(sumMap.shape[1])

    # Regularization for the voxel model
    reg_m2 = Regularization.Sparse(mesh, indActive=actv, mapping=wires.hetero)
    reg_m2.cell_weights = wires.hetero*wr
    reg_m2.norms = np.c_[0, 1, 1, 1]
    reg_m2.mref =  np.zeros(sumMap.shape[1])

    reg = reg_m1 + reg_m2

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1/wd

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                     maxIterLS=20, maxIterCG=10, tolCG=1e-3, tolG=1e-3, eps=1e-6)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    # Use pick a threshold parameter empirically based on the distribution of
    #  model parameters
    IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=1)
    update_Jacobi = Directives.UpdatePreconditioner()
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[IRLS, betaest, update_Jacobi])

    # Run the inversion
    m0 = np.ones(sumMap.shape[1])*1e-4  # Starting model
    prob.model = m0
    mrecSum = inv.run(m0)
    if plotIt:

        mesh.plot_3d_slicer(actvMap * model, aspect="equal", zslice=30, pcolorOpts={"cmap":'inferno_r'}, transparent='slider')

        mesh.plot_3d_slicer(actvMap * sumMap * mrecSum, aspect="equal", zslice=30, pcolorOpts={"cmap":'inferno_r'}, transparent='slider')



if __name__ == '__main__':
    run()
    plt.show()
