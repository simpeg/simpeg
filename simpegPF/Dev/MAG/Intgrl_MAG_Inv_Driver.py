from SimPEG import *
import simpegPF as PF
import pylab as plt

#import os

plt.close('all')

driver = PF.MagneticsDriver.MagneticsDriver_Inv('PYMAG3D_inv.inp')
mesh = driver.mesh
survey = driver.survey

rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs
wd = survey.std
    
ndata = survey.srcField.rxList[0].locs.shape[0]

actv = driver.activeCells
nC = len(actv)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create static map
static = driver.staticCells
dynamic = driver.dynamicCells

staticCells = Maps.InjectActiveCells(None, dynamic, driver.m0[static], nC=nC)
mstart = driver.m0[dynamic]

# Get magnetization vector for MOF
M_xyz = driver.magnetizationModel

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,d, 'Observed Data')

#%% Run inversion
prob = PF.Magnetics.Problem3D_Integral(mesh, mapping = staticCells, actInd=actv)
prob.solverOpts['accuracyTol'] = 1e-4
survey.pair(prob)

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd

# Write out the predicted
pred = prob.fields(mstart)
PF.Magnetics.writeUBCobs('Pred.dat', survey, pred)

wr = np.sum(prob.G**2.,axis=0)**0.5
wr = ( wr/np.max(wr) )
wr_out = actvMap * wr

plt.figure()
ax = plt.subplot()
mesh.plotSlice(wr_out, ax=ax, normal='Y', ind=midx ,clim=(-1e-3, wr.max()))
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')
reg = Regularization.Sparse(mesh, indActive=actv, mapping = staticCells)
reg.mref = driver.mref[dynamic]
reg.cell_weights = wr
    
#reg.mref = np.zeros(mesh.nC)
#eps_p = 5e-5
#eps_q = 5e-5
#norms   = [0., 1., 1., 1.]

opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=driver.bounds[0],upper=driver.bounds[1], maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
#beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
#update_beta = Directives.Scale_Beta(tol = 0.05, coolingRate=5)
betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS( norms=driver.lpnorms,  eps=driver.eps, f_min_change = 1e-3, minGNiter=6)
update_Jacobi = Directives.Update_lin_PreCond()
inv = Inversion.BaseInversion(invProb, directiveList=[IRLS,betaest,update_Jacobi])

# Run inversion
mrec = inv.run(mstart)

pred = prob.fields(mrec)

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data')
PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) )

#%% Plot out a section of the model

yslice = midx


m_out = actvMap*staticCells*reg.l2model
m_out[m_out==-100] = np.nan

# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2.sus',m_out)

plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (mrec.min(), mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (mrec.min(), mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')


ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (mrec.min(), mrec.max()))
plt.title('Cross Section')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

plt.figure()
ax = plt.subplot(121)
plt.hist(reg.l2model,100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model values - Smooth')
ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*(staticCells*reg.l2model),100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model gradient values - Smooth')

#%% Plot out a section of the model

yslice = midx+1

m_out = actvMap*staticCells*mrec
m_out[m_out==-100] = np.nan

plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (mrec.min(), mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (mrec.min(), mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (mrec.min(), mrec.max()))
plt.title('Cross Section')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

plt.figure()
ax = plt.subplot(121)
plt.hist(mrec,100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model values - Compact')
ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*(staticCells*mrec),100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model gradient values - Smooth')

plt.show()
