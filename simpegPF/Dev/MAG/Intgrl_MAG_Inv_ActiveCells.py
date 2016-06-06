from SimPEG import *
import simpegPF as PF
import pylab as plt

import os

driver = PF.MagneticsDriver.MagneticsDriver_Inv('PYMAG3D_inv.inp')
mesh = driver.mesh
survey = driver.survey

rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs
wd = survey.std

ndata = survey.srcField.rxList[0].locs.shape[0]

eps_p = 1e-4
eps_q = 1e-4

actv = driver.activeCells
nC = len(actv)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Get magnetization vector for MOF
M_xyz = driver.magnetizationModel

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)+1
midz = int(mesh.nCz/2)

vmin = 0
vmax = 1e-3
#%% Run inversion
prob = PF.Magnetics.Problem3D_Integral(mesh, mapping=idenMap, actInd=actv)
prob.solverOpts['accuracyTol'] = 1e-4
survey.pair(prob)

# Write out the predicted
pred = prob.fields(driver.m0)
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

reg = Regularization.Simple(mesh, indActive=actv, mapping=idenMap)
reg.mref = driver.mref
reg.cell_weights = wr

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/wd
opt = Optimization.ProjectedGNCG(maxIter=10,lower=0.,upper=1., maxIterCG= 10, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
beta_init = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
update_Jacobi = Directives.Update_lin_PreCond(onlyOnStart=True)

inv = Inversion.BaseInversion(invProb, directiveList=[beta,target,beta_init,update_Jacobi])

# Run inversion
mrec = inv.run(driver.m0)

m_out = actvMap*mrec

# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2.sus',m_out)

# Plot predicted
pred = prob.fields(mrec)
#PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
#PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')

print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) )

#%% Plot out a section of the model

yslice = midx
m_out[m_out==-100] = np.nan

plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')


ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax))
plt.title('Smooth Unconstrained')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

#%% Re-run inversion using a starting model with
# static cells
m0 = np.ones(mesh.nC)*1e-4
val = 0.002

# Reshape the model in order to create a static block
m0 = np.reshape(m0,(mesh.nCx,mesh.nCy,mesh.nCz), order = 'F')
m0[midx-6,midy,midz+2] = val
m0[midx+7,midy,midz+2] = val
m0 = mkvc(m0)
# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'ModelStart.sus',m0)

# Extract cells under topography and create new index for inactive
m0 = m0[actv]
ind_act = m0!=val

actvCells = Maps.InjectActiveCells(None, ind_act, val, nC=nC)
m0 = m0[ind_act]

# Change the mapping of the problem and run inversion
prob.mapping = actvCells

# Write out the predicted
pred = prob.fields(m0)
PF.Magnetics.writeUBCobs('Pred.dat', survey, pred)

reg = Regularization.Simple(mesh, indActive=actv, mapping=actvCells)
reg.mref = driver.mref[ind_act]
reg.cell_weights = wr

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/wd
opt = Optimization.ProjectedGNCG(maxIter=10,lower=0.,upper=1., maxIterCG= 10, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
beta_init = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
update_Jacobi = Directives.Update_lin_PreCond(onlyOnStart=True)

inv = Inversion.BaseInversion(invProb, directiveList=[beta,target,beta_init,update_Jacobi])

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*actvCells*mrec

# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2_Constrained.sus',m_out)

# Plot predicted
pred = prob.fields(mrec)
#PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data - l2 Inversion')
#PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')

print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) )

#%% Plot out a section of the model

yslice = midx
m_out[m_out==-100] = np.nan

plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax))
plt.title('Smooth Constrained')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

#%% Run one more round for sparsity
phim = invProb.phi_m_last
phid =  invProb.phi_d

# Set parameters for sparsity
reg = Regularization.Sparse(mesh, indActive = actv, mapping=actvCells)
reg.curModel = mrec
reg.mref = driver.mref[ind_act]
reg.cell_weights = wr
reg.eps_p = eps_p
reg.eps_q = eps_q
reg.norms   = driver.lpnorms


dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd
opt = Optimization.ProjectedGNCG(maxIter=10 , lower=0.,upper=1., maxIterCG= 20, tolCG = 1e-4)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = invProb.beta)

# Create inversion directives
beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
update_beta = Directives.Scale_Beta(tol = 0.05)
target = Directives.TargetMisfit()
IRLS =Directives.Update_IRLS( phi_m_last = phim )
update_Jacobi = Directives.Update_lin_PreCond(onlyOnStart=False)
save_log = Directives.SaveOutputEveryIteration()
save_log.fileName = 'LogName_blabla'

inv = Inversion.BaseInversion(invProb, directiveList=[beta,IRLS,update_beta,update_Jacobi,save_log])

m0 = mrec

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*actvCells*mrec

# Write final model out.
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l0l2_Constrained.sus',m_out)

pred = prob.fields(mrec)

#%% Plot obs data
#PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data')
#PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) )
#%% Plot out a section of the model

yslice = midy

m_out[m_out==-100] = np.nan

plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (mrec.min(),mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (mrec.min(),mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (mrec.min(),mrec.max()))
plt.title('Compact Constrained')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')


plt.show()