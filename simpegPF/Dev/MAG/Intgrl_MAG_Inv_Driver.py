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

beta_in = 1e+5
eps_p = 1e-4
eps_q = 1e-4

actv = driver.activeCells
nC = len(actv)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP = nC)

# Get magnetization vector for MOF
M_xyz = driver.magnetizationModel

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

# Get distance weighting function
#==============================================================================
# wr = PF.Magnetics.get_dist_wgt(mesh,rxLoc,actv,3.,np.min(mesh.hx)/4)
# #wrMap = PF.BaseMag.WeightMap(nC, wr)
#==============================================================================

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,d, 'Observed Data')

#%% Run inversion
prob = PF.Magnetics.Problem3D_Integral(mesh, mapping=idenMap, actInd=actv)
prob.solverOpts['accuracyTol'] = 1e-4
survey.pair(prob)
#survey.makeSyntheticData(data, std=0.01)
#survey.dobs=d
#survey.mtrue = model
# Write out the predicted
pred = prob.fields(driver.m0)
PF.Magnetics.writeUBCobs('Pred.dat', survey, pred)

wr = np.sum(prob.G**2.,axis=0)**0.5 / mesh.vol[actv]
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
reg.wght = wr
#reg.alpha_s = 1.

# Create pre-conditioner
diagA  = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
PC     = Utils.sdiag(diagA**-1.)

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/wd
opt = Optimization.ProjectedGNCG(maxIter=10,lower=0.,upper=1., maxIterCG= 20, tolCG=1e-3)
opt.approxHinv = PC

# opt = Optimization.InexactGaussNewton(maxIter=6)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta_in)
beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
#betaest = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()

inv = Inversion.BaseInversion(invProb, directiveList=[beta,target])

# Run inversion
mrec = inv.run(driver.m0)

m_out = actvMap*mrec

# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2.sus',m_out)
#Utils.meshutils.writeUBCTensorModel(home_dir+dsep+'wr.dat',mesh,wr_out)

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

#%% Run one more round for sparsity
phim = invProb.phi_m_last
phid =  invProb.phi_d

reg = Regularization.Sparse(mesh, indActive = actv, mapping = idenMap)
reg.recModel = mrec
reg.mref = driver.mref
reg.wght = wr
reg.eps_p = eps_p
reg.eps_q = eps_q
reg.norms   = driver.lpnorms


diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
PC     = Utils.sdiag(diagA**-1.)

#reg.alpha_s = 1.

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = wd
opt = Optimization.ProjectedGNCG(maxIter=20 ,lower=0.,upper=1., maxIterCG= 10, tolCG = 1e-4)
opt.approxHinv = PC
#opt.phim_last = reg.eval(mrec)

# opt = Optimization.InexactGaussNewton(maxIter=6)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = invProb.beta)
beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
update_beta = Directives.Scale_Beta(tol = 0.05)
#betaest = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
IRLS =Directives.Update_IRLS( phi_m_last = phim, phi_d_last = phid )

inv = Inversion.BaseInversion(invProb, directiveList=[beta,IRLS,update_beta])

m0 = mrec

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*mrec

Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l0l2.sus',m_out)

pred = prob.fields(mrec)

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data')
PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) )
#%% Plot out a section of the model

yslice = midx

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


plt.show()
