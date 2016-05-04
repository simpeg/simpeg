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
midy = int(mesh.nCy/2)
midz = int(mesh.nCz/2)

#%% Test script to impose static cells
m0 = np.ones(mesh.nC)*1e-3

# Reshape the model in order to create a static block
m0 = np.reshape(m0,(mesh.nCx,mesh.nCy,mesh.nCz), order = 'F')
m0[midx,midy,midz] = 0.5
m0 = mkvc(m0)

# Extract cells under topography and create new index for inactive
m0 = m0[actv]
ind_act = m0!=0.5

actvCells = Maps.InjectActiveCells(None, ind_act, 0.5, nC=nC)
m0 = m0[ind_act]

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,d, 'Observed Data')

#%% Run inversion
prob = PF.Magnetics.Problem3D_Integral(mesh, mapping=actvCells, actInd=actv)
prob.solverOpts['accuracyTol'] = 1e-4
survey.pair(prob)

# Write out the predicted
pred = prob.fields(m0)
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

reg = Regularization.Simple(mesh, indActive=actv, mapping=actvCells)
#reg.mref = m0*0
reg.wght = wr

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/wd
opt = Optimization.ProjectedGNCG(maxIter=10,lower=0.,upper=1., maxIterCG= 20, tolCG=1e-3)

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
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2.sus',m_out)

# Plot predicted
pred = prob.fields(mrec)
PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data - l2 Inversion')
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