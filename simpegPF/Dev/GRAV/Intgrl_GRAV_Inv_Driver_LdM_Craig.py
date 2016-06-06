#%%
from SimPEG import *
import simpegPF as PF
import pylab as plt
import os
import numpy as np


#home_dir = 'C:\Egnyte\Private\craigm\PHD\LdM\Gravity\Bouguer\SIMPEG\models\\all_models\\density_-1.2_0.3'
home_dir = '.\\'

inpfile = 'PYGRAV3D_inv.inp'
#inpfile = 'PYGRAV3D_inv_checkerboard.inp'

dsep = '\\'
os.chdir(home_dir)
plt.close('all')

#%% User input
# Initial beta
beta_in = 1e-2

# Plotting parameter
vmin = -0.1
vmax = 0.5

#weight exponent for default weighting
wgtexp = 3.  #dont forget the "."

#%%
driver = PF.GravityDriver.GravityDriver_Inv(home_dir + dsep + 'PYGRAV3D_inv.inp')
mesh = driver.mesh
survey = driver.survey

rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs
wd = survey.std

ndata = survey.srcField.rxList[0].locs.shape[0]

active = driver.activeCells
nC = len(active)

# Create active map to go from reduce set to full
activeMap = Maps.InjectActiveCells(mesh, active, -100)

# Create static map
static = driver.staticCells
dynamic = driver.dynamicCells

staticCells = Maps.InjectActiveCells(None, dynamic, driver.m0[static], nC=nC)
mstart = driver.m0[dynamic]

# Create reduced identity map
#idenMap = Maps.IdentityMap(nP=nC)

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)


#%% Plot obs data
#PF.Gravity.plot_obs_2D(survey,'Observed Data')

#%% Run inversion
prob = PF.Gravity.GravityIntegral(mesh, mapping = staticCells, actInd = active)
prob.solverOpts['accuracyTol'] = 1e-4

survey.pair(prob)

# Write out the predicted file and generate the forward operator
pred = prob.fields(mstart)

# Load weighting  file
if driver.wgtfile == 'DEFAULT':
    wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, active, wgtexp, np.min(mesh.hx)/4.)
    wr = wr**2.
else:
    wr = Mesh.TensorMesh.readModelUBC(mesh, home_dir + dsep + wgtfile)
    wr = wr[active]
    wr = wr**2.



#%% Plot depth weighting
plt.figure()
ax = plt.subplot(211)
datwgt=mesh.plotSlice(activeMap*wr, ax = ax, normal = 'Y', ind=midx+1 ,clim = (-1e-1, wr.max()), pcolorOpts={'cmap':'jet'})
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(datwgt[0],orientation="vertical")
cb.set_label('Weighting')

ax=plt.subplot(212)
plt.hist(wr,bins=100)
plt.yscale('log', nonposy='clip')
plt.title('Distribution of weights')
plt.tight_layout()
plt.savefig(home_dir + dsep + 'Weighting_' +str(wgtexp) +'.png', dpi=300)

#%% Create inversion objects

reg = Regularization.Sparse(mesh, indActive = active, mapping = staticCells)
reg.mref = driver.mref[dynamic]
reg.cell_weights = wr*mesh.vol[active]
eps_p = driver.eps[0]
eps_q = driver.eps[1]

opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=-2.,upper=2., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
#beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
#update_beta = Directives.Scale_Beta(tol = 0.05, coolingRate=5)
betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS( norms=driver.lpnorms,  eps=driver.eps, f_min_change = 1e-2)
update_Jacobi = Directives.Update_lin_PreCond()
inv = Inversion.BaseInversion(invProb, directiveList=[IRLS,betaest,update_Jacobi])

# Run inversion
mrec = inv.run(mstart)

m_out = activeMap*staticCells*reg.l2model


# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2_' +str(wgtexp) + '.den', m_out)
#Utils.meshutils.writeUBCTensorModel(home_dir+dsep+'wr.dat',mesh,wr_out)



pred_compact = prob.fields(mrec)
PF.Gravity.writeUBCobs(home_dir + dsep + 'Pred_compact' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.dat',survey,pred_compact)

pred = prob.fields(mrec)
# Plot predicted
#pred_smooth = prob.fields(mrec)
#PF.Gravity.writeUBCobs(home_dir + dsep + 'Pred_smooth' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.dat',survey,pred_smooth)
#PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
#PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')

print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) )
print "Misfit sum(obs-calc)/nobs: %.3f mGal"  %np.divide(np.sum(np.abs(d-pred)), len(d))
print "RMS misfit: %.3f mGal" %np.sqrt(np.divide(np.sum((d-pred)**2),len(d)))

#%% Plot out sections of the smooth model

yslice = midx+1
m_out[m_out==-100]=np.nan # set "air" to nan

print "\nMax density:" + str(np.nanmax(m_out))
print "\nMin density:" + str(np.nanmin(m_out))

plt.figure(figsize=(15,10))
plt.suptitle('Smooth Inversion: Depth weight = ' + str(wgtexp))
ax = plt.subplot(221)
dat1=mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-10, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='gray',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='k',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-10]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (g/cc$^3$)')

ax = plt.subplot(222)
dat = mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-13, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='gray',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='k',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-13]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (g/cc$^3$)')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.title('Cross Section')
plt.xlabel('Easting(m)');plt.ylabel('Elevation')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4), cmap='bwr')
cb.set_label('Density (g/cc$^3$)')
plt.savefig(home_dir + str('\Figure1_' +str(wgtexp) + '.png'), dpi=300, bb_inches='tight')

#plot histograms
plt.figure(figsize=(15,10))
ax = plt.subplot(121)
plt.hist(reg.l2model,100)
plt.yscale('log', nonposy='clip')
plt.xlim(reg.l2model.mean() - 6.*(reg.l2model.std()), reg.l2model.mean() + 6.*(reg.l2model.std()))
plt.xlabel('Density (g/cc$^3$)')
plt.title('Histogram of model values - Smooth')

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*(staticCells*reg.l2model),100)
plt.yscale('log', nonposy='clip')
plt.xlim(reg.l2model.mean() - 2.*(reg.l2model.std()), reg.l2model.mean() + 2.*(reg.l2model.std()))
plt.xlabel('Density (g/cc$^3$)')
plt.title('Histogram of model gradient values - Smooth')
plt.savefig(home_dir + str('\Figure2_' +str(wgtexp) + '.png'), dpi=300, bb_inches='tight')



#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,pred_compact,'Predicted Data', vmin = np.min(d), vmax = np.max(d))
plt.savefig(home_dir + str('\Figure3_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.png'), dpi=300, bb_inches='tight')
PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
plt.savefig(home_dir + str('\Figure4.png'), dpi=300, bb_inches='tight')
print "\nFinal misfit:" + str(np.sum( ((d-pred_compact)/wd)**2. ) )
print "Misfit sum(obs-calc)/nobs: %.3f mGal"  %np.divide(np.sum(np.abs(d-pred_compact)), len(d))
print "RMS misfit: %.3f mGal" %np.sqrt(np.divide(np.sum((d-pred_compact)**2),len(d)))



#%% Plot out a section of the compact model
m_out = activeMap*staticCells*mrec

Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l0l2_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.den',m_out)

yslice = midx+1
m_out[m_out==-100]=np.nan # set "air" to nan

print "\nMax density:" + str(np.nanmax(m_out))
print "\nMin density:" + str(np.nanmin(m_out))

plt.figure(figsize=(15,10))
plt.suptitle('Compact Inversion: Depth weight = ' + str(wgtexp) + ': $\epsilon_p$ = ' + str(eps_p) + ': $\epsilon_q$ = ' + str(eps_q))
ax = plt.subplot(221)
dat = mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-10, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='gray',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='k',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-10]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (g/cc$^3$)')

ax = plt.subplot(222)
dat = mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-13, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='gray',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='k',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-13]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (g/cc$^3$)')

ax = plt.subplot(212)
dat = mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.title('Cross Section')
plt.xlabel('Easting (m)');plt.ylabel('Elevation (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (g/cc$^3$)')

plt.savefig(home_dir + str('\Figure5_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.png'), dpi=300, bb_inches='tight')

#plot histograms
plt.figure(figsize=(15,10))
ax = plt.subplot(121)
plt.hist(mrec,100)
#plt.xlim(mrec.mean() - 6.*(mrec.std()), mrec.mean() + 6.*(mrec.std()))
plt.yscale('log', nonposy='clip')
plt.xlabel('Density (g/cc$^3$)')
plt.title('Histogram of model values - Sparse lp:'+str(driver.lpnorms[0]))

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*(staticCells*mrec),100)

#plt.xlim(mrec.mean() - 4.*(mrec.std()), mrec.mean() + 4.*(mrec.std()))
plt.xlabel('Density (g/cc$^3$)')
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model gradient values - Sparse lqx: ' + str(driver.lpnorms[1]) + ' lqy:'+ str(driver.lpnorms[2]) + ' lqz:' + str(driver.lpnorms[3]))
plt.savefig(home_dir + str('\Figure6_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.png'), dpi=300, bb_inches='tight')

#make a plot of the obs -calc, ie residual
plt.figure(figsize=(10,8))
residual = d - pred_compact
plt.hist(residual, 100)
plt.savefig(home_dir + str('\Figure7_residuals.png'), dpi=300, bb_inches='tight')
