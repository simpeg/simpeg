#%%
from SimPEG import *
import simpegPF as PF
import pylab as plt
import os
import numpy as np



home_dir = '.\\'

inpfile = 'PYGRAV3D_inv.inp'

dsep = '\\'
os.chdir(home_dir)
plt.close('all')

#%% User input
# Initial beta
beta_in = 1e-2

# Treshold values for compact norm
eps_p = 0.2# Compact model values #refer to histograms to choose appropriate value
eps_q = 0.1 # Compact model gradient

# Plotting parameter
vmin = -0.5
vmax = 0.5

#weight exponent for default weighting
wgtexp = 3.  #dont forget the "."

#value of fixed cells

fixedcell = -0.5

#%%
driver = PF.GravityDriver.GravityDriver_Inv(home_dir + dsep + 'PYGRAV3D_inv.inp')
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

ind_act = driver.staticCells
staticCells = Maps.InjectActiveCells(None, ind_act, driver.m0[ind_act==False], nC=nC)
mstart = mstart[ind_act]


# Load reference file
if isinstance(mref, float):
    mref = np.ones(nC) * mref
else:
    mref = Mesh.TensorMesh.readModelUBC(mesh,mref)
    mref = mref[actv]

mref = mref[ind_act]
# Get index of the center for plotting
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)


#%% Plot obs data
#PF.Gravity.plot_obs_2D(survey,'Observed Data')

#%% Run inversion
prob = PF.Gravity.GravityIntegral(mesh, mapping = staticCells, actInd = actv)
prob.solverOpts['accuracyTol'] = 1e-4

survey.pair(prob)

# Write out the predicted file and generate the forward operator
pred_start = prob.fields(mstart)

#PF.Gravity.writeUBCobs(home_dir + dsep + 'Pred_start' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.dat',survey,pred_start)

# Make depth weighting
#wr = np.sum(prob.G**2.,axis=0)**0.5 / mesh.vol[actv]
#wr = ( wr/np.max(wr) )
#wr_out = actvMap * wr


#A different weighting function from Dominic
#wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 2., np.min(mesh.hx)/4.)
#wr = wr**2.


# Load weighting  file
if wgtfile is None:  
    wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, wgtexp, np.min(mesh.hx)/4.)
    wr = wr**2.
else:
    wr = Mesh.TensorMesh.readModelUBC(mesh, home_dir + dsep + wgtfile)   
    wr = wr[actv]
    wr = wr**2.
    


#%% Plot depth weighting
plt.figure()
ax = plt.subplot(211)
datwgt=mesh.plotSlice(actvMap*wr, ax = ax, normal = 'Y', ind=midx+1 ,clim = (-1e-1, wr.max()), pcolorOpts={'cmap':'jet'})
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

print '\nRun smooth inversion \n'
# First start with an l2 (smooth model) regularization
reg = Regularization.Simple(mesh, indActive = actv, mapping = staticCells)
reg.mref = mref
reg.wght = wr



# Create pre-conditioner # should be without the *wr at the end but works better with it
#diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
#PC     = Utils.sdiag(diagA**-1.)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd
opt = Optimization.ProjectedGNCG(maxIter=20,lower=bounds[0],upper=bounds[1], maxIterCG= 20, tolCG = 1e-3)
#opt.approxHinv = PC


invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = beta_in)
beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
update_beta = Directives.Scale_Beta(tol = 0.05) 
target = Directives.TargetMisfit()
update_Jacobi = Directives.Update_lin_PreCond(onlyOnStart=True)

save_log = Directives.SaveOutputEveryIteration()
save_log.fileName = home_dir + dsep + 'SimPEG_inv_l2l2_log_' +str(wgtexp)

inv = Inversion.BaseInversion(invProb, directiveList=[beta,target,update_beta,update_Jacobi,save_log])
#inv = Inversion.BaseInversion(invProb, directiveList=[beta,target])

m0 = mstart

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*staticCells*mrec

# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2_' +str(wgtexp) + '.den', m_out)
#Utils.meshutils.writeUBCTensorModel(home_dir+dsep+'wr.dat',mesh,wr_out)

# Plot predicted
pred_smooth = prob.fields(mrec)
PF.Gravity.writeUBCobs(home_dir + dsep + 'Pred_smooth' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.dat',survey,pred_smooth)
#PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
#PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')

print "Final misfit:" + str(np.sum( ((d-pred_smooth)/wd)**2. ) ) 
print "Misfit sum(obs-calc)/nobs: %.3f mGal"  %np.divide(np.sum(np.abs(d-pred_smooth)), len(d))
print "RMS misfit: %.3f mGal" %np.sqrt(np.divide(np.sum((d-pred_smooth)**2),len(d)))

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
dat = mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-25, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='gray',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='k',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-25]) + ' m')
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
plt.hist(mrec,100)
plt.yscale('log', nonposy='clip')
plt.xlim(mrec.mean() - 6.*(mrec.std()), mrec.mean() + 6.*(mrec.std()))
plt.xlabel('Density (g/cc$^3$)')
plt.title('Histogram of model values - Smooth')

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*(staticCells*mrec),100)
plt.yscale('log', nonposy='clip')
plt.xlim(mrec.mean() - 2.*(mrec.std()), mrec.mean() + 2.*(mrec.std()))
plt.xlabel('Density (g/cc$^3$)')
plt.title('Histogram of model gradient values - Smooth')
plt.savefig(home_dir + str('\Figure2_' +str(wgtexp) + '.png'), dpi=300, bb_inches='tight')



#%% Run one more round for sparsity (Compact model)
print '\nRun compact inversion \n'
phim = invProb.phi_m_last
phid =  invProb.phi_d

reg = Regularization.Sparse(mesh, indActive = actv, mapping = staticCells)
reg.recModel = mrec
reg.mref = mref
reg.wght = wr
reg.eps_p = eps_p
reg.eps_q = eps_q
reg.norms   = lpnorms

#diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
#PC     = Utils.sdiag(diagA**-1.)
update_Jacobi = Directives.Update_lin_PreCond()

#reg.alpha_s = 1.

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd
opt = Optimization.ProjectedGNCG(maxIterLS=20, maxIter=20 ,lower=bounds[0],upper=bounds[1], maxIterCG= 50, tolCG = 1e-4)
#opt.approxHinv = PC
#opt.phim_last = reg.eval(mrec)

# opt = Optimization.InexactGaussNewton(maxIter=6)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = invProb.beta)
beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)

#update beta only if misfit is outside the tolerance
update_beta = Directives.Scale_Beta(tol = 0.05) #Tolerance value is in % of the target

#betaest = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
IRLS =Directives.Update_IRLS( phi_m_last = phim, phi_d_last = phid )

#save output to logfile
save_log = Directives.SaveOutputEveryIteration()
save_log.fileName = home_dir + dsep + 'SimPEG_inv_l0l2_log_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q)


inv = Inversion.BaseInversion(invProb, directiveList=[beta,IRLS,update_beta,update_Jacobi,save_log])



m0 = mrec

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*staticCells*mrec

Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l0l2_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.den',m_out)

pred_compact = prob.fields(mrec)
PF.Gravity.writeUBCobs(home_dir + dsep + 'Pred_compact' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.dat',survey,pred_compact)



#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,pred_compact,'Predicted Data', vmin = np.min(d), vmax = np.max(d))
plt.savefig(home_dir + str('\Figure3_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.png'), dpi=300, bb_inches='tight')
PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
plt.savefig(home_dir + str('\Figure4.png'), dpi=300, bb_inches='tight')
print "\nFinal misfit:" + str(np.sum( ((d-pred_compact)/wd)**2. ) ) 
print "Misfit sum(obs-calc)/nobs: %.3f mGal"  %np.divide(np.sum(np.abs(d-pred_compact)), len(d))
print "RMS misfit: %.3f mGal" %np.sqrt(np.divide(np.sum((d-pred_compact)**2),len(d)))



#%% Plot out a section of the compact model

yslice = midx
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
dat = mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-25, clim = (vmin,vmax), pcolorOpts={'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='gray',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='k',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-25]) + ' m')
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
plt.title('Histogram of model values - Sparse lp:'+str(lpnorms[0]))

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*(staticCells*mrec),100)
#plt.xlim(mrec.mean() - 4.*(mrec.std()), mrec.mean() + 4.*(mrec.std()))
plt.xlabel('Density (g/cc$^3$)')
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model gradient values - Sparse lqx: ' + str(lpnorms[1]) + ' lqy:'+ str(lpnorms[2]) + ' lqz:' + str(lpnorms[3]))
plt.savefig(home_dir + str('\Figure6_' +str(wgtexp) + '_' + str(eps_p) + '_' + str(eps_q) +'.png'), dpi=300, bb_inches='tight')