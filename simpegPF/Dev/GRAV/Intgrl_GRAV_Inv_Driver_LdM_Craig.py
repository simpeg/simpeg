#%%
from SimPEG import *
import simpegPF as PF
import pylab as plt
import os
import numpy as np

#home_dir = 'C:\Egnyte\Private\craigm\PHD\LdM\Gravity\Bouguer\SIMPEG\models'
home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev\GRAV'
#inpfile = 'PYGRAV3D_inv_LdM_Craig.inp'
inpfile = 'PYGRAV3D_inv_checkerboard.inp'
dsep = '\\'
os.chdir(home_dir)
plt.close('all')

#%% User input
# Initial beta
beta_in = 1e-2

# Treshold values for compact norm
eps_p = 0.01 # Compact model values
eps_q = 0.01 # ompact model gradient

# Plotting parameter
vmin = -0.1
vmax = 0.2
#%%
# Read input file
[mshfile, obsfile, topofile, mstart, mref, wgtfile, chi, alphas, bounds, lpnorms] = PF.Gravity.read_GRAVinv_inp(home_dir + dsep + inpfile)

# Load mesh file
mesh = Mesh.TensorMesh.readUBC(mshfile)

# Load in observation file
survey = PF.Gravity.readUBCgravObs(obsfile)

# Get obs location and data
rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs
wd = survey.std

ndata = survey.srcField.rxList[0].locs.shape[0]

# Load in topofile or create flat surface
if topofile == 'null':
    
    # All active
    actv = np.asarray(range(mesh.nC))
    
else: 
    
    topo = np.genfromtxt(topofile,skip_header=1)
    # Find the active cells
    actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')

nC = len(actv)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP = nC)

# Load starting model file
if isinstance(mstart, float):
    
    mstart = np.ones(nC) * mstart
else:
    mstart = Mesh.TensorMesh.readModelUBC(mesh,mstart)
    mstart = mstart[actv]

# Load reference file
if isinstance(mref, float):
    mref = np.ones(nC) * mref
else:
    mref = Mesh.TensorMesh.readModelUBC(mesh,mref)
    mref = mref[actv]
    
# Get index of the center for plotting
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)


#%% Plot obs data
#PF.Gravity.plot_obs_2D(survey,'Observed Data')

#%% Run inversion
prob = PF.Gravity.GravityIntegral(mesh, mapping = idenMap, actInd = actv)
prob.solverOpts['accuracyTol'] = 1e-4

survey.pair(prob)

# Write out the predicted file and generate the forward operator
pred = prob.fields(mstart)

PF.Gravity.writeUBCobs(home_dir + dsep + 'Pred0.dat',survey,pred)

# Make depth weighting
#wr = np.sum(prob.G**2.,axis=0)**0.5 / mesh.vol[actv]
#wr = ( wr/np.max(wr) )
#wr_out = actvMap * wr


#A different weighting function from Dominic
#wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 2., np.min(mesh.hx)/4.)
#wr = wr**2.


# Load weighting  file
if wgtfile is None:  
    wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 2., np.min(mesh.hx)/4.)
    wr = wr**2.
else:
    wr = Mesh.TensorMesh.readModelUBC(mesh, home_dir + dsep + wgtfile)   
    wr = wr[actv]
    wr = wr**2.
    


#%% Plot depth weighting
plt.figure()
ax = plt.subplot()
datwgt=mesh.plotSlice(actvMap*wr, ax = ax, normal = 'Y', ind=midx+1 ,clim = (-1e-1, wr.max()))
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(datwgt[0],orientation="vertical")
cb.set_label('Weighting')
plt.savefig(home_dir + dsep + 'Weighting.png', dpi=300)

#%% Create inversion objects

print '\nRun smooth inversion \n'
# First start with an l2 (smooth model) regularization
reg = Regularization.Simple(mesh, indActive = actv, mapping = idenMap)
reg.mref = mref
reg.wght = wr



# Create pre-conditioner 
diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
PC     = Utils.sdiag(diagA**-1.)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd
opt = Optimization.ProjectedGNCG(maxIter=20,lower=bounds[0],upper=bounds[1], maxIterCG= 20, tolCG = 1e-3)
opt.approxHinv = PC


invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = beta_in)
beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
target = Directives.TargetMisfit()

inv = Inversion.BaseInversion(invProb, directiveList=[beta,target])

m0 = mstart

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*mrec

# Write result
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2.den',m_out)
#Utils.meshutils.writeUBCTensorModel(home_dir+dsep+'wr.dat',mesh,wr_out)

# Plot predicted
pred = prob.fields(mrec)
#PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
#PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')

print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) ) 

#%% Plot out sections of the smooth model

yslice = midx+1

plt.figure(figsize=(15,10))
plt.suptitle('Smooth Inversion')
ax = plt.subplot(221)
dat1=mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='w',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (kg/m3)')

ax = plt.subplot(222)
dat = mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='w',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (kg/m3)')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax))
plt.title('Cross Section')
plt.xlabel('Easting(m)');plt.ylabel('Elevation')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (kg/m3)')
plt.savefig(home_dir + str('\Figure1.png'), dpi=300, bb_inches='tight')

#plot histograms
plt.figure(figsize=(15,10))
ax = plt.subplot(121)
plt.hist(mrec,100)
plt.yscale('log', nonposy='clip')
plt.xlim(mrec.mean() - 4.*(mrec.std()), mrec.mean() + 4.*(mrec.std()))
plt.title('Histogram of model values - Smooth')

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*mrec,100)
plt.yscale('log', nonposy='clip')
plt.xlim(mrec.mean() - 4.*(mrec.std()), mrec.mean() + 4.*(mrec.std()))
plt.title('Histogram of model gradient values - Smooth')
plt.savefig(home_dir + str('\Figure2.png'), dpi=300, bb_inches='tight')
#%% Run one more round for sparsity (Compact model)
print '\nRun compact inversion \n'
phim = invProb.phi_m_last
phid =  invProb.phi_d

reg = Regularization.Sparse(mesh, indActive = actv, mapping = idenMap)
reg.recModel = mrec
reg.mref = mref
reg.wght = wr
reg.eps_p = eps_p
reg.eps_q = eps_q
reg.norms   = lpnorms

diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
PC     = Utils.sdiag(diagA**-1.)

#reg.alpha_s = 1.

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd
opt = Optimization.ProjectedGNCG(maxIter=10 ,lower=bounds[0],upper=bounds[1], maxIterCG= 25, tolCG = 1e-4)
opt.approxHinv = PC
#opt.phim_last = reg.eval(mrec)

# opt = Optimization.InexactGaussNewton(maxIter=6)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = invProb.beta)
beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
#betaest = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
IRLS =Directives.Update_IRLS( phi_m_last = phim, phi_d_last = phid )

inv = Inversion.BaseInversion(invProb, directiveList=[beta,IRLS])

m0 = mrec

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*mrec

Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l0l2.den',m_out)

pred = prob.fields(mrec)

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data', vmin = np.min(d), vmax = np.max(d))
plt.savefig(home_dir + str('\Figure3.png'), dpi=300, bb_inches='tight')
PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
plt.savefig(home_dir + str('\Figure4.png'), dpi=300, bb_inches='tight')
print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) ) 
#%% Plot out a section of the compact model

yslice = midx
plt.figure(figsize=(15,10))
plt.suptitle('Compact Inversion')
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='w',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (kg/m3)')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (vmin,vmax))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.scatter(rxLoc[0:,0], rxLoc[0:,1], color='w',s=1)
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('Easting (m)');plt.ylabel('Northing (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (kg/m3)')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax))
plt.title('Cross Section')
plt.xlabel('Easting (m)');plt.ylabel('Elevation (m)')
plt.gca().set_aspect('equal', adjustable='box')
cb = plt.colorbar(dat1[0],orientation="vertical", ticks=np.linspace(vmin, vmax, 4))
cb.set_label('Density (kg/m3)')

plt.savefig(home_dir + str('\Figure5.png'), dpi=300, bb_inches='tight')


plt.figure(figsize=(15,10))
ax = plt.subplot(121)
plt.hist(mrec,100)
plt.xlim(mrec.mean() - 4.*(mrec.std()), mrec.mean() + 4.*(mrec.std()))
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model values - Sparse lp:'+str(lpnorms[0]))

ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*mrec,100)
plt.xlim(mrec.mean() - 4.*(mrec.std()), mrec.mean() + 4.*(mrec.std()))
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model gradient values - Sparse lqx: ' + str(lpnorms[1]) + ' lqy:'+ str(lpnorms[2]) + ' lqz:' + str(lpnorms[3]))
plt.savefig(home_dir + str('\Figure6.png'), dpi=300, bb_inches='tight')