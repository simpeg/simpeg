#%%
from SimPEG import *
import simpegPF as PF
import pylab as plt
import os
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\CraigModel\\GRAV\\checkerboard_tests'
inpfile = 'PYGRAV3D_inv.inp'
dsep = '\\'
os.chdir(home_dir)
plt.close('all')

#%% User input
# Initial beta
beta_in = 1e-2

# Treshold values for compact norm
eps_p = 1e-3 # Small model values
eps_q = 1e-3 # Small model gradient

# Plotting parameter
vmin = -0.3
vmax = 0.3
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
PF.Gravity.plot_obs_2D(survey,'Observed Data')

#%% Run inversion
prob = PF.Gravity.GravityIntegral(mesh, mapping = idenMap, actInd = actv)
prob.solverOpts['accuracyTol'] = 1e-4

survey.pair(prob)

# Write out the predicted file and generate the forward operator
pred = prob.fields(mstart)

PF.Gravity.writeUBCobs(home_dir + dsep + 'Pred0.dat',survey,pred)

# 
# Make depth weighting
#wr = np.sum(prob.G**2.,axis=0)**0.5 / mesh.vol[actv]
#wr = ( wr/np.max(wr) )
#wr_out = actvMap * wr

# Load weighting  file
if wgtfile is None:  
    wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3., np.min(mesh.hx)/4.)
    wr = wr**2.
else:
    wr = Mesh.TensorMesh.readModelUBC(mesh, home_dir + dsep + wgtfile)   
    wr = wr[actv]
    wr = wr**2.

#%% Plot depth weighting
plt.figure()
ax = plt.subplot()
mesh.plotSlice(actvMap*wr, ax = ax, normal = 'Y', ind=midx+1 ,clim = (0, wr.max()))
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

#%% Create inversion objects

# First start with an l2 regularization
reg = Regularization.Simple(mesh, indActive = actv, mapping = idenMap)
reg.mref = mref
reg.wght = wr


# Create pre-conditioner 
diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
PC     = Utils.sdiag(diagA**-1.)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd
opt = Optimization.ProjectedGNCG(maxIter=20,lower=bounds[0],upper=bounds[1], maxIterCG= 50, tolCG = 1e-4)
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
Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l2l2.sus',m_out)
#Utils.meshutils.writeUBCTensorModel(home_dir+dsep+'wr.dat',mesh,wr_out)

# Plot predicted
pred = prob.fields(mrec)
#PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
#PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')

print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) ) 

#%% Plot out sections of the smooth model

yslice = midx+1
m_out[m_out==-100] = np.nan

plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (vmin,vmax), pcolorOpts = {'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (vmin,vmax), pcolorOpts = {'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax), pcolorOpts = {'cmap':'bwr'})
plt.title('Cross Section')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

plt.figure()
ax = plt.subplot(121)
plt.hist(mrec,100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model values - Smooth')
ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*mrec,100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model gradient values - Smooth')
#%% Run one more round for sparsity
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
opt = Optimization.ProjectedGNCG(maxIter=20 , maxIterLS = 20,lower=bounds[0],upper=bounds[1], maxIterCG= 50, tolCG = 1e-4)
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

Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l0l2.sus',m_out)

pred = prob.fields(mrec)

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data', vmin = np.min(d), vmax = np.max(d))
PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) ) 
#%% Plot out a section of the model

yslice = midx
m_out[m_out==-100] = np.nan
plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-18, clim = (vmin,vmax) , pcolorOpts = {'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-18]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-23, clim = (vmin,vmax), pcolorOpts = {'cmap':'bwr'})
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-23]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (vmin,vmax), pcolorOpts = {'cmap':'bwr'})
plt.title('Cross Section')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

plt.figure()
ax = plt.subplot(121)
plt.hist(mrec,100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model values - Sparse lp:'+str(lpnorms[0]))
ax = plt.subplot(122)
plt.hist(reg.regmesh.cellDiffxStencil*mrec,100)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of model gradient values - Sparse lqx: ' + str(lpnorms[1]) + ' lqy:'+ str(lpnorms[2]) + ' lqz:' + str(lpnorms[3]))