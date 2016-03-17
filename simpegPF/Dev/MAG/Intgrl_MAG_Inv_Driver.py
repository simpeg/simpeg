#%%
from SimPEG import *
import simpegPF as PF
import pylab as plt

import os

#home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev'
#home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Nut_Cracker\\Induced_MAG3C'
home_dir = '.\\'
#home_dir = '.\\'

inpfile = 'PYMAG3D_inv.inp'

dsep = '\\'
os.chdir(home_dir)
## New scripts to be added to basecode
#from fwr_MAG_data import fwr_MAG_data
#from read_MAGfwr_inp import read_MAGfwr_inp

#%%
# Read input file
[mshfile, obsfile, topofile, mstart, mref, magfile, wgtfile, chi, alphas, bounds, lpnorms] = PF.Magnetics.read_MAGinv_inp(home_dir + dsep + inpfile)

# Load mesh file
mesh = Mesh.TensorMesh.readUBC(mshfile)
#mesh = Utils.meshutils.readUBCTensorMesh(mshfile) 

# Load in observation file
survey = PF.Magnetics.readUBCmagObs(obsfile)

rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs
wd = survey.std

ndata = survey.srcField.rxList[0].locs.shape[0]

beta_in = 1e+5

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
actvMap = Maps.ActiveCells(mesh, actv, -100)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP = nC)

# Load starting model file
if isinstance(mstart, float):
    
    mstart = np.ones(nC) * mstart
else:
    mstart = Utils.meshutils.readUBCTensorModel(mstart,mesh)
    mstart = mstart[actv]

# Load reference file
if isinstance(mref, float):
    mref = np.ones(nC) * mref
else:
    mref = Utils.meshutils.readUBCTensorModel(mref,mesh)
    mref = mref[actv]
    
# Get magnetization vector for MOF
if magfile=='DEFAULT':
    
    M_xyz = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * survey.srcField.param[1], np.ones(nC) * survey.srcField.param[2])
    
else:
    M_xyz = np.genfromtxt(magfile,delimiter=' \n',dtype=np.str,comments='!')

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

# Get distance weighting function
#==============================================================================
# wr = PF.Magnetics.get_dist_wgt(mesh,rxLoc,actv,3.,np.min(mesh.hx)/4)
# #wrMap = PF.BaseMag.WeightMap(nC, wr)
#==============================================================================

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,d,wd,'Observed Data')

#%% Run inversion
prob = PF.Magnetics.MagneticIntegral(mesh, mapping = idenMap, actInd = actv)
prob.solverOpts['accuracyTol'] = 1e-4

#survey = Survey.LinearSurvey()
survey.pair(prob)
#survey.makeSyntheticData(data, std=0.01)
#survey.dobs=d
#survey.mtrue = model
# Write out the predicted
pred = prob.fields(mstart)
PF.Magnetics.writeUBCobs(home_dir + dsep + 'Pred.dat',survey,pred)

wr = np.sum(prob.G**2.,axis=0)**0.5
wr = ( wr/np.max(wr) )
wr_out = actvMap * wr

plt.figure()
ax = plt.subplot()
mesh.plotSlice(wr_out, ax = ax, normal = 'Y', ind=midx ,clim = (-1e-3, wr.max()))
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

reg = Regularization.Simple(mesh, indActive = actv, mapping = idenMap)
reg.mref = mref
reg.wght = wr
#reg.alpha_s = 1.

# Create pre-conditioner 
diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()*wr
PC     = Utils.sdiag(diagA**-1.)


dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/wd
opt = Optimization.ProjectedGNCG(maxIter=10,lower=0.,upper=1., maxIterCG= 20, tolCG = 1e-3)
opt.approxHinv = PC

# opt = Optimization.InexactGaussNewton(maxIter=6)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = beta_in)
beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
#betaest = Directives.BetaEstimate_ByEig()
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

#%% Plot out a section of the model

yslice = midx

plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (-1e-3, mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (-1e-3, mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')


ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (-1e-3, mrec.max()))
plt.title('Cross Section')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

#%% Run one more round for sparsity
phim = invProb.phi_m_last
phid =  invProb.phi_d

reg = Regularization.Sparse(mesh, indActive = actv, mapping = idenMap)
reg.recModel = mrec
reg.mref = mref
reg.wght = wr
reg.eps = 1e-5
reg.p   = lpnorms[0]
reg.qx  = lpnorms[1]
reg.qz  = lpnorms[2]
reg.qy  = lpnorms[3]

diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()*(wr)
PC     = Utils.sdiag(diagA**-1.)

#reg.alpha_s = 1.

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = wd
opt = Optimization.ProjectedGNCG(maxIter=10 ,lower=0.,upper=1., maxIterCG= 10, tolCG = 1e-4)
opt.approxHinv = PC
#opt.phim_last = reg.eval(mrec)

# opt = Optimization.InexactGaussNewton(maxIter=6)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = invProb.beta)
beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
#betaest = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
IRLS =Directives.update_IRLS( phi_m_last = phim, phi_d_last = phid )

inv = Inversion.BaseInversion(invProb, directiveList=[beta,IRLS])

m0 = mrec

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*mrec

Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_inv_l0l2.sus',m_out)

pred = prob.fields(mrec)

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
PF.Magnetics.plot_obs_2D(rxLoc,d,wd,'Observed Data')
print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) ) 
#%% Plot out a section of the model

yslice = midx
plt.figure()
ax = plt.subplot(221)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (-1e-3, mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(222)
mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = (-1e-3, mrec.max()))
plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

ax = plt.subplot(212)
mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (-1e-3, mrec.max()))
plt.title('Cross Section')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')