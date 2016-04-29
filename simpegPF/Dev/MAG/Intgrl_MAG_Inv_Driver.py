#%%
from SimPEG import *
import simpegPF as PF
import pylab as plt

import os

home_dir = '.'

inpfile = 'PYMAG3D_inv.inp'

dsep = os.path.sep

os.chdir(home_dir)
plt.close('all')
#%%
# Read input file
[mshfile, obsfile, topofile, mstart, mref, magfile, wgtfile, chi, alphas, bounds, lpnorms] = PF.Magnetics.read_MAGinv_inp(home_dir + dsep + inpfile)

# Load mesh file
mesh = Mesh.TensorMesh.readUBC(mshfile)

# Load in observation file
survey = PF.Magnetics.readUBCmagObs(obsfile)

rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs
wd = survey.std

ndata = survey.srcField.rxList[0].locs.shape[0]

eps_p = 1e-4
eps_q = 1e-4
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

# Create reduced identity map
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

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')

#%% Run inversion
prob = PF.Magnetics.MagneticIntegral(mesh, mapping = idenMap, actInd = actv)
prob.solverOpts['accuracyTol'] = 1e-4

survey.pair(prob)

# Write out the predicted
pred = prob.fields(mstart)
PF.Magnetics.writeUBCobs(home_dir + dsep + 'Pred.dat',survey,pred)

wr = np.sum(prob.G**2.,axis=0)**0.5 / mesh.vol[actv]
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

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/wd
opt = Optimization.ProjectedGNCG(maxIter=10,lower=0.,upper=1., maxIterCG= 20, tolCG = 1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Add directives to the inversion
beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
beta_init = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
update_Jacobi = Directives.Update_lin_PreCond(onlyOnStart=True)

inv = Inversion.BaseInversion(invProb, directiveList=[beta,target,beta_init,update_Jacobi])

m0 = mstart

# Run inversion
mrec = inv.run(m0)

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

# Set parameters for sparsity
reg = Regularization.Sparse(mesh, indActive = actv, mapping = idenMap)
reg.recModel = mrec
reg.mref = mref
reg.wght = wr
reg.eps_p = eps_p
reg.eps_q = eps_q
reg.norms   = lpnorms


dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = wd
opt = Optimization.ProjectedGNCG(maxIter=20 ,lower=0.,upper=1., maxIterCG= 10, tolCG = 1e-4)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = invProb.beta)

# Create inversion directives
beta = Directives.BetaSchedule(coolingFactor=1, coolingRate=1)
update_beta = Directives.Scale_Beta(tol = 0.05)
target = Directives.TargetMisfit()
IRLS =Directives.Update_IRLS( phi_m_last = phim, phi_d_last = phid )
update_Jacobi = Directives.Update_lin_PreCond(onlyOnStart=False)
save_log = Directives.SaveOutputEveryIteration()
save_log.fileName = 'LogName_blabla'

inv = Inversion.BaseInversion(invProb, directiveList=[beta,IRLS,update_beta,update_Jacobi,save_log])

m0 = mrec

# Run inversion
mrec = inv.run(m0)

m_out = actvMap*mrec

# Write final model out.
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