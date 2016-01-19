import os

#home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev'
#home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo'
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Nut_Cracker\\Induced_MAG3C'

inpfile = 'PYMAG3D_inv.inp'

dsep = '\\'
os.chdir(home_dir)

#%%
from SimPEG import *
import simpegPF as PF
import pylab as plt

## New scripts to be added to basecode
#from fwr_MAG_data import fwr_MAG_data
#from read_MAGfwr_inp import read_MAGfwr_inp

#%%
# Read input file
[mshfile, obsfile, topofile, mstart, mref, magfile, wgtfile, chi, alphas, bounds, lpnorms] = PF.BaseMag.read_MAGinv_inp(home_dir + dsep + inpfile)

# Load mesh file
mesh = Utils.meshutils.readUBCTensorMesh(mshfile)
    
# Load in observation file
[B,M,dobs] = PF.BaseMag.readUBCmagObs(obsfile)

rxLoc = dobs[:,0:3]
d = dobs[:,3]
wd = dobs[:,4]

ndata = rxLoc.shape[0]

beta_in = 1e+2

# Load in topofile or create flat surface
if topofile == 'null':
    
    # All active
    actv = np.ones(mesh.nC)
    
else: 
    
    topo = np.genfromtxt(topofile,skip_header=1)
    # Find the active cells
    actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')

nC = int(sum(actv))

# Load starting model file
if isinstance(mstart, float):
    mstart = np.ones(nC) * mstart
else:
    mstart = Utils.meshutils.readUBCTensorModel(mstart,mesh)
    mstart = mstart[actv==1]

# Load reference file
if isinstance(mref, float):
    mref = np.ones(nC) * mref
else:
    mref = Utils.meshutils.readUBCTensorModel(mref,mesh)
    mref = mref[actv==1]
    
# Get magnetization vector for MOF
if magfile=='DEFAULT':
    
    M_xyz = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * M[0], np.ones(nC) * M[1])
    
else:
    M_xyz = np.genfromtxt(magfile,delimiter=' \n',dtype=np.str,comments='!')

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

# Create forward operator
F = PF.Magnetics.Intrgl_Fwr_Op(mesh,B,M_xyz,rxLoc,actv,'tmi')

# Get distance weighting function
wr = PF.Magnetics.get_dist_wgt(mesh,rxLoc,actv,3.,np.min(mesh.hx)/4)
wrMap = PF.BaseMag.WeightMap(mesh, wr)

wr_out = np.zeros(mesh.nC)
wr_out[actv==1] = wr
Utils.writeUBCTensorModel(home_dir+dsep+'wr.dat',mesh,wr_out)

# Write out the predicted
pred = F.dot(mstart)
PF.Magnetics.writeUBCobs(home_dir + dsep + 'Pred.dat',B,M,rxLoc,pred,wd)

#%%
plt.figure()
ax = plt.subplot()
mesh.plotSlice(wr_out, ax = ax, normal = 'Y', ind=midx)
plt.title('Distance weighting')
plt.xlabel('x');plt.ylabel('z')
plt.gca().set_aspect('equal', adjustable='box')

#%% Plot obs data
PF.Magnetics.plot_obs_2D(rxLoc,d,wd,'Observed Data')

#%% Run inversion
prob = PF.Magnetics.MagneticIntegral(mesh, F)
prob.solverOpts['accuracyTol'] = 1e-4
survey = Survey.LinearSurvey()
survey.pair(prob)
#survey.makeSyntheticData(data, std=0.01)
survey.dobs=d
#survey.mtrue = model

# Create pre-conditioner 
diagA = np.sum(F**2.,axis=0) + beta_in*np.ones(nC)
PC     = sp.spdiags(diagA**-1., 0, nC, nC);

reg = Regularization.Tikhonov(mesh, mapping=wrMap)
reg.mref = mref

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = wd
#opt = Optimization.ProjectedGNCG(maxIter=6,lower=-1.,upper=1.)
opt.approxHinv = PC

opt = Optimization.InexactGaussNewton(maxIter=6)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = beta_in)
beta = Directives.BetaSchedule(coolingFactor=8, coolingRate=2)
#betaest = Directives.BetaEstimate_ByEig()
target = Directives.TargetMisfit()
inv = Inversion.BaseInversion(invProb, directiveList=[beta, target])
reg.alpha_s =0.0025
m0 = mstart

# Run inversion
mrec = inv.run(m0)

m_out = np.ones(mesh.nC)
m_out[actv==1] = mrec

# Write result
Utils.meshutils.writeUBCTensorModel('SimPEG_inv.sus',mesh,m_out)

# Plot predicted
pred = F.dot(mrec)
PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')

print "Final misfit:" + str(np.sum( ((d-pred)/wd)**2. ) ) 