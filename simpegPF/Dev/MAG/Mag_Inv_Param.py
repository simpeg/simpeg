# -*- coding: utf-8 -*-
"""
Created on Wed Feb 03 21:34:50 2016

@author: dominiquef
"""
from SimPEG import *
import simpegPF as PF
from simpegPF import BaseMag as MAG
from numpy.polynomial import polynomial

import pylab as plt

import os

#home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegpf\simpegPF\Dev'
#home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Parametric_plane'
#home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG\\Lalor'
#home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\ownCloud\Research\Nate\Modeling'
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\Google Drive\\DevDomNateDBE\\DomNate\\Fault_synthetic\\NE'
#home_dir = '.\\'
plt.close('all')

inpfile = 'PYMAG3D_inv.inp'

dsep = '\\'
os.chdir(home_dir)
## New scripts to be added to basecode
#from fwr_MAG_data import fwr_MAG_data
#from read_MAGfwr_inp import read_MAGfwr_inp
beta_in = 1e+0
ndv = -100
#%%
# Read input file
[mshfile, obsfile, topofile, m0_val, mref, magfile, wgtfile, chi, alphas, bounds, lpnorms] = PF.Magnetics.read_MAGinv_inp(home_dir + dsep + inpfile)

#obsfile = 'Synthetic.obs'
#obsfile ='Lalor_rtp_2pc_10nT_RegRem.obs'
#obsfile = 'Obs_ALL.obs'

# Discretization for new mesh
dx = 50.

# Load mesh file
mesh = Mesh.TensorMesh.readUBC(mshfile)
z0 = mesh.x0[2] + np.sum(mesh.hz)
#mesh = Utils.meshutils.readUBCTensorMesh(mshfile)

#V2D = polynomial.polyvander2d(mesh.vectorCCx,mesh.vectorCCy,[1,1])
# Load in observation file
survey = PF.Magnetics.readUBCmagObs(obsfile)
rxLoc_full = survey.srcField.rxList[0].locs
data = survey.dobs
wd = survey.std

npad = 10
#%% Pick points from dats and generate local mesh
PF.Magnetics.plot_obs_2D(rxLoc_full,data, levels = [0.])

#PF.Magnetics.plot_obs_2D(dobs[:,:3],dobs[:,3],dobs[:,4],'Observed Data')
gin = np.asarray(plt.ginput(100, timeout = 0))
#gin = np.asarray([[ -81.82517326, -167.83403552],
#       [ -21.0157401 ,  133.78075295]])
for ii in range(gin.shape[0]-1):
    
    dl_len = np.sqrt( np.sum((gin[ii,:] - gin[ii+1,:])**2) )
    dl_x = ( gin[ii,0] - gin[ii+1,0] ) / dl_len
    dl_y = ( gin[ii+1,1] - gin[ii,1]  ) / dl_len
    azm =  -np.arctan(dl_x/dl_y)
    
    # Create rotation matrix
    Rz = np.array([[np.cos(azm), -np.sin(azm)],
                   [np.sin(azm), np.cos(azm)]])
    
    # Re-center the experiment
    midx = np.median([gin[ii,0],gin[ii+1,0]])
    midy = np.median([gin[ii+1,1],gin[ii,1]])
    
    temp = np.vstack([rxLoc_full[:,0].T- midx, rxLoc_full[:,1].T- midy])
    
    # Rotate
    ROTxy = Rz.dot(temp)
    
    # Grab data points within a box
    indx = (np.abs(ROTxy[0,:]) < dl_len) & (np.abs(ROTxy[1,:]) < dl_len/2)
    
    subrx = MAG.RxObs(np.c_[ROTxy[0,indx].T, ROTxy[1,indx].T, rxLoc_full[indx,2]])
    d = data[indx]
    uncert = wd[indx]
    
    srcParam = np.asarray(survey.srcField.param)
    srcParam[2] = srcParam[2] - np.rad2deg(azm)
    srcField = MAG.SrcField([subrx],srcParam)
    survey = MAG.LinearSurvey(srcField)
    survey.dobs =  d
    survey.std =  uncert
    PF.Magnetics.writeUBCobs(home_dir+'\\Obsloc_local.dat',survey,survey.dobs)
    rxLoc = survey.srcField.rxList[0].locs
    PF.Magnetics.plot_obs_2D(rxLoc,d,'Observed Data')
    
    ndata = rxLoc.shape[0]
    
    # Get extent limits
    xlim = np.max(np.abs(rxLoc[:,0]))
    ylim = np.max(np.abs(rxLoc[:,1]))
    
    ncx = int(4*xlim/dx)
    ncy = int(2*ylim/dx)
    ncz = int(np.min([ncx,ncy]))
    
    hxind = [(dx,npad,-1.3),(dx, ncx),(dx,npad,1.3)]
    hyind = [(dx,npad,-1.3),(dx, ncy),(dx,npad,1.3)]
    hzind = [(dx,npad,-1.3),(dx, ncz)]
    
    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')
    
    mesh.x0[2] = np.max(rxLoc[:,2]) - np.sum(mesh.hz) # Keep top of mesh at same location as before
    Mesh.TensorMesh.writeUBC(mesh,home_dir+dsep+'Mesh_local.msh')
    
    
    # Load in topofile or create flat surface
    if topofile == 'null':
    
        # All active
        actv = np.asarray(range(mesh.nC))
    
    else:
    
        topo = np.genfromtxt(topofile,skip_header=1)
        temp = np.vstack([topo[:,0].T- midx, topo[:,1].T- midy])
    
        # Rotate
        ROTxy = Rz.dot(temp)
        ROT_topo = np.c_[ROTxy[0,:].T, ROTxy[1,:].T, topo[:,2]]
    
        # Find the active cells
        actv = PF.Magnetics.getActiveTopo(mesh,ROT_topo,'N')
    
    nC = len(actv)
    idenMap = Maps.IdentityMap(nP = nC)
    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, ndv)
    
    # Load starting model file
#    if isinstance(mstart, float):
    
    mstart = np.ones(nC) * m0_val
#    else:
#        mstart = Utils.meshutils.readUBCTensorModel(mstart,mesh)
#        mstart = mstart[actv]
    
    
    
    # Get magnetization vector for MOF
    if magfile=='DEFAULT':
    
        M_xyz = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * survey.srcField.param[1], np.ones(nC) * survey.srcField.param[2])
    
    else:
        M_xyz = np.genfromtxt(magfile,delimiter=' \n',dtype=np.str,comments='!')
    
    
    # Get index of the center
    #==============================================================================
    # midx = int(mesh.nCx/2)
    # midy = int(mesh.nCy/2)
    #==============================================================================
    
    # Create forward operator
    #F = PF.Magnetics.Intrgl_Fwr_Op(mesh,B,M_xyz,rxLoc,actv,'tmi')
    
    #%% Run inversion
    # First start with regular inversion for regional removal
    prob = PF.Magnetics.MagneticIntegral(mesh, mapping = idenMap, actInd = actv)
    prob.solverOpts['accuracyTol'] = 1e-4
    
    survey.pair(prob)
    
    #%%
    pred = prob.fields(mstart)
    
    wr = np.sum(prob.G**2.,axis=0)**0.5 / mesh.vol[actv]
    wr = ( wr/np.max(wr) )
    #IWr = Utils.sdiag(1/wr)
    #wrMap = PF.BaseMag.WeightMap(mesh, wr)
    
    #prob.mapping = wrMap
    #prob._G = prob._G * IWr
    
    reg = Regularization.Simple(mesh, indActive = actv, mapping = idenMap)
    reg.mref = 0.
    reg.wght = wr
    #reg.alpha_s = 1.
    
    # Create pre-conditioner 
    diagA = np.sum(prob.G**2.,axis=0) + beta_in*(reg.W.T*reg.W).diagonal()
    PC     = Utils.sdiag(diagA**-1.)
    
    
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1./survey.std
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
    
    #%% Temporary plotting scipts
    yslice = 14
    plt.figure()
    ax = plt.subplot(221)
    mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-5, clim = (mrec.min(), mrec.max()))
    plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
    plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
    plt.xlabel('x');plt.ylabel('z')
    plt.gca().set_aspect('equal', adjustable='box')
    
    ax = plt.subplot(222)
    mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-8, clim = ( mrec.min(), mrec.max()))
    plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
    plt.title('Z: ' + str(mesh.vectorCCz[-8]) + ' m')
    plt.xlabel('x');plt.ylabel('z')
    plt.gca().set_aspect('equal', adjustable='box')
    
    ax = plt.subplot(212)
    mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = ( mrec.min(), mrec.max()))
    plt.title('Cross Section')
    plt.xlabel('x');plt.ylabel('z')
    plt.gca().set_aspect('equal', adjustable='box')
    
    #%% Remove core cells and forward model
    m_out = np.reshape(m_out,(mesh.nCx,mesh.nCy,mesh.nCz), order = 'F')
    m_out[npad:-npad,npad:-npad,npad:] = m_out[npad:-npad,npad:-npad,npad:]*0.
    
    m_out = mkvc(m_out)
    m_pad = m_out[actv]
    
    Mesh.TensorMesh.writeModelUBC(mesh,home_dir+'\\SimPEG_Inv_l2l2.sus',actvMap*mrec)
    Mesh.TensorMesh.writeModelUBC(mesh,home_dir+'\\SimPEG_Scooped.sus',actvMap*m_pad)
    # Forward model the fields and substract from data
    fwr = prob.fields(m_pad)
    
    d_res = survey.dobs - fwr
    
    PF.Magnetics.plot_obs_2D(rxLoc,fwr,'Forward Scoop Data', levels = [0.])
    
    PF.Magnetics.plot_obs_2D(rxLoc,d_res,'RegRem Data', levels = [0.])
    
    survey.dobs =  d_res
    #%% Reduce the space to only the core region
    
    # Create new mesh for local inversion
    hxind = [(dx, ncx)]
    hyind = [(dx, ncy)]
    hzind = [(dx, ncz)]
    
    x0 = mesh.x0
    z0 = x0[2] + np.sum(mesh.hz)
    
    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')
    
    mesh.x0[2] = z0 - np.sum(mesh.hz)
    
    core = m_out==0
    grnd = core[actv]    
    
    inds = np.asarray([inds for inds, elem in enumerate(grnd, 1) if elem], dtype = int) - 1
    
    #%% Re-run with poly map
    # Load in topofile or create flat surface
    if topofile == 'null':
    
        # All active
        actv = np.asarray(range(mesh.nC))
    
    else:
    
        topo = np.genfromtxt(topofile,skip_header=1)
        temp = np.vstack([topo[:,0].T- midx, topo[:,1].T- midy])
    
        # Rotate
        ROTxy = Rz.dot(temp)
        ROT_topo = np.c_[ROTxy[0,:].T, ROTxy[1,:].T, topo[:,2]]
    
        # Find the active cells
        actv = PF.Magnetics.getActiveTopo(mesh,ROT_topo,'N')
    
    nC = len(actv)
    idenMap = Maps.IdentityMap(nP = nC)
    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, ndv)
    
    # Load starting model file
    mstart = np.ones(nC) * m0_val
    
    
    
    
    #%% Invert with polymap for plane
    #
    #
    # Create active map to go from reduce set to full
    #actvMap = Maps.ActiveCells(mesh, actv, -100)
    #
    ## Creat reduced identity map
    #idenMap = Maps.IdentityMap(nP = nC)
    #
    XYZ = mesh.gridCC
    
    order = [1,1]
    
    YZ = Utils.ndgrid(mesh.vectorCCy, mesh.vectorCCz)
    
    V = polynomial.polyvander2d(YZ[:,0], YZ[:,1], order)
    
    #f = polynomial.polyval2d(XYZ[:,1], XYZ[:,2], c.reshape((order[0]+1,order[1]+1))) - XYZ[:,0]
    
    polymap = Maps.PolyMap(mesh, order, normal='X', logSigma=False, actInd = actv)
    polymap.slope = 1.0
    #polymap.actInd = actv
    
    #m0 = np.r_[1e-2, 0., 0.0, -0.5, 0.2, 0.]
    m0 = np.r_[1e-4, 0, 1., 0., 0., 0.]
    
    
    
    #Mesh.TensorMesh.writeModelUBC(mesh,home_dir+dsep+'True_m.sus',polymap*m0)
    #Mesh.TensorMesh.writeModelUBC(mesh,home_dir+dsep+'Starting_m.sus',actvMap*polymap*m0)
    
    m1D = Mesh.TensorMesh([(order[0]+1)*(order[1]+1)+2])
    
    weight = ((V**2).sum(axis=0))**0.5
    weight = weight / weight.max()
    prob_core = PF.Magnetics.MagneticIntegral(mesh, mapping = idenMap*polymap, actInd = actv)
    prob_core.solverOpts['accuracyTol'] = 1e-4
    
    survey.pair(prob_core)
    
    #prob_core._G = prob.G[:,inds]
    
    #%%
    pred = prob_core.fields(m0)
    
    wr = np.sum(prob_core.G**2.,axis=0)**0.5 / mesh.vol[actv]
    wr = ( wr/np.max(wr) )
    IWr = Utils.sdiag(1/wr)
    wrMap = PF.BaseMag.WeightMap(mesh, wr)
    
    prob_core.mapping = wrMap * polymap
    prob_core._G = prob_core.G * IWr
    
    reg =  Regularization.Simple(m1D)
    reg.alpha_x = 0.
    reg.alpha_y = 0.
    reg.alpha_z = 0.
    reg.norms   = [2., 2., 2., 2.]
    #reg.mref = np.r_[0., 1., 421500., 1000., 10., 0.5]
    reg.mref = np.zeros(6)
    #==============================================================================
    # reg.wght = np.asarray([  1.00000000e+00 ,6.50176844e-01 ,4.20328370e-06  ,1.52004209e-03,
    #    4.82475226e-05  ,1.37620903e-02])**2.
    #==============================================================================
    lower = np.r_[0.,0.,-1e+8,-1e+8,-1e+8,-1e+8]
    upper = np.r_[1.,1.,1e+8,1e+8,1e+8,1e+8]
    #reg.mref = mref
    #reg.alpha_s = 1.
    
    
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1./survey.std
    opt = Optimization.ProjectedGNCG(maxIter=15,maxIterLS=50, maxIterCG = 10,tolCG = 1e-3, lower = lower,upper=upper)
    #opt.approxHinv = sp.eye(6)
    
    beta_in = 1e+2
    
    # opt = Optimization.InexactGaussNewton(maxIter=6)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = beta_in)
    beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
    #betaest = Directives.BetaEstimate_ByEig()
    target = Directives.TargetMisfit()
    up_Wj = Directives.Update_Wj()
    up_Wj.itr = 2
    
    inv = Inversion.BaseInversion(invProb, directiveList=[beta,target,up_Wj])
    
    #m0 = mstart
    #Utils.diagEst(prob.Jtvec,2)
    
    # Run inversion
    mrec = inv.run(m0)
    
    sus = polymap*mrec
    m_out = actvMap * sus
    
    # Write result
    Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_Param.sus',m_out)
    Mesh.TensorMesh.writeUBC(mesh,'Working_mesh.msh')
    
    m_out = actvMap *polymap*m0
    Mesh.TensorMesh.writeModelUBC(mesh,'SimPEG_Param_m0.sus',m_out)
    # Plot predicted
    pred = prob_core.fields(mrec)
    
    PF.Magnetics.plot_obs_2D(rxLoc,pred,'Predicted Data', vmin = np.min(survey.dobs), vmax = np.max(survey.dobs))
    
    PF.Magnetics.plot_obs_2D(rxLoc,survey.dobs-pred,'Residual Data')
    
    #PF.Magnetics.writeUBCobs(home_dir + dsep + 'Pred_Final.pre',B,M,rxLoc,pred,np.ones(len(d)))
    
    
    print "Final misfit:" + str(np.sum( ((d-pred)/uncert)**2. ) )
    
    #%% Write parametric surface
    yz = Utils.ndgrid(np.r_[-ncy/2*dx,0,ncy/2*dx],np.r_[mesh.vectorCCz[-1],mesh.vectorCCz[0]])
    
    xout = polynomial.polyval2d(yz[:,0],yz[:,1],np.reshape(mrec[2:],(2,2)))
    
    
    xyz = np.c_[xout,yz]

    # Rotate back to global coordinates
    # Create rotation matrix
    Rz = np.array([[np.cos(-azm), -np.sin(-azm)],
                   [np.sin(-azm), np.cos(-azm)]])
    
    
    temp = Rz.dot( np.vstack([xyz[:,0].T, xyz[:,1].T]) )
    
    if ii == 0:
    # Rotate
        ROTxyz = np.c_[temp[0,:].T + midx, temp[1,:].T + midy, xyz[:,2]]
    else:
        ROTxyz = np.vstack([ROTxyz,np.c_[temp[0,:].T + midx, temp[1,:].T + midy, xyz[:,2]]])

#Write out the surface        
with file(home_dir + dsep + 'Surf.dat','w') as fid:
        np.savetxt(fid, ROTxyz, fmt='%e',delimiter=' ',newline='\n')

#%% Right GOCAD ts file
with file(home_dir + dsep + 'Surf.ts','w') as fid:

    fid.write('GOCAD TSurf 1\n')
    fid.write('HEADER {name:Mag_Param}\n')
    fid.write('TFACE\n')
    for ii in range(ROTxyz.shape[0]):
        fid.write('VRTX %i %6.2f %6.2f %6.2f\n' %(ii+1, ROTxyz[ii,0],ROTxyz[ii,1],ROTxyz[ii,2]) )

    for ii in range(ROTxyz.shape[0]/3):
        fid.write('TRGL %i %i %i\n' %(ii+1, ii+2, np.mod(ii+3,6)+1 ))
        fid.write('TRGL %i %i %i\n' %(ii+2, np.mod(ii+3,6)+1, np.mod(ii+4,6)+1 ))

    fid.write('END\n')