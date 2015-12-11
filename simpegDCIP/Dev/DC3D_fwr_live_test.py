"""
        Experimental script for the forward modeling of DC resistivity data 
        along survey lines defined by the user. The program loads in a 3D mesh
        and model which is used to design pole-dipole or dipole-dipole survey
        lines.
        
        Uses SimPEG to generate the forward problem and compute the LU
        factorization.
        
        Calls DCIP2D for the inversion of a projected 2D section from the full
        3D model.

        Assumes flat topo for now...
            
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
"""


#%%
from SimPEG import np, Utils, Mesh, mkvc, sp
import simpegDCIP as DC
import pylab as plt
from pylab import get_current_fig_manager
import time
import re
from readUBC_DC3Dobs import readUBC_DC3Dobs
from readUBC_DC2DModel import readUBC_DC2DModel
from writeUBC_DCobs import writeUBC_DCobs
import scipy.interpolate as interpolation
from plot_pseudoSection import plot_pseudoSection
from gen_DCIPsurvey import gen_DCIPsurvey
from convertObs_DC3D_to_2D import convertObs_DC3D_to_2D
import os
import sys

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegdc\simpegDCIP\Dev'
dsep = '\\'
#from scipy.linalg import solve_banded

# Load UBC mesh 3D
mesh = Utils.meshutils.readUBCTensorMesh(home_dir + '\Mesh_20m.msh')
#mesh = Utils.meshutils.readUBCTensorMesh(home_dir + '\MtIsa_20m.msh')

# Load model
#model = Utils.meshutils.readUBCTensorModel(home_dir + '\MtIsa_3D.con',mesh)
model = Utils.meshutils.readUBCTensorModel(home_dir + '\Synthetic.con',mesh)

# Specify survey type
stype = 'pdp'

# Survey parameters
a = 40
n = 10

# Forward solver
slvr = 'BiCGStab' #'LU'

# Inversion parameter
pct = 0.01
flr = 1e-4
chifact = 100
ref_mod = 1e-3

#%% Create system
#Set boundary conditions
mesh.setCellGradBC('neumann')

Div = mesh.faceDiv
Grad = mesh.cellGrad
Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

A = Div*Msig*Grad

# Change one corner to deal with nullspace
A[0,0] = 1
A = sp.csc_matrix(A)

start_time = time.time()

if re.match(slvr,'BiCGStab'):
    # Create Jacobi Preconditioner
    dA = A.diagonal()
    P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])

elif re.match(slvr,'LU'):
    # Factor A matrix
    Ainv = sp.linalg.splu(A)    
    print("LU DECOMP--- %s seconds ---" % (time.time() - start_time))

#%% Create survey
# Display top section 
top = int(mesh.nCz)-1

plt.figure()
ax_prim = plt.subplot(1,1,1)
mesh.plotSlice(model, ind=top, normal='Z', grid=True, pcolorOpts={'alpha':0.8}, ax =ax_prim)
plt.xlim([423000,424000])
plt.ylim([546200,547200])
plt.gca().set_aspect('equal', adjustable='box')
    
plt.show()
cfm1=get_current_fig_manager().window
gin=[1]

# Keep creating sections until returns an empty ginput (press enter on figure)
while bool(gin)==True:
    
    # Bring back the plan view figure and pick points     
    cfm1.activateWindow()
    plt.sca(ax_prim)
        
    # Takes two points from ginput and create survey
    gin = plt.ginput(2, timeout = 0)
    # gin = [(423393.22916666663, 546431.77083333337), (423588.54166666663, 546864.0625)]
    
    
    if not gin:
        print 'SimPED - Simulation has ended with return'
        break
        
    #gin = [(465.9879032258068, 12141.34375), (1648.0443548387098, 12149.083333333334)]
    
    # Add z coordinate to all survey... assume flat
    nz = mesh.vectorNz
    var = np.c_[np.asarray(gin),np.ones(2).T*nz[-1]]
    
    # Snap the endpoints to the grid. Easier to create 2D section.
    indx = Utils.closestPoints(mesh, var )
    endl = np.c_[mesh.gridCC[indx,0],mesh.gridCC[indx,1],np.ones(2).T*nz[-1]]
          
    [Tx, Rx] = gen_DCIPsurvey(endl, mesh, stype, a, n)
     
    dl_len = np.sqrt( np.sum((Tx[0][0:2,0] - Tx[-1][0:2,1])**2) ) 
    dl_x = ( Tx[-1][0,1] - Tx[0][0,0] ) / dl_len
    dl_y = ( Tx[-1][1,1] - Tx[0][1,0]  ) / dl_len
    azm =  np.arctan(dl_y/dl_x)
      
    # Plot stations along line   
    plt.scatter(Tx[0][0,:],Tx[0][1,:],s=20,c='g')
    plt.scatter(Rx[0][:,0::3],Rx[0][:,1::3],s=20,c='y')
    
    #%% Forward model data
    data = []#np.zeros( nstn*nrx )
    unct = []
    problem = DC.ProblemDC_CC(mesh)
        
    for ii in range(len(Tx)):
        start_time = time.time()
        
        # Select dipole locations for receiver
        rxloc_M = np.asarray(Rx[ii][:,0:3])
        rxloc_N = np.asarray(Rx[ii][:,3:])
        
        # Number of receivers
        nrx = rxloc_M.shape[0]
    
        
        
        if re.match(stype,'dpdp'):
            inds = Utils.closestPoints(mesh, np.asarray(Tx[ii]).T )
            RHS = mesh.getInterpolationMat(np.asarray(Tx[ii]).T, 'CC').T*( [-1,1] / mesh.vol[inds] )   
            
        elif re.match(stype,'pdp'): 
            
            # Create an "inifinity" pole
            tx =  np.squeeze(Tx[ii][:,0:1])
            tinf = tx + np.array([dl_x,dl_y,0])*dl_len*2
            inds = Utils.closestPoints(mesh, np.c_[tx,tinf].T)
            RHS = mesh.getInterpolationMat(np.asarray(Tx[ii]).T, 'CC').T*( [-1] / mesh.vol[inds] )  
        
        # Solve for phi on pole locations
        P1 = mesh.getInterpolationMat(rxloc_M, 'CC')
        P2 = mesh.getInterpolationMat(rxloc_N, 'CC')
    
        if re.match(slvr,'BiCGStab'):
            # Create Jacobi Preconditioner
            dA = A.diagonal()
            P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])        
                   
        elif re.match(slvr,'LU'):
            #Direct Solve
            phi = Ainv.solve(RHS)        
        
        # Iterative Solve
        Ainvb = sp.linalg.bicgstab(P*A,P*RHS, tol=1e-5)
        phi = mkvc(Ainvb[0])
        
        # Compute potential at each electrode
        dtemp = (P1*phi - P2*phi)*np.pi
        
        data.append( dtemp )     
        unct.append( np.abs(dtemp) * pct + flr)
       
        print("--- %s seconds ---" % (time.time() - start_time))  
        
        
    #%% Write data file in UBC-DCIP3D format
    writeUBC_DCobs(home_dir+'\FWR_data3D.dat',Tx,Rx,data,unct,'3D')     
    
    
    #%% Load 3D data
    [Tx, Rx, data, wd] = readUBC_DC3Dobs(home_dir + '\FWR_data3D.dat')
    
    
    #%% Convert 3D obs to 2D and write to file
    [Tx2d, Rx2d] = convertObs_DC3D_to_2D(Tx,Rx)
    
    writeUBC_DCobs(home_dir+'\FWR_3D_2_2D.dat',Tx2d,Rx2d,data,unct,'2D')        
    
    #%% Create a 2D mesh along axis of Tx end points and keep z-discretization    
    dx = np.min( [ np.min(mesh.hx), np.min(mesh.hy) ])
    nc = np.ceil(dl_len/dx)+3
    
    padx = dx*np.power(1.4,range(1,15))
    
    # Creating padding cells
    h1 = np.r_[padx[::-1], np.ones(nc)*dx , padx]
    
    # Create mesh with 0 coordinate centerer on the ginput points in cell center
    mesh2d = Mesh.TensorMesh([h1, mesh.hz], x0=(-np.sum(padx)-dx/2,mesh.x0[2]))
    
    # Create array of points for interpolating from 3D to 2D mesh
    xx = Tx[0][0,0] + mesh2d.vectorCCx * np.cos(azm)
    yy = Tx[0][1,0] + mesh2d.vectorCCx * np.sin(azm)
    zz = mesh2d.vectorCCy
    
    [XX,ZZ] = np.meshgrid(xx,zz)
    [YY,ZZ] = np.meshgrid(yy,zz)
    
    xyz2d = np.c_[mkvc(XX),mkvc(YY),mkvc(ZZ)]
    
    #plt.scatter(xx,yy,s=20,c='y')
    
    
    F = interpolation.NearestNDInterpolator(mesh.gridCC,model)
    m2D = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T
    
     
    #==============================================================================
    # mesh2d = Mesh.TensorMesh([mesh.hx, mesh.hz], x0=(mesh.x0[0]-endl[0,0],mesh.x0[2]))
    # m3D = np.reshape(model, (mesh.nCz, mesh.nCy, mesh.nCx))
    # m2D = m3D[:,1,:]
    #==============================================================================
    
    plt.figure()
    axs = plt.subplot(2,1,1)
    
    plt.xlim([0,nc*dx])
    plt.ylim([mesh2d.vectorNy[-1]-dl_len,mesh2d.vectorNy[-1]])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),alpha=0.5, cmap='gray')#axes = [mesh2d.vectorNx[0],mesh2d.vectorNx[-1],mesh2d.vectorNy[0],mesh2d.vectorNy[-1]])
    #mesh2d.plotImage(mkvc(m2D), grid=True, ax=axs)
    
    #%% Plot pseudo section
    
    plot_pseudoSection(Tx2d,Rx2d,data,nz[-1],stype)
    plt.colorbar
    plt.show()

    #%% Create dcin2d inversion files and run
    inv_dir = home_dir + '\Inv2D' 
    if not os.path.exists(inv_dir):
        os.makedirs(inv_dir)
        
    mshfile2d = 'Mesh_2D.msh'
    modfile2d = 'MtIsa_2D.con'
    obsfile2d = 'FWR_3D_2_2D.dat'
    inp_file = 'dcinv2d.inp'
    
    
    # Export 2D mesh
    fid = open(inv_dir + dsep + mshfile2d,'w')
    fid.write('%i\n'% mesh2d.nCx)
    fid.write('%f %f 1\n'% (mesh2d.vectorNx[0],mesh2d.vectorNx[1]))  
    np.savetxt(fid, np.c_[mesh2d.vectorNx[2:],np.ones(mesh2d.nCx-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
    fid.write('\n')
    fid.write('%i\n'% mesh2d.nCy)
    fid.write('%f %f 1\n'%( 0,mesh2d.hy[-1]))   
    np.savetxt(fid, np.c_[np.cumsum(mesh2d.hy[-2::-1])+mesh2d.hy[-1],np.ones(mesh2d.nCy-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
    fid.close()
    
    # Export 2D model
    fid = open(inv_dir + dsep + modfile2d,'w')
    fid.write('%i %i\n'% (mesh2d.nCx,mesh2d.nCy))
    np.savetxt(fid, mkvc(m2D[::-1,:].T), fmt='%e',delimiter=' ',newline='\n')
    fid.close()
    
    # Export data file
    writeUBC_DCobs(inv_dir + dsep + obsfile2d,Tx2d,Rx2d,data,unct,'2D') 
    
    # Write input file
    fid = open(inv_dir + dsep + inp_file,'w')
    fid.write('OBS LOC_X %s \n'% obsfile2d)
    fid.write('MESH FILE %s \n'% mshfile2d)
    fid.write('CHIFACT 1 %f\n'% chifact)
    fid.write('TOPO DEFAULT  %s \n')
    fid.write('INIT_MOD DEFAULT\n')
    fid.write('REF_MOD VALUE %e\n'% ref_mod)
    fid.write('ALPHA DEFAULT\n')
    fid.write('WEIGHT DEFAULT\n')
    fid.write('STORE_ALL_MODELS FALSE\n')
    fid.write('INVMODE SVD\n')
    fid.write('USE_MREF TRUE\n')
    fid.close()
    
    os.chdir(inv_dir)
    os.system('dcinv2d ' + inp_file)
    
    #%%
    #Load model
    minv = readUBC_DC2DModel(inv_dir + dsep + 'dcinv2d.con')
    #plt.figure()
    axs = plt.subplot(2,1,2)
    
    plt.xlim([0,nc*dx])
    plt.ylim([mesh2d.vectorNy[-1]-dl_len,mesh2d.vectorNy[-1]])
    plt.gca().set_aspect('equal', adjustable='box')
    
    minv = np.reshape(minv,(mesh2d.nCy,mesh2d.nCx))
    plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),alpha=0.5, cmap='gray')
    plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(minv),alpha=0.5, clim=(np.min(np.log10(m2D)),np.max(np.log10(m2D))))
    plt.colorbar


#%%

    
