import os
from SimPEG import np, sp, Utils, Mesh, mkvc
import simpegDCIP as DC
import pylab as plt
#from ipywidgets import interact, IntSlider
from matplotlib import animation
from JSAnimation import HTMLWriter
import time
import re
from readUBC_DC2DMesh import readUBC_DC2DMesh
from readUBC_DC2DModel import readUBC_DC2DModel
from readUBC_DC2DLoc import readUBC_DC2DLoc
from convertObs_DC3D_to_2D import convertObs_DC3D_to_2D
from readUBC_DC3Dobs import readUBC_DC3Dobs

#%%
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Two_Sphere'
msh_file = 'Mesh_2D.msh'
mod_file = 'Model_2D.con'
obs_file = 'FWR_data3D.dat'

dsep = '\\'

# Forward solver
slvr = 'BiCGStab' #'LU'

# Preconditioner
pcdr = 'Jacobi'#'Gauss-Seidel'#

# Number of padding cells to remove from plotting
padc = 15

# Load UBC mesh 2D
mesh = readUBC_DC2DMesh(home_dir + dsep + msh_file)

# Load model
model = readUBC_DC2DModel(home_dir + dsep + mod_file)

# load obs file
[Tx,Rx,d,wd] = readUBC_DC3Dobs(home_dir + dsep + obs_file)
[Tx, Rx] = convertObs_DC3D_to_2D(Tx,Rx)
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
    if re.match(pcdr,'Jacobi'):
        dA = A.diagonal()
        P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])
    
    # Create Gauss-Seidel Preconditioner
    elif re.match(pcdr,'Gauss-Seidel'):
        LD = sp.tril(A,k=0)
        #LDinv = sp.linalg.splu(LD)

elif re.match(slvr,'LU'):
    # Factor A matrix
    Ainv = sp.linalg.splu(A)    
    print("LU DECOMP--- %s seconds ---" % (time.time() - start_time))
    
#%% Create SimPEG objects


# Create sub-mesh for plotting
hx = mesh.hx
hy = mesh.hy

hx_sub = hx[padc:-padc]
hy_sub = hy[padc:]
mesh_sub = Mesh.TensorMesh([hx_sub,hy_sub],(mesh.vectorNx[padc], mesh.vectorNy[padc]))
model_sub = model.reshape(mesh.nCy,mesh.nCx)
model_sub = mkvc(model_sub[padc:,padc:-padc].T)

xx = mesh_sub.vectorCCx
yy = mesh_sub.vectorCCy

#%% Solve
#txii = range(50,1950,100)
#jx_CC_sub = np.zeros((len(txii),mesh_sub.nCx,mesh_sub.nCy))
#jy_CC_sub = np.zeros((len(txii),mesh_sub.nCx,mesh_sub.nCy))

fig = plt.figure(figsize=(10,5))
axs = plt.axes(ylim = (yy[0],yy[-1]+mesh.hy[-1]*2), xlim = (xx[0],xx[-1]))#
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.ylim(yy[0],yy[-1]+mesh.hy[-1]*2)
plt.xlim(xx[0],xx[-1])
#im1 = axs.pcolormesh([],[],[], alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)
#im2 = axs.pcolormesh([],[],[],alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')
im1 = axs.pcolormesh(mesh_sub.vectorCCx,mesh_sub.vectorCCy,np.zeros((mesh_sub.nCy,mesh_sub.nCx)), alpha=0.75,vmin=-1e-2, vmax=1e-2)
im2 = axs.pcolormesh(mesh_sub.vectorCCx,mesh_sub.vectorCCy,np.zeros((mesh_sub.nCy,mesh_sub.nCx)), alpha=0.75,vmin=-1e-2, vmax=1e-2)
im3 = axs.streamplot(xx, yy, np.zeros((mesh_sub.nCy,mesh_sub.nCx)), np.zeros((mesh_sub.nCy,mesh_sub.nCx)),color='k')
im4 = axs.scatter([],[], c='r', s=200)
im5 = axs.scatter([],[], c='r', s=200)



#==============================================================================
# def init():
#     im1.set_data([[],[],[]])
#     im2.set_data([[],[],[]])
#     
#     return [im1]+[im2]
#==============================================================================

def animate(ii):
    
    

#for ii in range(len(txii)):
    
    removeStream()
    tx = np.asarray(np.c_[Tx[ii],np.ones(Tx[ii].shape[0])*mesh.vectorNy[-1]-1])
    inds = Utils.closestPoints(mesh, tx )
    RHS = mesh.getInterpolationMat( tx , 'CC').T*( [-1,1] / mesh.vol[inds] )
    
    if re.match(slvr,'BiCGStab'):
        
        if re.match(pcdr,'Jacobi'):
            dA = A.diagonal()
            P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])
    
            # Iterative Solve
            phi = sp.linalg.bicgstab(P*A,P*RHS, tol=1e-5)
            phi = mkvc(phi[0])
               
    elif re.match(slvr,'LU'):
        #Direct Solve
        phi = Ainv.solve(RHS)
    
    
    j = -Msig*Grad*phi
    j_CC = mesh.aveF2CCV*j
    
    # Compute charge density solving div*grad*phi
    Q = -mesh.faceDiv*mesh.cellGrad*phi
    
    jx_CC = j_CC[0:mesh.nC].reshape(mesh.nCy,mesh.nCx)
    jy_CC = j_CC[mesh.nC:].reshape(mesh.nCy,mesh.nCx)
    
    #%% Grab only the core for presentation
    jx_CC_sub = jx_CC[padc:,padc:-padc]

    jy_CC_sub = jy_CC[padc:,padc:-padc]

    Q_sub = Q.reshape(mesh.nCy,mesh.nCx)   
    Q_sub = Q_sub[padc:,padc:-padc]
    
    J_rho = np.sqrt(jx_CC_sub**2 + jy_CC_sub**2)
    lw = np.log10(J_rho/J_rho.min())
    

    #axs.imshow(Q_sub,alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)    
    #axs.imshow(np.log10(model_sub.reshape(mesh_sub.nCy,mesh_sub.nCx)),alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')     
    global im1
    im1 = axs.pcolormesh(mesh_sub.vectorCCx,mesh_sub.vectorCCy,Q_sub, alpha=0.75,vmin=-1e-2, vmax=1e-2)
    
    global im2
    im2 = axs.pcolormesh(mesh_sub.vectorCCx,mesh_sub.vectorCCy,np.log10(model_sub.reshape(mesh_sub.nCy,mesh_sub.nCx)), alpha=0.25)

    global im3
    im3 = axs.streamplot(xx, yy, jx_CC_sub, jy_CC_sub,color='k',linewidth = lw,density=0.5)
    
    global im4
    im4 = axs.scatter(tx[0,0],mesh.vectorNy[-1], c='r', s=75, marker='v' )
    
    global im5
    im5 = axs.scatter(tx[1,0],mesh.vectorNy[-1], c='b', s=75, marker='v' )
    

    #plt.show()
    #im1.set_array(Q_sub)
    #im2.set_array(np.log10(model_sub.reshape(mesh_sub.nCy,mesh_sub.nCx)))
    #im2.set_array(mesh_sub.vectorCCx, mesh_sub.vectorCCy,jx_CC_sub.T,jy_CC_sub.T)

    #return [im1] + [im2]
#%% Create widget
def removeStream():
    global im1
    im1.remove()    
    
    global im2
    im2.remove()  
    
    global im3
    im3.lines.remove()
    axs.patches = []
    
    global im4
    im4.remove()
    
    global im5
    im5.remove()
#def viewInv(msh,iteration):



#, linewidth=lw.T
#%%   
#interact(viewInv,msh = mesh_sub, iteration = IntSlider(min=0, max=len(txii)-1 ,step=1, value=0))
# set embed_frames=True to embed base64-encoded frames directly in the HTML
anim = animation.FuncAnimation(fig, animate,
                               frames=len(Tx), interval=5)
                               
anim.save(home_dir + '\\animation.html', writer=HTMLWriter(embed_frames=True))
