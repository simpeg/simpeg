import os

home_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\Documents\GIT\SimPEG\simpegdc\simpegDCIP\Dev'

os.chdir(home_dir)

#%%
from SimPEG import np, Utils, Mesh, mkvc, SolverLU
import simpegDCIP as DC
import pylab as plt
#from ipywidgets import interact, IntSlider
from matplotlib import animation
from JSAnimation import HTMLWriter

from readUBC_DC2DMesh import readUBC_DC2DMesh
from readUBC_DC2DModel import readUBC_DC2DModel
from readUBC_DC2DLoc import readUBC_DC2DLoc

# Number of padding cells to remove from plotting
padc = 16

# Load UBC mesh 2D
mesh = readUBC_DC2DMesh('mesh2d_fine.txt')

# Load model
model = readUBC_DC2DModel('model2d_fine.con')

# load obs file
[txLoc,rxLoc,d,wd] = readUBC_DC2DLoc('obs2d_East.loc')

# Create SimPEG objects
rx = DC.RxDipole(rxLoc[:,0], rxLoc[:,1])
#tx = DC.SrcDipole([rx],txLoc[200,0],txLoc[200,1])

# Create sub-mesh for plotting
hx = mesh.hx
hy = mesh.hy

hx_sub = hx[padc:-padc]
hy_sub = hy[padc:]
mesh_sub = Mesh.TensorMesh([hx_sub,hy_sub],(hx_sub[0], -sum(hy_sub)))
model_sub = model.reshape(mesh.nCy,mesh.nCx)
model_sub = mkvc(model_sub[padc:,padc:-padc].T)

xx = mesh_sub.vectorCCx
yy = mesh_sub.vectorCCy

#%% Solve
txii = range(100,1900,20)
#jx_CC_sub = np.zeros((len(txii),mesh_sub.nCx,mesh_sub.nCy))
#jy_CC_sub = np.zeros((len(txii),mesh_sub.nCx,mesh_sub.nCy))

fig = plt.figure(figsize=(14,7))
axs = plt.axes(ylim=(-800,0), xlim=(25,2000))
im1 = axs.imshow([[],[]], alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)
im2 = axs.imshow([[],[]],alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')
#im3 = axs.streamplot(mesh_sub.vectorCCx, mesh_sub.vectorCCy, np.zeros((mesh_sub.nCy,mesh_sub.nCx)), np.zeros((mesh_sub.nCy,mesh_sub.nCx)),color='k')


def init():
    im1.set_data([[],[]])
    im2.set_data([[],[]])
    return [im1]+[im2]

def animate(ii):
    
    

#for ii in range(len(txii)):
    
    
    
    tx = DC.SrcDipole([rx],txii[ii],txii[ii])
    
    survey = DC.SurveyDC([tx])
    problem = DC.ProblemDC_CC(mesh)
    problem.pair(survey)
    problem.Solver = SolverLU
    
    u1 = problem.fields(model)
    
    Msig1 = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))
    
    j = Msig1*mesh.cellGrad*u1[tx, 'phi_sol']
    j_CC = mesh.aveF2CCV*j
    
    # Compute charge density solving div*grad*phi
    Q = mesh.faceDiv*mesh.cellGrad*u1[tx, 'phi_sol']
    
    jx_CC = j_CC[0:mesh.nC].reshape(mesh.nCy,mesh.nCx).T
    jy_CC = j_CC[mesh.nC:].reshape(mesh.nCy,mesh.nCx).T
    
    #%% Grab only the core for presentation
    jx_CC_sub = jx_CC[padc:-padc,padc:]
    jy_CC_sub = jy_CC[padc:-padc,padc:]
    Q_sub = Q.reshape(mesh.nCy,mesh.nCx)   
    Q_sub = Q_sub[padc:,padc:-padc]
    
    J_rho = np.sqrt(jx_CC_sub**2 + jy_CC_sub**2)
    lw = np.log10(J_rho/J_rho.min())
    
    #axs.imshow(Q_sub,alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)    
    #axs.imshow(np.log10(model_sub.reshape(mesh_sub.nCy,mesh_sub.nCx)),alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')     
    #im3 = axs.streamplot(mesh_sub.vectorCCx, mesh_sub.vectorCCy, jx_CC_sub.T, jy_CC_sub.T,color='k',linewidth = lw.T)

    #plt.show()
    im1.set_array(Q_sub)
    im2.set_array(np.log10(model_sub.reshape(mesh_sub.nCy,mesh_sub.nCx)))
    #im2.set_array(mesh_sub.vectorCCx, mesh_sub.vectorCCy,jx_CC_sub.T,jy_CC_sub.T)
    
    return [im1] + [im2] 
#%% Create widget

#def viewInv(msh,iteration):



#, linewidth=lw.T
#%%   
#interact(viewInv,msh = mesh_sub, iteration = IntSlider(min=0, max=len(txii)-1 ,step=1, value=0))
# set embed_frames=True to embed base64-encoded frames directly in the HTML
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(txii), interval=10)
                               
anim.save('animation.html', writer=HTMLWriter(embed_frames=True))
