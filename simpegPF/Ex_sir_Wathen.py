from SimPEG import *
import matplotlib.pyplot as plt
from simpegPF.MagAnalytics import spheremodel, MagSphereAnaFun, CongruousMagBC
import time

# Step1: Generate 3D tensor mesh
hxind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
hyind = ((5,25,1.3),(41, 12.5),(5,25,1.3))
hzind = ((5,25,1.3),(40, 12.5),(5,25,1.3))
hx, hy, hz = Utils.meshTensors(hxind, hyind, hzind)
M3 = Mesh.TensorMesh([hx, hy, hz], [-sum(hx)/2,-sum(hy)/2,-sum(hz)/2])

# Step2: Generate susceptibility model
mu0 = 4*np.pi*1e-7
chibkg = 0.
chiblk = 1.
chi = np.ones(M3.nC)*chibkg

sph_ind = spheremodel(M3, 0., 0., 0., 50)
chi[sph_ind] = chiblk
mu = (1.+chi)*mu0
# Step3: Generate Boundary
BC = [['neumann', 'neumann'], ['neumann', 'neumann'], ['neumann', 'neumann']]
Box = 1 # Primary field in x-direction (background)
Boy = 0 # Primary field in y-direction (background)
Boz = 0 # Primary field in z-direction (background)
B0 = np.r_[Box*np.ones(np.prod(M3.nFx)), Boy*np.ones(np.prod(M3.nFy)), Boz*np.ones(np.prod(M3.nFz))]

Bbc, Bbcderiv = CongruousMagBC(M3, np.array([Box, Boy, Boz]), chi)

# Step4: Compute system matrix and right hand side
Dface = M3.faceDiv
Pbc, Pin, Pout = M3.getBCProjWF(BC, discretization='CC')
Mc = Utils.sdiag(M3.vol)
Div = Mc*Dface*Pin.T*Pin
MfmuI = Utils.sdiag(1/M3.getFaceInnerProduct(1/mu).diagonal())
Mfmu0 = 1/mu0*M3.getFaceInnerProduct()
A = -Div*MfmuI*Div.T
rhs = -Div*MfmuI*Mfmu0*B0 + Div*B0 - Mc*Dface*Pout.T*Bbc

# Step5: Solve !!
start = time.clock()
m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(-1/A.diagonal()))
phi, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter = 1000, M =m1)
elapsed = (time.clock() - start)

print np.linalg.norm(A*phi-rhs)/np.linalg.norm(rhs)
print ('Cpu time = %10.3e ms') % (elapsed*1000)
# Step6: Update for B and project to receiver locations
B = MfmuI*Mfmu0*B0-B0-MfmuI*Div.T*phi

xr = np.linspace(-300, 300, 41)
yr = np.linspace(-300, 300, 41)
X, Y = np.meshgrid(xr, yr)
Z = np.ones((np.size(xr), np.size(yr)))*80

rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
Qfx = M3.getInterpolationMat(rxLoc,'Fx')
Qfy = M3.getInterpolationMat(rxLoc,'Fy')
Qfz = M3.getInterpolationMat(rxLoc,'Fz')

Bxr = np.reshape(Qfx*B, (np.size(xr), np.size(yr)), order='F')
Byr = np.reshape(Qfy*B, (np.size(xr), np.size(yr)), order='F')
Bzr = np.reshape(Qfz*B, (np.size(xr), np.size(yr)), order='F')
H0 = Box/mu0

flag = 'secondary'

Bxra, Byra, Bzra = MagSphereAnaFun(X, Y, Z, 50., 0., 0., 0., mu0, mu0*(1+chiblk), H0, flag)
Bxra = np.reshape(Bxra, (np.size(xr), np.size(yr)), order='F')
Byra = np.reshape(Byra, (np.size(xr), np.size(yr)), order='F')
Bzra = np.reshape(Bzra, (np.size(xr), np.size(yr)), order='F')

# Step6: comparison to analytics
fig, ax = plt.subplots(3,2, figsize = (10,15))
dat1 = ax[0,0].imshow(Bxr); fig.colorbar(dat1, ax=ax[0,0])
dat2 = ax[0,1].imshow(Bxra); fig.colorbar(dat2, ax=ax[0,1])
dat3 = ax[1,0].imshow(Byr); fig.colorbar(dat3, ax=ax[1,0])
dat4 = ax[1,1].imshow(Byra); fig.colorbar(dat4, ax=ax[1,1])
dat5 = ax[2,0].imshow(Bzr); fig.colorbar(dat5, ax=ax[2,0])
dat6 = ax[2,1].imshow(Bzra); fig.colorbar(dat6, ax=ax[2,1])
plt.show()