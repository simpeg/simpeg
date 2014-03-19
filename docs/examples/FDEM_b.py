import sys
sys.path.append('../../')

from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
import matplotlib.pyplot as plt

cs = 10.
ncx, ncy, ncz = 8, 8, 8
npad = 5
hx = Utils.meshTensors(((npad,cs), (ncx,cs), (npad,cs)))
hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
hz = Utils.meshTensors(((npad,cs), (ncz,cs), (npad,cs)))
mesh = Mesh.TensorMesh([hx,hy,hz], x0=[-hx.sum()/2.,-hy.sum()/2.,-hz.sum()/2.,])

model = Model.LogModel(mesh)

x = np.linspace(-10,10,5)
XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])
rxList = EM.FDEM.RxListFDEM(XYZ, 'Ex')
Tx0 = EM.FDEM.TxFDEM(np.r_[0.,0.,0.], 'VMD', 1e2, rxList)

survey = EM.FDEM.SurveyFDEM([Tx0])

prb = EM.FDEM.ProblemFDEM_b(model)
prb.pair(survey)

sig = 1e-1
sigma = np.ones(mesh.nC)*sig
sigma[mesh.gridCC[:,2] > 0] = 1e-8
m = np.log(sigma)

skin = 500*np.sqrt(1/(sig*Tx0.freq))
print 'The skin depth is: %4.2f m' % skin

prb.Solver = Utils.SolverUtils.DSolverWrap(sp.linalg.spsolve, factorize=False, checkAccuracy=True)

u = prb.fields(m)

plt.colorbar(mesh.plotImage(np.log10(np.abs(u[Tx0, 'b'].real)), 'Fz'))

bfz = mesh.r(u[Tx0, 'b'],'F','Fz','M')

x = np.linspace(-55,55,12)
XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])

P = mesh.getInterpolationMat(XYZ, 'Fz')

an = EM.Utils.Ana.FEM.hzAnalyticDipoleF(x, Tx0.freq, sig)

plt.figure(2)
plt.plot(x,np.log10(np.abs(P*np.imag(u[Tx0, 'b']))))
plt.plot(x,np.log10(np.abs(mu_0*np.imag(an))), 'r')
plt.xlabel('Distance, m')
plt.ylabel('Log10 Response imag($B_z$)')
plt.legend(('Numeric','Analytic'))
plt.title('Half Space Response for FDEM')

plt.show()
