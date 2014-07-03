import sys
sys.path.append('../../')

from SimPEG import *
from simpegFLOW import Richards
import matplotlib.pyplot as plt


M = Mesh.TensorMesh([np.ones(40)])
M.setCellGradBC('dirichlet')
params = Richards.Empirical.HaverkampParams().celia1990
params['Ks'] = np.log(params['Ks'])
E = Richards.Empirical.Haverkamp(M, **params)

bc = np.array([-61.5,-20.7])
h = np.zeros(M.nC) + bc[0]


def getFields(timeStep,method):
    timeSteps = np.ones(360/timeStep)*timeStep
    prob = Richards.RichardsProblem(M, mapping=E, timeSteps=timeSteps,
                                    boundaryConditions=bc, initialConditions=h,
                                    doNewton=False, method=method)
    return prob.fields(params['Ks'])

Hs_M10 = getFields(10., 'mixed')
Hs_M30 = getFields(30., 'mixed')
Hs_M120= getFields(120.,'mixed')
Hs_H10 = getFields(10., 'head')
Hs_H30 = getFields(30., 'head')
Hs_H120= getFields(120.,'head')

plt.figure(figsize=(13,5))
plt.subplot(121)
plt.plot(40-M.gridCC, Hs_M10[-1],'b-')
plt.plot(40-M.gridCC, Hs_M30[-1],'r-')
plt.plot(40-M.gridCC, Hs_M120[-1],'k-')
plt.ylim([-70,-10])
plt.title('Mixed Method')
plt.xlabel('Depth, cm')
plt.ylabel('Pressure Head, cm')
plt.legend(('$\Delta t$ = 10 sec','$\Delta t$ = 30 sec','$\Delta t$ = 120 sec'))
plt.subplot(122)
plt.plot(40-M.gridCC, Hs_H10[-1],'b-')
plt.plot(40-M.gridCC, Hs_H30[-1],'r-')
plt.plot(40-M.gridCC, Hs_H120[-1],'k-')
plt.ylim([-70,-10])
plt.title('Head-Based Method')
plt.xlabel('Depth, cm')
plt.ylabel('Pressure Head, cm')
plt.legend(('$\Delta t$ = 10 sec','$\Delta t$ = 30 sec','$\Delta t$ = 120 sec'))
plt.show()
