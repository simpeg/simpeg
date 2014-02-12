from BaseTDEM import ProblemBaseTDEM
from BaseTDEM import FieldsTDEM
import numpy as np

class ProblemTDEM_b(ProblemBaseTDEM):
    """
        docstring for ProblemTDEM_b
    """
    def __init__(self, mesh, model, **kwargs):
        ProblemBaseTDEM.__init__(self, mesh, model, **kwargs)

    solType = 'b'
    
    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, tInd):
        dt = self.getDt(tInd)
        return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui + (1/dt)*self.MfMui

    def getRHS(self, tInd, F):
        dt = self.getDt(tInd)
        return (1/dt)*self.MfMui*F.get_b(tInd-1)


    ####################################################
    # Derivatives
    ####################################################

    def J(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        p = self.G(m, v, u)
        y = self.solveAh(m, p)
        return self.data.projectFields(y)

    def G(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        p = FieldsTDEM(self.mesh, 1, self.times.size, 'b')
        c = self.mesh.getEdgeMassDeriv()*self.model.transformDeriv(m)*v
        for i in range(self.times.size):
            ei = u.get_e(i)
            pVal = np.empty_like(ei)
            for j in range(ei.shape[1]):
                pVal[:,j] = -ei[:,j]*c    
            p.set_e(pVal,i)
        return p

    def solveAh(self, m, p):
        def AhRHS(tInd, u):
            if tInd == 0:
                return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*p.get_e(tInd)
            else:
                dt = self.getDt(tInd)
                return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*p.get_e(tInd) + 1./dt*self.MfMui*u.get_b(tInd-1)

        def AhCalcFields(sol, solType, tInd):
            b = sol
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b - self.MeSigmaI*p.get_e(tInd)
            return {'b':b, 'e':e}

        Y = self.fields(m, useThisRhs=AhRHS, useThisCalcFields=AhCalcFields)
        return Y

if __name__ == '__main__':
    from SimPEG import *
    import simpegEM as EM
    from simpegEM.Utils.Ana import hzAnalyticDipoleT
    from scipy.constants import mu_0
    import matplotlib.pyplot as plt

    cs = 5.
    ncx = 20
    ncy = 6
    npad = 20
    hx = Utils.meshTensors(((0,cs), (ncx,cs), (npad,cs)))
    hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
    mesh = Mesh.Cyl1DMesh([hx,hy], -hy.sum()/2)
    model = Model.Vertical1DModel(mesh)

    opts = {'txLoc':0.,
            'txType':'VMD_MVP',
            'rxLoc':np.r_[150., 0.],
            'rxType':'bz',
            'timeCh':np.logspace(-4,-2,20),
            }
    dat = EM.TDEM.DataTDEM1D(**opts)

    prb = EM.TDEM.ProblemTDEM_b(mesh, model)
    prb.setTimes([1e-5, 5e-5, 2.5e-4], [150, 150, 150])
    sigma = np.ones(mesh.nCz)*1e-8
    sigma[mesh.vectorCCz<0] = 0.1
    prb.pair(dat)
    f = prb.fields(sigma)
    # prb.G(prb.sigma, prb.sigma)
    # prb.solveAh(prb.sigma, f)
    # prb.J(prb.sigma, prb.sigma, f)

    from SimPEG.Tests import checkDerivative
    m0 = sigma
    dx = np.zeros_like(sigma)
    dx[prb.mesh.vectorCCz<0] = 1e-4
    derChk = lambda m: [dat.dpred(m), lambda mx: prb.J(m0, mx, u=f)]
    passed = checkDerivative(derChk, m0, dx=dx, plotIt=False)

    # bz_calc = dat.dpred(sigma)
    # bz_ana = mu_0*hzAnalyticDipoleT(dat.rxLoc[0], prb.times, sigma[0])

    # plt.loglog(prb.times, np.abs(bz_calc.flatten()), label='TDEM_b')
    # plt.loglog(prb.times, np.abs(bz_ana), 'r', label='Analytic')
    # plt.legend()
    # plt.show()


