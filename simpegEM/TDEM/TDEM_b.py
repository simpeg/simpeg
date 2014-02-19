from BaseTDEM import ProblemBaseTDEM
from FieldsTDEM import FieldsTDEM
from SimPEG.Utils import mkvc
import numpy as np

class ProblemTDEM_b(ProblemBaseTDEM):
    """
        Time-Domain EM problem - B-formulation


        .. math::

            \dcurl \e^{(t+1)} + \\frac{\\b^{(t+1)} - \\b^{(t)}}{\delta t} = 0 \\\\
            \dcurl^\\top \MfMui \\b^{(t+1)} - \MeSig \e^{(t+1)} = \Me \j_s^{(t+1)}
    """
    def __init__(self, mesh, model, **kwargs):
        ProblemBaseTDEM.__init__(self, mesh, model, **kwargs)

    solType = 'b'

    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, tInd):
        """
            :param int tInd: Time index
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """

        dt = self.getDt(tInd)
        return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui + (1/dt)*self.MfMui

    def getRHS(self, tInd, F):
        dt = self.getDt(tInd)
        return (1/dt)*self.MfMui*F.get_b(tInd-1)


    ####################################################
    # Derivatives
    ####################################################

    def Jvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        p = self.Gvec(m, v, u)
        y = self.solveAh(m, p)
        return self.data.dpred(m, u=y)

    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        p = self.data.projectFieldsAdjoint(v)
        y = self.solveAht(m, p)
        w = self.Gtvec(m, y, u)
        return w

    def Gvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        p = FieldsTDEM(self.mesh, 1, self.times.size, 'b')
        c = self.mesh.getEdgeMassDeriv()*self.model.transformDeriv(None)*v
        for i in range(self.times.size):
            ei = u.get_e(i)
            pVal = np.empty_like(ei)
            for j in range(ei.shape[1]):
                pVal[:,j] = -ei[:,j]*c

            p.set_e(pVal,i)
            p.set_b(np.zeros((self.mesh.nF,1)), i)
        return p

    def Gtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        tmp = np.zeros((self.mesh.nE,self.data.nTx))
        for i in range(self.nTimes):
            tmp += v.get_e(i)*u.get_e(i)
        p = -mkvc(self.model.transformDeriv(None).T*self.mesh.getEdgeMassDeriv().T*tmp)
        return p


    def solveAh(self, m, p):
        def AhRHS(tInd, u):
            rhs = self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*p.get_e(tInd) + p.get_b(tInd)
            if tInd == 0:
                return rhs
            dt = self.getDt(tInd)
            return rhs + 1./dt*self.MfMui*u.get_b(tInd-1)

        def AhCalcFields(sol, solType, tInd):
            b = sol
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b - self.MeSigmaI*p.get_e(tInd)
            return {'b':b, 'e':e}

        self.makeMassMatrices(m)
        return self.forward(m, AhRHS, AhCalcFields)

    def solveAht(self, m, p):
        
        def AhtRHS(tInd, u):
            rhs = self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*p.get_e(tInd) + p.get_b(tInd)
            if tInd == self.nTimes-1:
                return rhs
            dt = self.getDt(tInd+1)
            return rhs + 1./dt*self.MfMui*u.get_b(tInd+1)
        
        def AhtCalcFields(sol, solType, tInd):
            b = sol
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b - self.MeSigmaI*p.get_e(tInd)
            return {'b':b, 'e':e}

        self.makeMassMatrices(m)
        return self.adjoint(m, AhtRHS, AhtCalcFields)

    ####################################################
    # Functions for tests
    ####################################################

    def AhVec(self, m, u=None):
        if u is None:
            u = self.fields(m)
        self.makeMassMatrices(m)
        dt = self.getDt(0)
        b = 1/dt*self.MfMui*u.get_b(0) + self.MfMui*self.mesh.edgeCurl*u.get_e(0)
        e = self.mesh.edgeCurl.T*self.MfMui*u.get_b(0) - self.MeSigma*u.get_e(0)
        f = FieldsTDEM(self.mesh, 1, self.times.size, 'b')
        f.set_b(b, 0)
        f.set_e(e, 0)
        for i in range(1,self.nTimes):
            dt = self.getDt(i)
            b = 1/dt*self.MfMui*u.get_b(i) + self.MfMui*self.mesh.edgeCurl*u.get_e(i) - 1/dt*self.MfMui*u.get_b(i-1)
            e = self.mesh.edgeCurl.T*self.MfMui*u.get_b(i) - self.MeSigma*u.get_e(i)
            f.set_b(b, i)
            f.set_e(e, i)
        return f

    def AhtVec(self, m, u=None):
        if u is None:
            u = self.fields(m)
        self.makeMassMatrices(m)
        f = FieldsTDEM(self.mesh, 1, self.times.size, 'b')
        for i in range(self.nTimes-1):
            b = 1/self.getDt(i)*self.MfMui*u.get_b(i) + self.MfMui*self.mesh.edgeCurl*u.get_e(i) - 1/self.getDt(i+1)*self.MfMui*u.get_b(i+1)
            e = self.mesh.edgeCurl.T*self.MfMui*u.get_b(i) - self.MeSigma*u.get_e(i)
            f.set_b(b, i)
            f.set_e(e, i)
        N = self.nTimes - 1
        b = 1/self.getDt(N)*self.MfMui*u.get_b(N) + self.MfMui*self.mesh.edgeCurl*u.get_e(N)
        e = self.mesh.edgeCurl.T*self.MfMui*u.get_b(N) - self.MeSigma*u.get_e(N)
        f.set_b(b, N)
        f.set_e(e, N)
        return f



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
    # prb.setTimes([1e-5, 5e-5, 2.5e-4], [150, 150, 150])
    # prb.setTimes([1e-5, 5e-5, 2.5e-4], [10, 10, 10])
    prb.setTimes([1e-5], [1])
    prb.pair(dat)
    sigma = np.random.rand(mesh.nCz)




