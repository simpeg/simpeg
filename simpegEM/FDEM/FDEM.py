from SimPEG import Problem
import numpy as np
from scipy.constants import mu_0
from SimPEG.Utils import sdiag, mkvc


class ProblemFDEM_e(Problem.BaseProblem):
    """
        Frequency-Domain EM problem - E-formulation


        .. math::

            \dcurl E + i \omega B = 0 \\\\
            \dcurl^\\top \MfMui B - \MeSig E = \Me \j_s
    """
    def __init__(self, mesh, model, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, model, **kwargs)

    solType = 'b'

    #TODO:
    # j_s
    # getOmega
    # getFieldsObject

    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MfMui(self): return self._MfMui

    @property
    def Me(self): return self._Me

    @property
    def MeSigma(self): return self._MeSigma

    @property
    def MeSigmaI(self): return self._MeSigmaI

    def makeMassMatrices(self, m):
        self._Me = self.mesh.getEdgeInnerProduct()
        self._MeSigma = self.mesh.getEdgeInnerProduct(m)
        # TODO: this will not work if tensor conductivity
        self._MeSigmaI = sdiag(1/self.MeSigma.diagonal())
        #TODO: assuming constant mu
        self._MfMui = self.mesh.getFaceInnerProduct(1/mu_0)

    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, omegaInd):
        """
            :param int tInd: Time index
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        omega = self.getOmega(omegaInd)
        return self.mesh.edgeCurl.T*self.MfMui*self.mesh.edgeCurl + 1j*omega*self.MeSigma

    def getRHS(self, omegaInd):
        omega = self.getOmega(omegaInd)
        return -1j*omega*self.Me*self.j_s


    def fields(self, m, useThisRhs=None):
        RHS = useThisRhs or self.getRHS

        self.makeMassMatrices(m)

        F = self.getFieldsObject()


        return


    def Jvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        raise NotImplementedError('Jvec todo!')


    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)
        raise NotImplementedError('Jtvec todo!')




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




