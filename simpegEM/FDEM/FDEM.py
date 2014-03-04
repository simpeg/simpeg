from SimPEG import Problem, Solver, np, sp
from scipy.constants import mu_0
from SimPEG.Utils import sdiag, mkvc
from FieldsFDEM import FieldsFDEM
from DataFDEM import DataFDEM


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
    storeTheseFields = 'e'

    dataPair = DataFDEM

    solveOpts = {'factorize':False, 'backend':'scipy'}

    j_s = None


    def getFieldsObject(self):
        F = FieldsFDEM(self.mesh, self.data.nTx, self.data.nFreq, store=self.storeTheseFields)
        return F

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
        #TODO: hardcoded to sigma as the model
        sigma = self.model.transform(m)
        self._Me = self.mesh.getEdgeInnerProduct()
        self._MeSigma = self.mesh.getEdgeInnerProduct(sigma)
        # TODO: this will not work if tensor conductivity
        self._MeSigmaI = sdiag(1/self.MeSigma.diagonal())
        #TODO: assuming constant mu
        self._MfMui = self.mesh.getFaceInnerProduct(1/mu_0*np.ones(self.mesh.nC))

    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, freqInd):
        """
            :param int fInd: Frequency index
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        omega = self.data.omega[freqInd]
        return self.mesh.edgeCurl.T*self.MfMui*self.mesh.edgeCurl + 1j*omega*self.MeSigma

    def getRHS(self, freqInd):
        omega = self.data.omega[freqInd]
        return -1j*omega*self.Me*self.j_s


    def fields(self, m, useThisRhs=None):
        RHS = useThisRhs or self.getRHS

        self.makeMassMatrices(m)

        F = self.getFieldsObject()

        for freqInd in range(self.data.nFreq):
            A = self.getA(freqInd)
            b = self.getRHS(freqInd)
            e = Solver(A, options=self.solveOpts).solve(b)

            F.set_e(e, freqInd)
            omega = self.data.omega[freqInd]
            #TODO: check if mass matrices needed:
            b = -1./(1j*omega)*self.mesh.edgeCurl*e
            F.set_b(b, freqInd)

        return F


    def Jvec(self, m, v, u=None):
        # TODO: only 1 transmitter for now
        # TODO: all P's the same
        if u is None:
           u = self.fields(m)

        Jvs = range(self.data.nFreq)
        P  = self.data.projectFieldsDeriv(u)

        for i, freqInd in enumerate(range(self.data.nFreq)):
            e = u.get_e(freqInd)
            omega = self.data.omega[freqInd]
            for txInd in self.data.nTx
            b = 1j*omega*self.mesh.getEdgeInnerProductDeriv(m,v=e)*self.model.transformDeriv(m)*v
            A = self.getA(freqInd)
            Ab = Solver(A, options=self.solveOpts).solve(b)
            Jvs[i] = -P*Ab

        Jv = np.concatenate(Jvs)

        return Jv


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
    ncx = 6
    ncy = 6
    ncz = 6
    npad = 3
    hx = Utils.meshTensors(((npad,cs), (ncx,cs), (npad,cs)))
    hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
    hz = Utils.meshTensors(((npad,cs), (ncz,cs), (npad,cs)))
    mesh = Mesh.TensorMesh([hx,hy,hz])

    XY = Utils.ndgrid(np.linspace(20,50,3), np.linspace(20,50,3))
    rxLoc = np.c_[XY, np.ones(XY.shape[0])*40]

    model = Model.LogModel(mesh)

    opts = {'txLoc':0.,
        'txType':'VMD_MVP',
        'rxLoc': rxLoc,
        'rxType':'bz',
        'freq': np.logspace(0,3,4),
        }
    dat = EM.FDEM.DataFDEM(**opts)

    prb = EM.FDEM.ProblemFDEM_e(mesh, model)
    prb.pair(dat)

    sigma = np.log(np.ones(mesh.nC)*1e-3)

    j_sx = np.zeros(mesh.vnEx)
    j_sx[6,6,6] = 1
    j_s = np.r_[Utils.mkvc(j_sx),np.zeros(mesh.nEy+mesh.nEz)]

    prb.j_s = j_s
    f = prb.fields(sigma)

    colorbar(mesh.plotSlice((f.get_e(3)), 'E', ind=11, normal='Z', view='real')[0])
    plt.show()





