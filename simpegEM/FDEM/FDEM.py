from SimPEG import Problem, Solver, Utils, np, sp
from scipy.constants import mu_0
from SurveyFDEM import SurveyFDEM, DataFDEM, FieldsFDEM

def omega(freq):
    """Change frequency to angular frequency, omega"""
    return 2.*np.pi*freq

class ProblemFDEM_e(Problem.BaseProblem):
    """
        Frequency-Domain EM problem - E-formulation


        .. math::

            \dcurl E + i \omega B = 0 \\\\
            \dcurl^\\top \MfMui B - \MeSig E = \Me \j_s
    """
    def __init__(self, model, **kwargs):
        Problem.BaseProblem.__init__(self, model, **kwargs)

    solType = 'b'
    storeTheseFields = 'e'

    surveyPair = SurveyFDEM
    dataPair = DataFDEM

    solveOpts = {'factorize':False, 'backend':'scipy'}


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
        self._MeSigmaI = Utils.sdiag(1/self.MeSigma.diagonal())
        #TODO: assuming constant mu
        self._MfMui = self.mesh.getFaceInnerProduct(1/mu_0)

    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, freq):
        """
            :param int fInd: Frequency index
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        return self.mesh.edgeCurl.T*self.MfMui*self.mesh.edgeCurl + 1j*omega(freq)*self.MeSigma

    def getRHS(self, freq):
        #TODO: this needs to also depend on your transmitter!

        return -1j*omega(freq)*self.Me*self.j_s


    def fields(self, m, useThisRhs=None):
        RHS = useThisRhs or self.getRHS

        self.makeMassMatrices(m)

        F = FieldsFDEM(self.mesh, self.survey)

        for freq in self.survey.freqs:
            A = self.getA(freq)
            b = self.getRHS(freq)
            e = Solver(A, options=self.solveOpts).solve(b)

            F[freq, 'e'] = e
            #TODO: check if mass matrices needed:
            b = -1./(1j*omega(freq))*self.mesh.edgeCurl*e
            F[freq, 'b'] = b

        return F


    def Jvec(self, m, v, u=None):
        if u is None:
           u = self.fields(m)

        Jv = self.dataPair(self.survey)

        for i, freq in enumerate(self.survey.freqs):
            e = u[freq, 'e']
            A = self.getA(freq)
            solver = Solver(A, options=self.solveOpts)

            for tx in self.survey.getTransmitters(freq):
                dMe_dsig = self.mesh.getEdgeInnerProductDeriv(m, v=e)
                dsig_dm = self.model.transformDeriv(m)

                b = 1j*omega(freq) * ( dMe_dsig * ( dsig_dm * v ) )
                Ab = solver.solve(b)

                P  = tx.projectFieldsDeriv(self.mesh, u)

                Jv[tx] = -P*Ab

        return Utils.mkvc(Jv)


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
    survey = EM.FDEM.SurveyFDEM(**opts)

    prb = EM.FDEM.ProblemFDEM_e(mesh, model)
    prb.pair(survey)

    sigma = np.log(np.ones(mesh.nC)*1e-3)

    j_sx = np.zeros(mesh.vnEx)
    j_sx[6,6,6] = 1
    j_s = np.r_[Utils.mkvc(j_sx),np.zeros(mesh.nEy+mesh.nEz)]

    prb.j_s = j_s
    f = prb.fields(sigma)

    plt.colorbar(mesh.plotSlice((f.get_e(3)), 'E', ind=11, normal='Z', view='real')[0])
    plt.show()





