from SimPEG import *
import simpegEM as EM
# from simpegem1d import Utils1D
from scipy.constants import mu_0
import matplotlib.pyplot as plt

class TDEMinversion(object):
    """ Wrapper for TDEMinversion """
    opt = None
    survey = None
    prb = None
    obj = None
    regmesh = None
    m0 = None
    inv = None
    surveyinfo = None
    probleminfo = None

    def __init__(self, regmesh, m0, **kwargs):

        self.regmesh = regmesh
        self.m0 = m0

    def setSurveyProb(self, **kwargs):
        self.surveyinfo = kwargs['surveyinfo']
        self.probleminfo = kwargs['probleminfo']
        rx = self.surveyinfo['rx']
        tx = self.surveyinfo['tx']
        mesh = self.probleminfo['mesh']
        mapping = self.probleminfo['mapping']
        timeSteps = self.probleminfo['timeSteps']
        self.survey = EM.TDEM.SurveyTDEM([tx])
        self.prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
        self.prb.pair(self.survey)
        self.prb.Solver = self.probleminfo['Solver']
        self.prb.timeSteps = timeSteps

    def setInv(self, **kwargs):

        self.opt = Optimization.InexactGaussNewton(**kwargs['opt'])
        self.beta = Parameters.BetaSchedule(**kwargs['beta'])
        self.reg = Regularization.Tikhonov(self.regmesh, **kwargs['reg'])
        self.obj = ObjFunction.BaseObjFunction(self.survey, self.reg, beta=self.beta)
        self.inv = Inversion.BaseInversion(self.obj, self.opt)

    def setDobs(self, dobs, std, floor):

        self.survey.dobs = dobs
        self.survey.std = std
        self.survey.floor = floor
        self.survey.Wd = 1/(abs(dobs)*std+floor)

    def run(self):
        C =  Utils.Counter()
        self.prb.counter = C
        self.opt.counter = C
        self.opt.LSshorten = 0.5
        self.opt.remember('xc')

        return self.inv.run(self.m0)

if __name__ == '__main__':

    cs, ncx, ncz, npad = 5., 25, 15, 15
    hx = [(cs,ncx), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')

    active = mesh.vectorCCz<0.
    layer = (mesh.vectorCCz<0.) & (mesh.vectorCCz>=-100.)
    actMap = Maps.ActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ComboMap(mesh, [Maps.ExpMap, Maps.Vertical1DMap, actMap])
    sig_half = 2e-3
    sig_air = 1e-8
    sig_layer = 1e-3
    sigma = np.ones(mesh.nCz)*sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer
    mtrue = np.log(sigma[active])

    rxOffset=1e-3
    rx = EM.TDEM.RxTDEM(np.array([[rxOffset, 0., 30]]), np.logspace(-5,-3, 31), 'bz')
    tx = EM.TDEM.TxTDEM(np.array([0., 0., 80]), 'VMD_MVP', [rx])
    survey = EM.TDEM.SurveyTDEM([tx])
    prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
    prb.Solver = SolverLU
    prb.timeSteps = [(1e-06, 20), (1e-05, 20), (0.0001, 20)]
    prb.pair(survey)
    dtrue = survey.dpred(mtrue)

    alpha_s = 1e-2
    alpha_x = 1

    surveyinfo = {'rx':rx, 'tx':tx}
    prbinfo = {'mesh': mesh, 'mapping': mapping, 'timeSteps':prb.timeSteps, 'Solver':prb.Solver}
    optinfo = {'maxIter':10}
    reginfo = {'alpha_s': alpha_s, 'alpha_x': alpha_x}
    betainfo = {'coolingFactor':5, 'coolingRate':2, 'beta0_ratio': 1e0}
    Invoptions = {'opt': optinfo, 'beta': betainfo, 'reg': reginfo}
    SurvProboptions = {'surveyinfo': surveyinfo, 'probleminfo': prbinfo}
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])

    m0 = np.log(np.ones(mtrue.size)*sig_half)

    std = 0.05
    floor = np.linalg.norm(dtrue)*1e-5
    noise = std*abs(dtrue)*np.random.randn(*dtrue.shape)+floor
    dobs = dtrue+noise

    TDEMinversion = TDEMinversion(regMesh, m0)
    TDEMinversion.setSurveyProb(**SurvProboptions)
    TDEMinversion.setInv(**Invoptions)
    TDEMinversion.setDobs(dobs, std, floor)

    mopt = TDEMinversion.run()

    plt.semilogx(sigma[active], mesh.vectorCCz[active], 'b.-')
    plt.semilogx(np.exp(mopt), mesh.vectorCCz[active], 'r.-')
    plt.xlabel('Conductivity (S/m)', fontsize = 14)
    plt.ylim(-600, 0)
    plt.xlim(5e-4, 1e-2)
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.legend(('True', 'Pred'), loc=1, fontsize = 14)
    plt.show()
