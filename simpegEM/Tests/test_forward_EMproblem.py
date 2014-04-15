import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
from simpegEM.Utils.Ana import hzAnalyticDipoleT
import matplotlib.pyplot as plt

import simpegem1d as EM1D

class TDEM_bTests(unittest.TestCase):

    def setUp(self):

        # cs = 20.
        # ncx = 15
        # ncy = 10
        # npad = 15
        # hx = Utils.meshTensors(((0,cs), (ncx,cs), (npad,cs)))
        # hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
        # mesh = Mesh.CylMesh([hx,1,hz], [0,0,-hz.sum()/2])

        cs, nc, npad = 20., 15, 10
        hx = Utils.meshTensors(((npad,cs), (nc,cs), (npad,cs)))
        hy = Utils.meshTensors(((npad,cs), (nc,cs), (npad,cs)))
        hz = Utils.meshTensors(((npad,cs), (nc,cs), (npad,cs)))
        mesh = Mesh.TensorMesh([hx,hy,hz], [-hx.sum()/2.,-hy.sum()/2.,-hz.sum()/2.])

        active = mesh.vectorCCz<0.
        actMap = Maps.ActiveCells(mesh, active, -8, nC=mesh.nCz)
        mapping = Maps.ComboMap(mesh,
                    [Maps.ExpMap, Maps.Vertical1DMap, actMap])


        opts = {'txLoc':np.array([0., 0., 0.]),
                'txType':'VMD_MVP',
                'rxLoc':np.array([[10., 0., 0.]]),
                'rxType':'bz',
                'timeCh':np.logspace(-5,-4, 21),
                }

        sig_half = 1e-3
        self.sig_half = sig_half
        self.dat = EM.TDEM.SurveyTDEM1D(**opts)
        self.prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
        try:
            from mumpsSCI import MumpsSolver
            self.prb.Solver = MumpsSolver
        except ImportError:
            pass
        self.prb.setTimes([1e-6], [100])
        # self.prb.setTimes([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4], [40, 40, 40, 40, 40, 40])

        self.sigma = np.ones(mesh.nCz)*1e-8
        self.sigma[active] = sig_half
        self.sigma = np.log(self.sigma[active])
        self.showIt = True
        self.prb.pair(self.dat)


        TDsurvey = EM1D.BaseEM1D.EM1DSurveyTD()
        TDsurvey.rxLoc = np.array([0., 0., 30.])
        TDsurvey.txLoc = np.array([0., 0., 80.])
        TDsurvey.fieldtype = 'secondary'
        TDsurvey.waveType = 'stepoff'
        TDsurvey.time = self.prb.times #np.logspace(-5, -2, 64)
        TDsurvey.setFrequency(TDsurvey.time)


        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN
        LocSigZ = -mesh1D.gridCC
        nlay = depth.size
        topo = np.r_[0., 0., 0.]
        TDsurvey.depth = depth
        TDsurvey.topo = topo
        TDsurvey.LocSigZ = LocSigZ
        TDsurvey.HalfSwitch = True
        TDsurvey.Setup1Dsystem()

        chi_half = 0.

        expmap = EM1D.BaseEM1D.BaseEM1DMap(mesh1D)
        mappingReal = Maps.ComboMap(mesh1D, [expmap])
        m_1D = np.log(np.ones(nlay)*sig_half)

        TDsurvey.rxType = 'Bz'
        WT0, WT1, YBASE = EM1D.DigFilter.LoadWeights()
        options = {'WT0': WT0, 'WT1': WT1, 'YBASE': YBASE}

        prob = EM1D.EM1D.EM1D(mesh1D, mapping=mappingReal, **options)
        prob.pair(TDsurvey)
        prob.chi = np.zeros(TDsurvey.nlay)

        survey = TDsurvey
        options = options
        prob.CondType = 'Real'
        prob.survey.txType = 'VMD'
        prob.survey.offset = 1e-5

        m_1D = np.log(np.ones(prob.survey.nlay)*sig_half)
        Bz = survey.dpred(m_1D)
        self.Bzanal = Bz



    def test_analitic_b(self):
        bz_calc = self.dat.dpred(self.sigma)
        # bz_ana = self.Bzanal
        bz_ana = mu_0*hzAnalyticDipoleT(self.dat.rxLoc[0,0], self.prb.times, self.sig_half)
        ind = self.prb.times > 1e-5
        diff = np.linalg.norm(bz_calc[ind].flatten() - bz_ana[ind].flatten())/np.linalg.norm(bz_ana[ind].flatten())

        if self.showIt == True:

            plt.loglog(self.prb.times[bz_calc>0], bz_calc[bz_calc>0], 'r', self.prb.times[bz_calc<0], -bz_calc[bz_calc<0], 'r--')
            plt.loglog(self.prb.times, abs(bz_ana), 'b*')
            plt.xlim(1e-5, 1e-2)
            plt.show()

        print 'Difference: ', diff
        self.assertTrue(diff < 0.10)



if __name__ == '__main__':
    unittest.main()
