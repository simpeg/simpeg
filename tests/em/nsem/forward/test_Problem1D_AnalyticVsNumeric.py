from __future__ import print_function
import unittest
import SimPEG as simpeg
from SimPEG.EM import NSEM
import numpy as np
# Define the tolerances
TOLr = 5e-2
TOLp = 5e-2


def getAppResPhs(NSEMdata):
    # Make impedance
    def appResPhs(freq, z):
        app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
        app_phs = np.arctan2(z.imag, z.real)*(180/np.pi)
        return app_res, app_phs
    zList = []
    for src in NSEMdata.survey.srcList:
        zc = [src.freq]
        for rx in src.rxList:
            if 'i' in rx.rxType:
                m = 1j
            else:
                m = 1
            zc.append(m*NSEMdata[src, rx])
        zList.append(zc)
    return [
        appResPhs(zList[i][0], np.sum(zList[i][1:3]))
        for i in np.arange(len(zList))
    ]


def calculateAnalyticSolution(srcList, mesh, model):
    surveyAna = NSEM.Survey(srcList)
    data1D = NSEM.Data(surveyAna)
    for src in surveyAna.srcList:
        elev = src.rxList[0].locs[0]
        anaEd, anaEu, anaHd, anaHu = NSEM.Utils.MT1Danalytic.getEHfields(
            mesh, model, src.freq, elev
        )
        anaE = anaEd+anaEu
        anaH = anaHd+anaHu
        # Scale the solution
        # anaE = (anaEtemp/anaEtemp[-1])#.conj()
        # anaH = (anaHtemp/anaEtemp[-1])#.conj()
        anaZ = anaE/anaH
        for rx in src.rxList:
            data1D[src, rx] = getattr(anaZ, rx.component)
    return data1D


def dataMis_AnalyticPrimarySecondary(sigmaHalf):

    # Make the survey
    # Primary secondary
    survey, sig, sigBG, mesh = NSEM.Utils.testUtils.setup1DSurvey(
        sigmaHalf, False, structure=True
    )
    # Analytic data
    problem = NSEM.Problem1D_ePrimSec(mesh, sigmaPrimary=sig, sigma=sig)
    problem.pair(survey)

    dataAnaObj = calculateAnalyticSolution(survey.srcList, mesh, sig)

    data = survey.dpred()
    dataAna = simpeg.mkvc(dataAnaObj)
    return np.all((data - dataAna)/dataAna < 2.)


class TestNumericVsAnalytics(unittest.TestCase):

    def setUp(self):
        pass

    # Primary/secondary
    def test_appRes2en2_ps(self):
        self.assertTrue(dataMis_AnalyticPrimarySecondary(2e-2))

if __name__ == '__main__':
    unittest.main()
