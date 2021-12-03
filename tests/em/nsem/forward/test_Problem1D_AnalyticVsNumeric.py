from __future__ import print_function
import unittest
from SimPEG import mkvc
from SimPEG.electromagnetics import natural_source as nsem
import numpy as np

# Define the tolerances
TOLr = 5e-2
TOLp = 5e-2


def getAppResPhs(nsemdata):
    # Make impedance
    def appResPhs(freq, z):
        app_res = ((1.0 / (8e-7 * np.pi ** 2)) / freq) * np.abs(z) ** 2
        app_phs = np.arctan2(z.imag, z.real) * (180 / np.pi)
        return app_res, app_phs

    zList = []
    for src in nsemdata.survey.source_list:
        zc = [src.frequency]
        for rx in src.receiver_list:
            if "i" in rx.rxType:
                m = 1j
            else:
                m = 1
            zc.append(m * nsemdata[src, rx])
        zList.append(zc)
    return [
        appResPhs(zList[i][0], np.sum(zList[i][1:3])) for i in np.arange(len(zList))
    ]


def calculateAnalyticSolution(source_list, mesh, model):
    surveyAna = nsem.Survey(source_list)
    data1D = nsem.Data(surveyAna)
    for src in surveyAna.source_list:
        elev = src.receiver_list[0].locations[0]
        anaEd, anaEu, anaHd, anaHu = nsem.utils.analytic_1d.getEHfields(
            mesh, model, src.frequency, elev
        )
        anaE = anaEd + anaEu
        anaH = anaHd + anaHu
        # Scale the solution
        # anaE = (anaEtemp/anaEtemp[-1])#.conj()
        # anaH = (anaHtemp/anaEtemp[-1])#.conj()
        anaZ = anaE / anaH
        for rx in src.receiver_list:
            data1D[src, rx] = getattr(anaZ, rx.component)
    return data1D


def dataMis_AnalyticPrimarySecondary(sigmaHalf):

    # Make the survey
    # Primary secondary
    survey, sig, sigBG, mesh = nsem.utils.test_utils.setup1DSurvey(
        sigmaHalf, False, structure=True
    )
    # Analytic data
    simulation = nsem.Simulation1DPrimarySecondary(
        mesh, sigmaPrimary=sig, sigma=sig, survey=survey
    )

    dataAnaObj = calculateAnalyticSolution(survey.source_list, mesh, sig)

    data = simulation.dpred()
    dataAna = mkvc(dataAnaObj)
    return np.all((data - dataAna) / dataAna < 2.0)


class TestNumericVsAnalytics(unittest.TestCase):
    def setUp(self):
        pass

    # Primary/secondary
    def test_appRes2en2_ps(self):
        self.assertTrue(dataMis_AnalyticPrimarySecondary(2e-2))


if __name__ == "__main__":
    unittest.main()
