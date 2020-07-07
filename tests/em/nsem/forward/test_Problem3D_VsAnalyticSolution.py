from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import unittest
from scipy.constants import mu_0

import numpy as np
from SimPEG.electromagnetics import natural_source as nsem


np.random.seed(1100)

TOLr = 1
TOLp = 2
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0
freq = [1e-1, 2e-1]
addrandoms = True


def appResPhsHalfspace_eFrom_ps_Norm(sigmaHalf, appR=True, expMap=False):
    if appR:
        label = "resistivity"
    else:
        label = "phase"
    print(
        "Apparent {:s} test of eFormulation primary/secondary at {:g}\n\n".format(
            label, sigmaHalf
        )
    )

    # Calculate the app  phs
    survey, simulation = nsem.utils.test_utils.setupSimpegNSEM_ePrimSec(
        nsem.utils.test_utils.halfSpace(sigmaHalf), expMap=expMap
    )
    data = nsem.Data(survey=survey, dobs=simulation.dpred(simulation.model))
    recData = data.toRecArray("Complex")
    app_rpxy = nsem.utils.appResPhs(recData["freq"], recData["zxy"])[0]
    # app_rpyx = nsem.utils.appResPhs(recData['freq'], recData['zyx'])[0]
    if appR:
        return np.linalg.norm(
            np.abs(np.log10(app_rpxy[0]) - np.log10(1.0 / sigmaHalf))
            * np.log10(sigmaHalf)
        )
    else:
        return np.linalg.norm(np.abs(app_rpxy[1] + 135.0) / 135.0)


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        # Make the survey and the problem
        pass

    # # Test apparent resistivity and phase
    def test_appRes1en2(self):
        self.assertLess(appResPhsHalfspace_eFrom_ps_Norm(1e-2), TOLr)

    def test_appPhs1en2(self):
        self.assertLess(appResPhsHalfspace_eFrom_ps_Norm(1e-2, False), TOLp)

    def test_appRes1en1(self):
        self.assertLess(appResPhsHalfspace_eFrom_ps_Norm(1e-1), TOLr)

    def test_appPhs1en1(self):
        self.assertLess(appResPhsHalfspace_eFrom_ps_Norm(1e-1, False), TOLp)


if __name__ == "__main__":
    unittest.main()
