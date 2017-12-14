from __future__ import print_function
import unittest
import numpy as np
from SimPEG.EM.Static import DC, Utils as DCUtils


class DCUtilsTests(unittest.TestCase):

    def setup(self):
        DCsurvey = DCUtils.readUBC_DC3Dobs('./benchmark_files/dPred_Fullspace.txt')
        DCsurvey = dpred['DCsurvey']
        self.survey = DC.survey

    def test_ElecSep(self):

        # Compute dipoles distances from survey
        AB, MN, AM, AN, BM, BN = DCUtils.calc_ElecSep(self.survey)

        # Load benchmarks files from UBC-GIF codes
        AB_GIF = np.loadtxt('./benchmark_files/Test_calc_ElecSep/AB_GIF.txt')
        MN_GIF = np.loadtxt('./benchmark_files/Test_calc_ElecSep/MN_GIF.txt')
        AM_GIF = np.loadtxt('./benchmark_files/Test_calc_ElecSep/AM_GIF.txt')
        AN_GIF = np.loadtxt('./benchmark_files/Test_calc_ElecSep/AN_GIF.txt')
        BM_GIF = np.loadtxt('./benchmark_files/Test_calc_ElecSep/BM_GIF.txt')
        BN_GIF = np.loadtxt('./benchmark_files/Test_calc_ElecSep/BN_GIF.txt')

        # Assert agreements between the two codes
        test_AB = np.allclose(AB_GIF, AB)
        test_MN = np.allclose(MN_GIF, MN)
        test_AM = np.allclose(AM_GIF, AM)
        test_AN = np.allclose(AN_GIF, AN)
        test_BM = np.allclose(BM_GIF, BM)
        test_BN = np.allclose(BN_GIF, BN)

        passed = np.all(test_AB, test_MN, test_AM, test_AN, test_BM, test_BN)

        self.assertTrue(passed)

    def test_calc_rhoApp(self):

        # Compute apparent resistivity from survey
        rhoapp = DCUtils.calc_rhoApp(dpred, dpred.dobs, surveyType='dipole-dipole', spaceType='whole-space')

        # Load benchmarks files from UBC-GIF codes
        rhogif = np.loadtxt('./benchmark_files/Test_calc_rhoApp/RhoApp_GIF.txt')

        # Assert agreements between the two codes
        passed = np.allclose(rhoapp, rhogif)

        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
