from __future__ import print_function
import unittest
import numpy as np
from SimPEG.EM.Static import DC, Utils as DCUtils
from SimPEG.Utils import io_utils
import shutil
import os


class DCUtilsTests(unittest.TestCase):

    def setUp(self):

        url = 'https://storage.googleapis.com/simpeg/tests/dc_utils/'
        cloudfiles = ['dPred_Fullspace.txt', 'AB_GIF.txt',
                      'MN_GIF.txt', 'AM_GIF.txt', 'AN_GIF.txt',
                      'BM_GIF.txt', 'BN_GIF.txt', 'RhoApp_GIF.txt']

        self.basePath = os.path.expanduser('~/Downloads/simpegtemp')
        self.files = io_utils.download(
            [url+f for f in cloudfiles],
            folder=self.basePath,
            overwrite=True
        )

        survey_file = os.path.sep.join([self.basePath, 'dPred_Fullspace.txt'])
        DCsurvey = DCUtils.readUBC_DC3Dobs(survey_file)
        DCsurvey = DCsurvey['DCsurvey']
        self.survey = DCsurvey

    def test_ElecSep(self):

        # Compute dipoles distances from survey
        AB, MN, AM, AN, BM, BN = DCUtils.calc_ElecSep(self.survey)

        # Load benchmarks files from UBC-GIF codes
        AB_file = os.path.sep.join([self.basePath, 'AB_GIF.txt'])
        MN_file = os.path.sep.join([self.basePath, 'MN_GIF.txt'])
        AM_file = os.path.sep.join([self.basePath, 'AM_GIF.txt'])
        AN_file = os.path.sep.join([self.basePath, 'AN_GIF.txt'])
        BM_file = os.path.sep.join([self.basePath, 'BM_GIF.txt'])
        BN_file = os.path.sep.join([self.basePath, 'BN_GIF.txt'])

        AB_GIF = np.loadtxt(AB_file)
        MN_GIF = np.loadtxt(MN_file)
        AM_GIF = np.loadtxt(AM_file)
        AN_GIF = np.loadtxt(AN_file)
        BM_GIF = np.loadtxt(BM_file)
        BN_GIF = np.loadtxt(BN_file)

        # Assert agreements between the two codes
        test_AB = np.allclose(AB_GIF, AB)
        test_MN = np.allclose(MN_GIF, MN)
        test_AM = np.allclose(AM_GIF, AM)
        test_AN = np.allclose(AN_GIF, AN)
        test_BM = np.allclose(BM_GIF, BM)
        test_BN = np.allclose(BN_GIF, BN)

        passed = np.all([test_AB, test_MN, test_AM, test_AN, test_BM, test_BN])

        self.assertTrue(passed)

    def test_calc_rhoApp(self):

        # Compute apparent resistivity from survey
        rhoapp = DCUtils.calc_rhoApp(self.survey, dobs=self.survey.dobs,
                                     surveyType='dipole-dipole',
                                     spaceType='whole-space', eps=1e-16)

        # Load benchmarks files from UBC-GIF codes
        rhoappfile = os.path.sep.join([self.basePath, 'RhoApp_GIF.txt'])
        rhogif = np.loadtxt(rhoappfile)
        # remove value with almost null geometric factor
        idx = rhoapp<1e8
        # Assert agreements between the two codes
        passed = np.allclose(rhoapp[idx], rhogif[idx])
        self.assertTrue(passed)

        # Clean up the working directory
        shutil.rmtree(self.basePath)

if __name__ == '__main__':
    unittest.main()
