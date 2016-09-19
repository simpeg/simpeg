from __future__ import print_function
import unittest
from SimPEG import Mesh, np, PF
from SimPEG.Utils import io_utils
from scipy.constants import mu_0
import os
import urllib
import shutil


class MagSensProblemTests(unittest.TestCase):

    def setUp(self):
        url = 'https://storage.googleapis.com/simpeg/tests/potential_fields/'
        cloudfiles = ['MagData.obs', 'Gaussian.topo', 'Mesh_10m.msh',
                      'ModelStart.sus', 'SimPEG_Mag_Input.inp']

        self.basePath = io_utils.remoteDownload(url, cloudfiles)

    def test_magnetics_inversion(self):

        inp_file = self.basePath + 'SimPEG_Mag_Input.inp'

        driver = PF.MagneticsDriver.MagneticsDriver_Inv(inp_file)

        print(driver.mesh)
        print(driver.survey)
        print(driver.m0)
        print(driver.mref)
        print(driver.activeCells)
        print(driver.staticCells)
        print(driver.dynamicCells)
        print(driver.chi)
        print(driver.nC)
        print(driver.alphas)
        print(driver.bounds)
        print(driver.lpnorms)
        print(driver.eps)

        # Write obs to file
        PF.Magnetics.writeUBCobs(self.basePath + 'FWR_data.dat',
                                 driver.survey, driver.survey.dobs)

        # Clean up the working directory
        shutil.rmtree(self.basePath)

if __name__ == '__main__':
    unittest.main()
