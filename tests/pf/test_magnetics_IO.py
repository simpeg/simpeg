import unittest
from SimPEG import Mesh, np, PF
from scipy.constants import mu_0
import os


class MagSensProblemTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_magnetics_inversion(self):

        inp_file = os.getcwd() + os.path.sep.join([
                os.path.split(__file__)[0],
                'assets', 'magnetics', 'PYMAG3D_inv.inp'
            ])
        print inp_file
        driver = PF.MagneticsDriver.MagneticsDriver_Inv(inp_file)

        print driver.mesh
        print driver.survey
        print driver.m0
        print driver.mref
        print driver.activeCells
        print driver.staticCells
        print driver.dynamicCells
        print driver.chi
        print driver.nC
        print driver.alphas
        print driver.bounds
        print driver.lpnorms
        print driver.eps

if __name__ == '__main__':
    unittest.main()
