import unittest
from SimPEG import *
from simpegPF import BaseMag
import matplotlib.pyplot as plt
import simpegPF as PF
from scipy.constants import mu_0



class MagSensProblemTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_magnetics_inversion(self):

        driver = PF.MagneticsIO.MagneticsDriver_Inv('assets/magnetics/SimPEG_MAG3D_inv.inp')

        print driver.mesh
        print driver.survey
        print driver.m0
        print driver.mref
        print driver.activeCells




if __name__ == '__main__':
    unittest.main()
