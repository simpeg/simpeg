from __future__ import print_function
import unittest
import numpy as np
from SimPEG import DC
from SimPEG import Utils
import matplotlib.pyplot as plt


class DCSurveyDesignTests(unittest.TestCase):

    def test(self, showIt=False):

        dc = DC.SurveyDesign()
        dc.genLocs_2D('pole-dipole', 0., 100., 10., 15)
        dc.plot2Dgeometry(iSrc=7, showIt=showIt)
        dc.setMesh_2D(2.5, 2.5)
        dc.genDCSurvey_2D()
        sigma = np.ones(dc.mesh.nC) * 0.01
        dc.runForwardSimulation(sigma)
        dc.plotData_2D(showIt=showIt)
        dc.plotfields_2D(showIt=showIt)
        self.assertTrue(True)
        print (">> SurveyDesign class for DC runs")

if __name__ == '__main__':
    unittest.main()
