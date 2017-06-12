import numpy as np
import os
from SimPEG.EM.Utils.CurrentUtils import (
    getStraightLineCurrentIntegral, getSourceTermLineCurrentPolygon
    )
import unittest
from SimPEG.Utils import io_utils


class LineCurrentTests(unittest.TestCase):

    def setUp(self):
        url = 'https://storage.googleapis.com/simpeg/tests/em_utils/'
        cloudfile = 'currents.npy'
        self.basePath = io_utils.download(url + cloudfile)

    def test(self):

        hx, hy, hz = 2, 2, 2
        ax, ay, az = 0.1, 0.3, 0.4
        bx, by, bz = 1, 0.8, 0.7

        sx_true = np.r_[0.475875, 0.176625, 0.176625, 0.070875]
        sy_true = np.r_[0.265625, 0.096875, 0.096875, 0.040625]
        sz_true = np.r_[0.1605, 0.057, 0.057, 0.0255]

        sx, sy, sz = getStraightLineCurrentIntegral(
            hx, hy, hz, ax, ay, az, bx, by, bz
        )

        s_true = np.r_[sx_true, sy_true, sz_true]
        s = np.r_[sx, sy, sz]
        err = np.linalg.norm(s_true-s) / np.linalg.norm(s_true)
        print (">> Test getStraightLineCurrentIntegral")
        if err < 1e-5:
            passed = True
        else:
            passed = False
            print (("Failed, Error = %d") % (err))

        self.assertTrue(passed)


        hx = np.ones(10)*1.
        hy = np.ones(10)*2.
        hz = np.ones(10)*3.

        px = np.r_[0, 3, 5]
        py = np.r_[0, 3, 5]
        pz = np.r_[2., 3., 4.]

        xorig = np.r_[0., 0., 0.]

        out = getSourceTermLineCurrentPolygon(xorig, hx, hy, hz, px, py, pz)
        out_true = np.load(self.basePath)
        err = np.linalg.norm(out-out_true)
        print (">> Test getSourceTermLineCurrentPolygon")

        if err < 1e-5:
            passed = True
        else:
            passed = False
            print (("Failed, Error = %d") % (err))

        self.assertTrue(passed)

        os.remove(self.basePath)

if __name__ == '__main__':
    unittest.main()

