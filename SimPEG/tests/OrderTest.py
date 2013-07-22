import sys
sys.path.append('../../')
from SimPEG import TensorMesh
import numpy as np
import unittest


class OrderTest(unittest.TestCase):
    """Order test sets up the basics for testing order of decrease for a function on a mesh."""

    name = "Order Test"
    expectedOrder = 2
    meshSizes = [4, 8, 16, 32]

    def setupMesh(self, nc):
        # Define the mesh
        h1 = np.ones(nc)/nc
        h2 = np.ones(nc)/nc
        h3 = np.ones(nc)/nc
        h = [h1, h2, h3]
        self.M = TensorMesh(h)

    def getError(self):
        """Overwrite this function  with the guts of the test."""
        return 1.

    def orderTest(self):
        order = []
        err_old = 0.
        nc_old = 0.
        for ii, nc in enumerate(self.meshSizes):
            self.setupMesh(nc)
            err = self.getError()
            if ii == 0:
                print ''
                print 'Testing order of:  ' + self.name
                print '__________________________________________'
                print '   h  |   inf norm  |  ratio   |  order'
                print '~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~|~~~~~~~~~~'
                print '%4i  |  %8.2e   |' % (nc, err)
            else:
                order.append(np.log(err/err_old)/np.log(float(nc_old)/float(nc)))
                print '%4i  |  %8.2e   |  %6.4f  |  %6.4f' % (nc, err, err_old/err, order[-1])
            err_old = err
            nc_old = nc
        print '------------------------------------------'
        self.assertTrue(len(np.where(np.array(order) > 0.9*self.expectedOrder)[0]) > np.floor(0.75*len(order)))


if __name__ == '__main__':
    unittest.main()
