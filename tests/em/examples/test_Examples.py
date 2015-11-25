import unittest, os
from SimPEG.EM import Examples

class EM_ExamplesRunning(unittest.TestCase):

    def test_CylInversion(self):
        Examples.CylInversion.run(plotIt=False)

if __name__ == '__main__':
    unittest.main()
