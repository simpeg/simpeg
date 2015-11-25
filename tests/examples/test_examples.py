import unittest
import sys
from SimPEG.Examples import Linear, DCfwd
import numpy as np

class TestLinear(unittest.TestCase):
    def test_running(self):
        Linear.run(100, plotIt=False)
        self.assertTrue(True)

class TestDCfwd(unittest.TestCase):
    def test_running(self):
        DCfwd.run(plotIt=False)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
