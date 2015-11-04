import unittest
import sys
from SimPEG.FLOW.Examples import Celia1990
import numpy as np

class TestCelia1990(unittest.TestCase):
    def test_running(self):
        Celia1990.run(plotIt=False)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
