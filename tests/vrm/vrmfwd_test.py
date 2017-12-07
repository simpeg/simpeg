import unittest
import SimPEG.VRM as VRM
import numpy as np

# SUMMARY OF TESTS
#
# 1) Does bz and dbz/dt match analytic solution for a circular loop in a half-space?
# 2) Can you use circular loop, dipole and tx lines to get the same result?
# 3) Under reasonable circumstances, can you get the linear forward problem to match the log uniform?

class VRM_fwd_tests(unittest.TestCase):

    def test_against_analytic():

        



if __name__ == '__main__':
    unittest.main()
