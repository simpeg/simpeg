import unittest
import simpegDC as DC


class DCAnalyticTests(unittest.TestCase):

    def test_forwardAnalytic(self):
        self.assertTrue(DC.Examples.Verification.run() < 0.1)


if __name__ == '__main__':
    unittest.main()
