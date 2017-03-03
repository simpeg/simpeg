import matplotlib
matplotlib.use('Agg')

from SimPEG.Examples import EM_Heagyetal2016_Casing as CasingExample
import unittest

TOL = 1e-6


class CasingExampleTest(unittest.TestCase):

    def setUp(self):
        print('Setting up Heagyetal Casing Example')
        # self.storedCasing = CasingExample.PrimSecCasingStoredResults()
        self.runCasing = CasingExample.PrimSecCasingExample()

        # download stored results
        # self.basePath = self.storedCasing.downloadStoredResults()
        # print('... Done')

    # def tearDown(self):
    #     self.storedCasing.removeStoredResults(self.basePath)

    def test_compare_results(self):
        print('----- Testing Prim Sec Casing Mapping -------')
        self.runCasing.primaryMapping.test(num=3)
        # recomputed = self.runCasing.run(runTests=True, plotIt=False)
        # stored = self.storedCasing.run(plotIt=False)
        # print('... Done')

        # for key in stored.keys():
        #     print('   Comparing {}'.format(key))

        #     norm_comp = np.linalg.norm(recomputed[key])
        #     norm_stor = np.linalg.norm(stored[key])
        #     norm_diff = np.linalg.norm(recomputed[key] - stored[key])

        #     tol = TOL * (norm_comp + norm_stor)/2.

        #     passed = norm_diff < TOL

        #     print('     norm computed: {comp},'
        #           '     norm stored: {stor},'
        #           '     norm diff: {diff},'
        #           '     passed? {passed}'.format(
        #             comp))

if __name__ == '__main__':
    unittest.main()
