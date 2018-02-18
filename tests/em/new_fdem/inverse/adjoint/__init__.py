if __name__ == '__main__':
    import glob
    import unittest
    test_file_strings = glob.glob('test_*.py')
    module_strings = [str[0:len(str)-3] for str in test_file_strings]
    suites = [unittest.defaultTestLoader.loadTestsFromName(str) for str
              in module_strings]
    testSuite = unittest.TestSuite(suites)

    unittest.TextTestRunner(verbosity=2).run(testSuite)
