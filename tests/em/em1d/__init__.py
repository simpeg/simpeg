if __name__ == "__main__":
    import glob
    import unittest

    test_file_strings = glob.glob("test*.py")
    module_strings = [s[0 : len(s) - 3] for s in test_file_strings]
    suites = [
        unittest.defaultTestLoader.loadTestsFromName(s) for s in module_strings
    ]
    testSuite = unittest.TestSuite(suites)

    unittest.TextTestRunner(verbosity=2).run(testSuite)
