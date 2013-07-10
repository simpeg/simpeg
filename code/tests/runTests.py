import glob
import unittest

# This code will run all tests in directory named test_*.py

test_file_strings = glob.glob('test_*.py')
module_strings = [str[0:len(str)-3] for str in test_file_strings]
suites = [unittest.defaultTestLoader.loadTestsFromName(str) for str
          in module_strings]
testSuite = unittest.TestSuite(suites)
text_runner = unittest.TextTestRunner().run(testSuite)
