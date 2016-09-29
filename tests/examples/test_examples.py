from __future__ import print_function
import unittest
import sys
import os
from SimPEG import Examples
import numpy as np

class compareInitFiles(unittest.TestCase):
    def test_compareInitFiles(self):
        print('Checking that __init__.py up-to-date in SimPEG/Examples')
        fName = os.path.abspath(__file__)
        ExamplesDir = os.path.sep.join(fName.split(os.path.sep)[:-3] + ['SimPEG', 'Examples'])

        files = os.listdir(ExamplesDir)

        pyfiles = []
        [pyfiles.append(py.rstrip('.py')) for py in files if py.endswith('.py') and py != '__init__.py']

        setdiff = set(pyfiles) - set(Examples.__examples__)

        print(' Any missing files? ', setdiff)

        didpass = (setdiff == set())

        self.assertTrue(didpass, "Examples not up to date, run 'python __init__.py' from SimPEG/Examples to update")

def get(test):
    def test_func(self):
        print('\nTesting {0!s}.run(plotIt=False)\n'.format(test))
        getattr(Examples, test).run(plotIt=False)
        self.assertTrue(True)
    return test_func
attrs = dict()

for test in Examples.__examples__:
    attrs['test_'+test] = get(test)

TestExamples = type('TestExamples', (unittest.TestCase,), attrs)

if __name__ == '__main__':
    unittest.main()
