import unittest
import sys
from SimPEG import Examples
import numpy as np

def get(test):
    def func(self):
        print '\nTesting %s.run(plotIt=False)\n'%test
        getattr(Examples, test).run(plotIt=False)
        self.assertTrue(True)
    return func
attrs = dict()
for test in Examples.__examples__:
    attrs['test_'+test] = get(test)

TestExamples = type('TestExamples', (unittest.TestCase,), attrs)


if __name__ == '__main__':
    unittest.main()
