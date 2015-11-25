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
tests = [_ for _ in dir(Examples) if not _.startswith('_')]
for test in tests:
    attrs['test_'+test] = get(test)
del get, tests, _

TestExamples = type('TestExamples', (unittest.TestCase,), attrs)


if __name__ == '__main__':
    unittest.main()
