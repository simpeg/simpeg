import unittest
import sys
import os
import subprocess

# test that all of the examples run
TESTDIR = os.path.abspath(__file__)

# where are the examples?
ExamplesDir = os.path.sep.join(TESTDIR.split(os.path.sep)[:-3] + ['examples'])


def setUp():
    expaths = []  # list of notebooks, with file paths
    exnames = []  # list of notebook names (for making the tests)

    # walk the test directory and find all examples (start with plot_XXX)
    for dirname, dirnames, filenames in os.walk(ExamplesDir):
        for filename in filenames:
            if filename.endswith(".py") and not filename.endswith(".pyc"):
                expaths.append(os.path.abspath(dirname)) # get abspath of notebook
                exnames.append("".join(filename[:-3])) # strip off the file extension

    return expaths, exnames


def get(exname, expath):

    # use nbconvert to execute the notebook
    def test_func(self):
        passing = True
        print("\n--------------- Testing {0} ---------------".format(exname))
        print("   {0}".format(expath))

        sys.path.append(expath)
        ex = __import__(exname)
        subprocess.call(["python {}.py".format()])

        print("... ok ")

    return test_func


attrs = dict()
expaths, exnames = setUp()
print(expaths[0])
print(exnames[0])

# build test for each notebook
for i, ex in enumerate(exnames):
    attrs["test_"+ex] = get(ex, expaths[i])

# create class to unit test notebooks
TestNotebooks = type("TestNotebooks", (unittest.TestCase,), attrs)


if __name__ == "__main__":
    unittest.main()
