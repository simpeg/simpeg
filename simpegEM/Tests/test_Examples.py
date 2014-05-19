import unittest, os
import simpegEM as EM

class EM_ExamplesRunning(unittest.TestCase):

    def test_CylInversion(self):
        execfile(os.path.join(EM.__path__[0], 'Examples', 'CylInversion.py'))

if __name__ == '__main__':
    unittest.main()
