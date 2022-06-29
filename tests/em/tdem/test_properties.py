import unittest
import numpy as np
np.random.seed(43)

from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import time_domain as tdem

class TestTxRxPropertyFailures(unittest.TestCase):

    def test_receiver_properties(self):

        xyz = np.c_[0., 0., 0.]
        times = np.logspace(-5, -2, 4)
        projComp = 'Fx'
        rx = tdem.receivers.BaseRx(xyz, times, projComp=projComp)

        self.assertTrue((rx.projComp==projComp))
        self.assertRaises(AttributeError, tdem.receivers.BaseRx, None, times)
        self.assertRaises(AttributeError, tdem.receivers.BaseRx, xyz, None)
        self.assertRaises(TypeError, tdem.receivers.BaseRx, xyz, component='potato')
        self.assertRaises(TypeError, tdem.receivers.BaseRx, xyz, component=6.)
        
        print('Test receiver property raises passes')

    def test_source_properties(self):

        xyz = np.r_[0., 0., 0.]

        # Base source
        src = tdem.sources.BaseTDEMSrc([], location=xyz, srcType='inductive')
        self.assertTrue((src.srcType=='inductive'))
        
        # MagDipole
        self.assertRaises(TypeError, tdem.sources.MagDipole, [], location='not_a_vector')
        self.assertRaises(ValueError, tdem.sources.MagDipole, [], location=[0., 0., 0., 0.])
        self.assertRaises(TypeError, tdem.sources.MagDipole, [], xyz, orientation=['list', 'of', 'string'])
        self.assertRaises(ValueError, tdem.sources.MagDipole, [], xyz, orientation=[1, 0, 0, 0])

        # CircularLoop
        self.assertRaises(ValueError, tdem.sources.CircularLoop, [], location=[0., 0., 0.], current=0.)

        # LineCurrent
        self.assertRaises(TypeError, tdem.sources.LineCurrent, [], location=['a','b','c'])
        self.assertRaises(TypeError, tdem.sources.LineCurrent, [], location=np.random.rand(5, 3, 2))
        self.assertRaises(ValueError, tdem.sources.LineCurrent, [], location=np.random.rand(5, 3), current=0.)

        print('Test source property raises passes')

    def test_survey_properties(self):

        self.assertRaises(AttributeError, tdem.survey.Survey, None)

        src = fdem.sources.MagDipole([], 1., np.r_[0., 0., 1.])
        self.assertRaises(TypeError, tdem.survey.Survey, src)

        
if __name__ == "__main__":
    unittest.main()