import unittest
import numpy as np
np.random.seed(43)

from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import time_domain as tdem

class TestTxRxPropertyFailures(unittest.TestCase):

    def test_receiver_properties(self):

        xyz = np.c_[0., 0., 0.]
        projComp = 'Fx'
        rx = fdem.receivers.BaseRx(xyz, projComp=projComp)

        self.assertTrue((rx.projComp==projComp))
        self.assertRaises(ValueError, fdem.receivers.BaseRx, xyz, component='potato')
        self.assertRaises(TypeError, fdem.receivers.BaseRx, xyz, component=6.)
        
        print('Test receiver property raises passes')

    def test_source_properties(self):

        xyz = np.r_[0., 0., 0.]
        frequency = 1.

        # Base source
        src = fdem.sources.BaseFDEMSrc([], location=xyz, freq=frequency)
        self.assertTrue((src.frequency==frequency))
        self.assertRaises(AttributeError, fdem.sources.BaseFDEMSrc, [], frequency=None, location=xyz)
        
        # MagDipole
        self.assertRaises(TypeError, fdem.sources.MagDipole, [], frequency, location='not_a_vector')
        self.assertRaises(ValueError, fdem.sources.MagDipole, [], frequency, location=[0., 0., 0., 0.])
        self.assertRaises(TypeError, fdem.sources.MagDipole, [], frequency, xyz, orientation=['list', 'of', 'string'])
        self.assertRaises(ValueError, fdem.sources.MagDipole, [], frequency, xyz, orientation=[1, 0, 0, 0])

        # CircularLoop
        self.assertRaises(ValueError, fdem.sources.CircularLoop, [], frequency, location=[0., 0., 0.], current=0.)

        # LineCurrent
        self.assertRaises(TypeError, fdem.sources.LineCurrent, [], frequency, location=['a','b','c'])
        self.assertRaises(TypeError, fdem.sources.LineCurrent, [], frequency, location=np.random.rand(5, 3, 2))
        self.assertRaises(ValueError, fdem.sources.LineCurrent, [], frequency, location=np.random.rand(5, 3), current=0.)

        print('Test source property raises passes')

    def test_survey_properties(self):

        self.assertRaises(AttributeError, fdem.survey.Survey, None)

        rx = fdem.receivers.PointMagneticFluxDensity(np.c_[0., 0., 0.])
        src = tdem.sources.MagDipole([], np.r_[0., 0., 1.])
        self.assertRaises(TypeError, fdem.survey.Survey, src)

        
if __name__ == "__main__":
    unittest.main()