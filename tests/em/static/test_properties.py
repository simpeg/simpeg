import unittest
import numpy as np
np.random.seed(43)

from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static import induced_polarization as ip
from SimPEG.electromagnetics.static import spectral_induced_polarization as sip

class TestTxRxPropertyFailures(unittest.TestCase):

    def test_receiver_properties(self):

        xyz_1 = np.c_[0., 0., 0.]
        xyz_2 = np.c_[10., 0., 0.]
        times = np.logspace(-4, -2, 3)

        # Base DC receiver
        self.assertRaises(ValueError, dc.receivers.BaseRx, xyz_1, projField='potato')
        self.assertRaises(TypeError, dc.receivers.BaseRx, xyz_1, projField=6.)
        self.assertRaises(ValueError, dc.receivers.BaseRx, xyz_1, data_type='potato')

        # DC Dipole receiver
        self.assertRaises(AttributeError, dc.receivers.Dipole, locations=None)
        self.assertRaises(ValueError, dc.receivers.Dipole, locations=[xyz_1, xyz_2, xyz_1])
        self.assertRaises(ValueError, dc.receivers.Dipole, locations=[xyz_1, np.r_[xyz_2, xyz_1]])

        # Base SIP receiver
        self.assertRaises(ValueError, sip.receivers.BaseRx, locations=[xyz_1, xyz_2], orientation='potato')
        self.assertRaises(ValueError, sip.receivers.BaseRx, locations=[xyz_1, xyz_2], data_type='potato')

        # SIP Dipole receiver
        self.assertRaises(AttributeError, sip.receivers.Dipole, times=times, locations=None)
        self.assertRaises(ValueError, sip.receivers.Dipole, times=times, locations=[xyz_1, xyz_2, xyz_1])
        self.assertRaises(ValueError, sip.receivers.Dipole, times=times, locations=[xyz_1, np.r_[xyz_2, xyz_1]])

        
        print('Test receiver property raises passes')

    def test_source_properties(self):

        xyz_1 = np.c_[0., 0., 0.]
        xyz_2 = np.c_[10., 0., 0.]
        times = np.logspace(-4, -2, 3)

        # Base SIP source
        self.assertRaises(TypeError, sip.sources.BaseSrc, dc.receivers.Dipole(locations=[xyz_1, xyz_2]), location=xyz_1)
        rx = sip.receivers.Dipole(locations=[xyz_1, xyz_2], times=times)
        self.assertRaises(ValueError, sip.sources.BaseSrc, rx, location=xyz_1, current=0.)

        # SIP Dipole receiver
        self.assertRaises(AttributeError, sip.sources.Dipole, rx, location=None)
        self.assertRaises(ValueError, sip.sources.Dipole, rx, location=[xyz_1, xyz_2, xyz_1])
        self.assertRaises(ValueError, sip.sources.Dipole, rx, location=[xyz_1, np.r_[xyz_2, xyz_1]])

        print('Test source property raises passes')

    def test_survey_properties(self):

        xyz_1 = np.c_[0., 0., 0.]
        xyz_2 = np.c_[10., 0., 0.]
        times = np.logspace(-4, -2, 3)
        
        # DC survey
        self.assertRaises(AttributeError, dc.survey.Survey, None)

        # SIP survey
        rx = sip.receivers.Dipole(locations=[xyz_1, xyz_2], times=times)
        src = sip.sources.Dipole(rx, location=[xyz_1, xyz_2])
        self.assertRaises(AttributeError, sip.survey.Survey, None)
        self.assertRaises(TypeError, sip.survey.Survey, src, survey_geometry=6.)
        self.assertRaises(ValueError, sip.survey.Survey, src, survey_geometry='potato')

        print('Test survey property raises passes')

        
if __name__ == "__main__":
    unittest.main()