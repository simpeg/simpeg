"""
Verification example for a multi-layer model with induced polarisation

This example is based on the synthetic model in 
Li et al (2019), A discussion of 2D induced polarization effects in airborne
electromagnetic and inversion with a robust 1D laterally constrained inversion
scheme, Geophysics, vol 84, no 2, doi 10.1190/geo2018-0102.1

"""
import numpy as np
import unittest

from simpeg import maps
import simpeg.electromagnetics.time_domain as tdem


# Survey config (Lin et al 2019)
SURVEY_CONFIG = dict(
    waveform=dict(
        current = np.array((0.,1.,1.,0.)), # [-]
        time = np.array((-10.e-3,-9.e-3,0.,5.e-6)), # [s]
        peak_current=100, # [A]
    ),
    ground=dict(
        resistivities = np.array([5000., 500., 5000.]), # [Ohm.m]
        thicknesses = np.array([20., 50.]), # [m]
        eta = np.array([0., 350e-3, 0.]), # [V/V]
        tau = np.array([0., 1.e-3, 0.]), # [s]
        c = np.array([0., 0.5, 0.]), # -
    ),
    tx=dict(
        nturns=16, # [-]
        elevation = (elevation:=30), # [m]
        area = (area:=300), # [m2]
        center = (center:=np.array((0,0,elevation))),
        radius = (radius:=np.sqrt(area/np.pi)),
        theta = (theta:=np.linspace(0, 2*np.pi, 200, endpoint=False)),
        pts = (pts:=np.vstack((
            radius*np.cos(theta), radius*np.sin(theta), [0.]*len(theta)
        ))),
        location = pts.T + center,
    ),
    rx=dict(
        location = center,
        orientation = 'z',
        gates = 10**np.linspace(-2, -5, 31) # time channels [s]
    )
)

class EM1D_TD_IP_test(unittest.TestCase):
    """ Testing surveys consistency of IP in time-domain computations

    - models with/without IPs
    - models without IP vs with IP and null chargeabilities
    - models with homogeneous/heterogeneous chargeabilities
    """
    
    @classmethod
    def setUpClass(cls):
        """ Configuring survey and simulations for all test methods """
        # Simpeg objects
        cls.survey = dict(
            waveform = (waveform:=tdem.sources.PiecewiseLinearWaveform(
                times=SURVEY_CONFIG['waveform']['time'],
                currents=SURVEY_CONFIG['waveform']['current'],
            )),
            model_mapping = maps.IdentityMap(
                nP=len(SURVEY_CONFIG['ground']['thicknesses'])+1
            ),
            receiver_list = (receiver_list:=[
                tdem.receivers.PointMagneticFluxTimeDerivative(
                    SURVEY_CONFIG['rx']['location'],
                    SURVEY_CONFIG['rx']['gates'],
                    orientation=SURVEY_CONFIG['rx']['orientation'],
                )
            ]),
            source_list = (source_list:=[
                tdem.sources.LineCurrent(
                    receiver_list=receiver_list,
                    location=SURVEY_CONFIG['tx']['location'],
                    waveform=waveform,
                    current=(
                        SURVEY_CONFIG['waveform']['peak_current']
                        *SURVEY_CONFIG['tx']['nturns']
                    )
                )
            ]),
            survey = tdem.Survey(source_list)
        )

        # Pelton parameters for all simulations
        eta_null = np.zeros(3)
        tau_null = np.zeros(3)
        c_null = np.zeros(3)
        
        eta_heter = SURVEY_CONFIG['ground']['eta']
        tau_heter = SURVEY_CONFIG['ground']['tau']
        c_heter = SURVEY_CONFIG['ground']['c']
        
        eta_single = SURVEY_CONFIG['ground']['eta'][1]
        tau_single = SURVEY_CONFIG['ground']['tau'][1]
        c_single = SURVEY_CONFIG['ground']['c'][1]
        
        eta_hom = np.ones(3)*eta_single
        tau_hom = np.ones(3)*tau_single
        c_hom = np.ones(3)*c_single

        # When using IP, note that infinite frequency resistivity/conductivity
        # is expected while Lin et al i(2019) rely on 0-frequency parameters
        cls.resistivities_0 = SURVEY_CONFIG['ground']['resistivities']
        cls.resistivities_inf_null = cls.resistivities_0*(1-eta_null)
        cls.resistivities_inf_heter = cls.resistivities_0*(1-eta_heter)
        cls.resistivities_inf_single = cls.resistivities_0*(1-eta_single)
        cls.resistivities_inf_hom = cls.resistivities_0*(1-eta_hom)

        # Preparing all simulations
        cls.simulation_no_ip = tdem.Simulation1DLayered(
            survey=cls.survey['survey'],
            thicknesses=SURVEY_CONFIG['ground']['thicknesses'],
            rhoMap=cls.survey['model_mapping'],
        )
        cls.simulation_null_ip = tdem.Simulation1DLayered(
            survey=cls.survey['survey'],
            thicknesses=SURVEY_CONFIG['ground']['thicknesses'],
            rhoMap=cls.survey['model_mapping'],
            eta=eta_null,
            tau=tau_null,
            c=c_null
        )
        cls.simulation_heter_ip = tdem.Simulation1DLayered(
            survey=cls.survey['survey'],
            thicknesses=SURVEY_CONFIG['ground']['thicknesses'],
            rhoMap=cls.survey['model_mapping'],
            eta=eta_heter,
            tau=tau_heter,
            c=c_heter
        )
        cls.simulation_hom_ip = tdem.Simulation1DLayered(
            survey=cls.survey['survey'],
            thicknesses=SURVEY_CONFIG['ground']['thicknesses'],
            rhoMap=cls.survey['model_mapping'],
            eta=eta_hom,
            tau=tau_hom,
            c=c_hom
        )
        cls.simulation_single_ip = tdem.Simulation1DLayered(
            survey=cls.survey['survey'],
            thicknesses=SURVEY_CONFIG['ground']['thicknesses'],
            rhoMap=cls.survey['model_mapping'],
            eta=SURVEY_CONFIG['ground']['eta'][1],
            tau=SURVEY_CONFIG['ground']['tau'][1],
            c=SURVEY_CONFIG['ground']['c'][1]
        )

    def test_null_ip(self):
        """ Consistency of model with null chargeability """
        dbdt_no_ip = np.abs(self.simulation_no_ip.dpred(
            self.resistivities_0
        ))
        dbdt_null_ip = np.abs(self.simulation_null_ip.dpred(
            self.resistivities_inf_null
        ))
        self.assertTrue(
            np.allclose(dbdt_no_ip, dbdt_null_ip),
            'No IP and null chargeabilitiy should return same results'
        )
    
    def test_ip(self):
        """ Consistency of models with/without IP """
        dbdt_no_ip = np.abs(self.simulation_no_ip.dpred(
            self.resistivities_0
        ))
        dbdt_hom_ip = np.abs(self.simulation_hom_ip.dpred(
            self.resistivities_inf_hom
        ))
        self.assertFalse(
            np.allclose(dbdt_no_ip, dbdt_hom_ip),
            'Models with and without IP should not return same results'
        )
    
    def test_hom_ip(self):
        """ Consistency of models with single / multiple homogeneous chargeabilities """
        dbdt_single_ip = np.abs(self.simulation_single_ip.dpred(
            self.resistivities_inf_single
        ))
        dbdt_hom_ip = np.abs(self.simulation_hom_ip.dpred(
            self.resistivities_inf_hom
        ))
        self.assertTrue(
            np.allclose(dbdt_single_ip, dbdt_hom_ip),
            'Models with single or homogeneous multiple parameters should'
            '  return same results'
        )
    
    def test_heter_ip(self):
        """ Consistency of models with hom/heter-ogeneous chargeabilities """
        dbdt_no_ip = np.abs(self.simulation_no_ip.dpred(
            self.resistivities_0
        ))
        dbdt_hom_ip = np.abs(self.simulation_hom_ip.dpred(
            self.resistivities_inf_hom
        ))
        dbdt_heter_ip = np.abs(self.simulation_heter_ip.dpred(
            self.resistivities_inf_heter
        ))
        self.assertFalse(
            np.allclose(dbdt_no_ip, dbdt_heter_ip),
            'No IP and heterogeneous chargeabilities results should not match'
        )
        self.assertFalse(
            np.allclose(dbdt_hom_ip, dbdt_heter_ip),
            'Hom/heter-ogeneous chargeabilities results should not match'
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
