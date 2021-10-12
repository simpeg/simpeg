import numpy as np
import SimPEG.electromagnetics.static.resistivity as spr
import unittest

class DC1DSimulationApparentRes(unittest.TestCase):
    def setUp(self):
        cond = np.r_[
            0.48254, 0.16143, 0.15505, 0.16997, 0.43467, 0.61810, 0.53219, 0.35801, 0.29295, 0.29500, 0.33625, 0.40644,
            0.49038, 0.55753, 0.57126, 0.50775, 0.42632, 0.35041, 0.29114, 0.28111, 0.27149, 0.26140, 0.25085, 0.24054,
            0.23030, 0.22013, 0.20979, 0.20294, 0.19773, 0.18896, 0.17846, 0.16891, 0.15830, 0.14665, 0.13594, 0.12784,
            0.12245, 0.11436, 0.10349, 0.09257, 0.08480, 0.07930, 0.07056, 0.05947, 0.04832, 0.03968, 0.03406, 0.02586,
            0.01455, 0.00670, 0.00100]
        thick = np.r_[
            2.00000, 2.15600, 2.32300, 2.50400, 2.69900, 2.90900, 3.13500, 3.37900, 3.64200, 3.92500, 4.23000, 4.55900,
            4.91500, 5.29500, 5.70800, 6.15300, 6.62900, 7.14600, 2.64200, 2.63300, 2.42700, 2.87800, 2.67200, 2.75100,
            2.63400, 2.71300, 2.72600, 0.87500, 1.86400, 2.75300, 2.76800, 2.25600, 3.32000, 2.80900, 2.82300, 1.43800,
            1.39900, 2.85100, 2.86600, 2.88000, 1.20400, 1.69000, 2.90900, 2.92300, 2.93800, 1.61000, 1.34300, 2.96800,
            2.98200, 1.14800]


        n_spacings = np.r_[2.00000, 4.00000, 8.00000, 16.00000, 32.00000, 64.00000, 128.00000]

        n_spacings = np.atleast_1d(n_spacings)

        if np.any(n_spacings < 2.0):
            raise NotImplementedError('Cannot calculate apparent resistivities for a spacing less than 2 meters '
                                      'is not supported.')

        n_spacings = np.sort(n_spacings)

        y = np.zeros_like(n_spacings)
        z = np.full_like(n_spacings, -1.0)
        a_locations = np.column_stack((-1.5 * n_spacings, y, z))
        b_locations = np.column_stack((1.5 * n_spacings, y, z))
        m_locations = np.column_stack((-0.5 * n_spacings, y, z))
        n_locations = np.column_stack((0.5 * n_spacings, y, z))

        receivers = [spr.receivers.Dipole(locations_m=m, locations_n=n) for m, n in zip(m_locations, n_locations)]
        sources = [spr.sources.Dipole(receiver_list=[receiver], location_a=a, location_b=b)
                   for receiver, a, b in zip(receivers, a_locations, b_locations)]

        survey = spr.Survey(sources)
        simulation = spr.Simulation1DLayers(survey=survey,
                                            data_type="apparent_resistivity",
                                            sigma=cond,
                                            thicknesses=thick
                                            )

        self.simulation = simulation
        self.app_res = np.r_[2.4332348140,3.1932686617,3.8364532280,3.4062204675,2.3837093273,2.0948867160,3.3301559763]

    def test_misfit(self, tol=1e-5):
        calc_app_res = self.simulation.dpred()
        error = np.sqrt(((calc_app_res - self.app_res)/self.app_res)**2)
        print(f'Error in Wenner App Res {error}')
        exceed_tol = np.any(error > tol)
        self.assertFalse(exceed_tol)

