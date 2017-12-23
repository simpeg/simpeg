from SimPEG import DC
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestsIO_2D(unittest.TestCase):

    def setUp(self):
        self.plotIt = False
        np.random.seed(1)
        # Obtain ABMN locations

        xmin, xmax = 0., 200.
        ymin, ymax = 0., 0.
        zmin, zmax = 0, 0
        self.endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        # Generate DC survey object
        self.surveys = []
        self.survey_types = [
            "dipole-dipole", "dipole-pole", "pole-dipole", "pole-pole"
        ]
        for i, survey_type in enumerate(self.survey_types):
            survey = DC.Utils.gen_DCIPsurvey(
                self.endl, self.survey_types[i], dim=2, a=10, b=10, n=10
            )
            self.surveys.append(survey)

    def test_apparent_data(self):
        passed_list = []
        for i, survey in enumerate(self.surveys):
            # Initiate I/O class for DC
            IO = DC.IO()
            survey.getABMN_locations()
            survey = IO.from_ambn_locations_to_survey(
                survey.a_locations, survey.b_locations,
                survey.m_locations, survey.n_locations,
                survey_type=self.survey_types[i],
                data_dc_type='apparent_resistivity',
                data_ip_type='apparent_chargeability',
                data_dc=np.ones(survey.nD)*100.,
                data_ip=np.ones(survey.nD)
            )
            if self.plotIt:
                IO.plotPseudoSection(data_type='apparent_resistivity')
                IO.plotPseudoSection(data_type='volt')
                IO.plotPseudoSection(data_type='apparent_conductivity')
                IO.plotPseudoSection(data_type='apparent_chargeability')
                IO.plotPseudoSection(data_type='volt_ip')
                plt.show()

            mesh, actind = IO.set_mesh()
            passed_temp = np.linalg.norm(
                IO.voltages - IO.data_dc * IO.G
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_resistivity - IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_conductivity - 1./IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_chargeability - IO.data_ip
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.voltages_ip - IO.data_ip * IO.voltages
                ) < 1e-10
            passed_list.append(passed_temp)

        self.assertTrue(np.all(passed_list))

    def test_voltage_data(self):
        passed_list = []
        for i, survey in enumerate(self.surveys):
            # Initiate I/O class for DC
            IO = DC.IO()
            survey.getABMN_locations()
            G = DC.Utils.geometric_factor(
                survey, survey_type=self.survey_types[i],
                space_type='half-space'
                )
            rho_a = 100.
            survey = IO.from_ambn_locations_to_survey(
                survey.a_locations, survey.b_locations,
                survey.m_locations, survey.n_locations,
                survey_type=self.survey_types[i],
                data_dc_type='volt',
                data_ip_type='volt',
                data_dc=rho_a * G,
                data_ip=G
            )
            if self.plotIt:
                IO.plotPseudoSection(data_type='apparent_resistivity')
                IO.plotPseudoSection(data_type='volt')
                IO.plotPseudoSection(data_type='apparent_conductivity')
                IO.plotPseudoSection(data_type='apparent_chargeability')
                IO.plotPseudoSection(data_type='volt_ip')
                plt.show()

            mesh, actind = IO.set_mesh()
            passed_temp = np.linalg.norm(
                IO.voltages - IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_resistivity - IO.data_dc / IO.G
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_conductivity - IO.G / IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.voltages_ip - IO.data_ip
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_chargeability - IO.data_ip / IO.voltages
                ) < 1e-10
            passed_list.append(passed_temp)

        self.assertTrue(np.all(passed_list))

    def test_topo(self):
        IO = DC.IO()
        self.survey = DC.Utils.gen_DCIPsurvey(
            self.endl, "dipole-dipole", dim=2, a=10, b=10, n=10
        )
        self.survey.getABMN_locations()
        self.survey = IO.from_ambn_locations_to_survey(
            self.survey.a_locations, self.survey.b_locations,
            self.survey.m_locations, self.survey.n_locations,
            'dipole-dipole', data_dc_type='apparent_resistivity',
            data_dc=np.ones(self.survey.nD)*100.
        )

        if self.plotIt:
            IO.plotPseudoSection(data_type='apparent_resistivity')
            plt.show()

        mesh, actind = IO.set_mesh()
        topo, mesh1D = DC.Utils.genTopography(mesh, -10, 0, its=100)
        mesh, actind = IO.set_mesh(topo=np.c_[mesh1D.vectorCCx, topo])
        options_topo = ["top", "center"]
        for option in options_topo:
            self.survey.drapeTopo(mesh, actind, option=option)
            if self.plotIt:
                mesh.plotImage(actind)
                plt.plot(
                    self.survey.electrode_locations[:, 0],
                    self.survey.electrode_locations[:, 1], 'k.'
                    )
                plt.show()


class TestsIO_3D(unittest.TestCase):

    def setUp(self):
        self.plotIt = False
        np.random.seed(1)
        # Obtain ABMN locations

        xmin, xmax = 0., 200.
        ymin, ymax = 0., 0.
        zmin, zmax = 0, 0
        self.endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        # Generate DC survey object
        self.surveys = []
        self.survey_types = [
            "dipole-dipole", "dipole-pole", "pole-dipole", "pole-pole"
        ]
        for i, survey_type in enumerate(self.survey_types):
            survey = DC.Utils.gen_DCIPsurvey(
                self.endl, self.survey_types[i], dim=3, a=10, b=10, n=10
            )
            self.surveys.append(survey)

    def test_apparent_data(self):
        passed_list = []
        for i, survey in enumerate(self.surveys):
            # Initiate I/O class for DC
            IO = DC.IO()
            survey.getABMN_locations()
            survey = IO.from_ambn_locations_to_survey(
                survey.a_locations, survey.b_locations,
                survey.m_locations, survey.n_locations,
                survey_type=self.survey_types[i],
                data_dc_type='apparent_resistivity',
                data_ip_type='apparent_chargeability',
                data_dc=np.ones(survey.nD)*100.,
                data_ip=np.ones(survey.nD),
                dimension=3
            )
            if self.plotIt:
                IO.plotPseudoSection(data_type='apparent_resistivity')
                IO.plotPseudoSection(data_type='volt')
                IO.plotPseudoSection(data_type='apparent_conductivity')
                IO.plotPseudoSection(data_type='apparent_chargeability')
                IO.plotPseudoSection(data_type='volt_ip')
                plt.show()

            mesh, actind = IO.set_mesh()
            passed_temp = np.linalg.norm(
                IO.voltages - IO.data_dc * IO.G
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_resistivity - IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_conductivity - 1./IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_chargeability - IO.data_ip
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.voltages_ip - IO.data_ip * IO.voltages
                ) < 1e-10
            passed_list.append(passed_temp)

        self.assertTrue(np.all(passed_list))

    def test_voltage_data(self):
        passed_list = []
        for i, survey in enumerate(self.surveys):
            # Initiate I/O class for DC
            IO = DC.IO()
            survey.getABMN_locations()
            G = DC.Utils.geometric_factor(
                survey, survey_type=self.survey_types[i],
                space_type='half-space'
                )
            rho_a = 100.
            survey = IO.from_ambn_locations_to_survey(
                survey.a_locations, survey.b_locations,
                survey.m_locations, survey.n_locations,
                survey_type=self.survey_types[i],
                data_dc_type='volt',
                data_ip_type='volt',
                data_dc=rho_a * G,
                data_ip=G,
                dimension=3
            )

            mesh, actind = IO.set_mesh()
            passed_temp = np.linalg.norm(
                IO.voltages - IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_resistivity - IO.data_dc / IO.G
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_conductivity - IO.G / IO.data_dc
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.voltages_ip - IO.data_ip
                ) < 1e-10
            passed_list.append(passed_temp)

            passed_temp = np.linalg.norm(
                IO.apparent_chargeability - IO.data_ip / IO.voltages
                ) < 1e-10
            passed_list.append(passed_temp)

        self.assertTrue(np.all(passed_list))

    def test_topo(self):
        IO = DC.IO()
        self.survey = DC.Utils.gen_DCIPsurvey(
            self.endl, "dipole-dipole", dim=2, a=10, b=10, n=10
        )
        self.survey.getABMN_locations()
        self.survey = IO.from_ambn_locations_to_survey(
            self.survey.a_locations, self.survey.b_locations,
            self.survey.m_locations, self.survey.n_locations,
            'dipole-dipole', data_dc_type='apparent_resistivity',
            data_dc=np.ones(self.survey.nD)*100.
        )

        if self.plotIt:
            IO.plotPseudoSection(data_type='apparent_resistivity')
            plt.show()

        mesh, actind = IO.set_mesh()
        topo, mesh1D = DC.Utils.genTopography(mesh, -10, 0, its=100)
        mesh, actind = IO.set_mesh(topo=np.c_[mesh1D.vectorCCx, topo])
        options_topo = ["top", "center"]
        for option in options_topo:
            self.survey.drapeTopo(mesh, actind, option=option)
            if self.plotIt:
                mesh.plotImage(actind)
                plt.plot(
                    self.survey.electrode_locations[:, 0],
                    self.survey.electrode_locations[:, 1], 'k.'
                    )
                plt.show()

if __name__ == '__main__':
    unittest.main()
