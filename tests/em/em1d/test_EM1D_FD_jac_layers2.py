from simpeg import maps
from discretize import tests, TensorMesh
import simpeg.electromagnetics.frequency_domain as fdem
import numpy as np
from scipy.constants import mu_0
from scipy.sparse import diags


class TestEM1D_FD_Jacobian_MagDipole:

    # Tests 2nd order convergence of Jvec and Jtvec for magnetic dipole sources.
    # - All src and rx orientations
    # - All rx components
    # - Span many frequencies
    # - Tests derivatives wrt sigma, mu, thicknesses and h
    def setup_class(self):
        # Layers and topography
        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        # Survey Geometry
        height = 1e-5
        src_location = np.array([0.0, 0.0, 100.0 + height])
        rx_location = np.array([5.0, 5.0, 100.0 + height])
        frequencies = np.logspace(1, 8, 9)
        orientations = ["x", "y", "z"]
        components = ["real", "imag", "both"]

        # Define sources and receivers
        source_list = []
        for f in frequencies:
            for tx_orientation in orientations:
                receiver_list = []

                for rx_orientation in orientations:
                    for comp in components:
                        receiver_list.append(
                            fdem.receivers.PointMagneticFieldSecondary(
                                rx_location, orientation=rx_orientation, component=comp
                            )
                        )

                source_list.append(
                    fdem.sources.MagDipole(
                        receiver_list,
                        frequency=f,
                        location=src_location,
                        orientation=tx_orientation,
                    )
                )

        # Survey
        survey = fdem.Survey(source_list)

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.height = height
        self.frequencies = frequencies
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1

        wire_map = maps.Wires(
            ("mu", self.nlayers),
            ("sigma", self.nlayers),
            ("h", 1),
            ("thicknesses", self.nlayers - 1),
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        self.mu_map = maps.ExpMap(nP=self.nlayers) * wire_map.mu
        self.thicknesses_map = maps.ExpMap(nP=self.nlayers - 1) * wire_map.thicknesses
        nP = len(source_list)
        surject_mesh = TensorMesh([np.ones(nP)])
        self.h_map = maps.SurjectFull(surject_mesh) * maps.ExpMap(nP=1) * wire_map.h

        sim = fdem.Simulation1DLayered(
            survey=self.survey,
            sigmaMap=self.sigma_map,
            muMap=self.mu_map,
            thicknessesMap=self.thicknesses_map,
            hMap=self.h_map,
            topo=self.topo,
        )

        self.sim = sim

    def test_EM1DFDJvec_Layers(self):
        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_1D = np.r_[
            np.log(mu), np.log(sig), np.log(self.height), np.log(self.thicknesses)
        ]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        def derChk(m):
            return [fwdfun(m), lambda mx: jacfun(m, mx)]

        dm = m_1D * 0.5

        passed = tests.check_derivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15, random_seed=9186724
        )
        assert passed

    def test_EM1DFDJtvec_Layers(self):
        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_true = np.r_[
            np.log(mu), np.log(sig), np.log(self.height), np.log(self.thicknesses)
        ]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * 1.5 * mu_half),
            np.log(np.ones(self.nlayers) * sigma_half),
            np.log(0.5 * self.height),
            np.log(self.thicknesses) * 0.9,
        ]
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini - dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = np.linalg.norm(dpred - dobs) ** 2
            dmisfit = 2.0 * self.sim.Jtvec(
                m, dr
            )  # derivative of ||dpred - dobs||^2 gives factor of 2
            return misfit, dmisfit

        def derChk(m):
            return misfit(m, dobs)

        passed = tests.check_derivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-27, random_seed=2345
        )
        assert passed

    def test_jtjdiag(self):
        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        model = np.r_[
            np.log(mu), np.log(sig), np.log(self.height), np.log(self.thicknesses)
        ]

        rng = np.random.default_rng(seed=42)
        weights_matrix = diags(rng.random(size=self.sim.survey.nD))
        jtj_diag = self.sim.getJtJdiag(model, W=weights_matrix)

        J = self.sim.getJ(model)
        expected = np.diag(J.T @ weights_matrix.T @ weights_matrix @ J)
        np.testing.assert_allclose(expected, jtj_diag)


class TestEM1D_FD_Jacobian_CircularLoop:
    # Tests 2nd order convergence of Jvec and Jtvec for horizontal loop sources.
    # - All rx orientations
    # - All rx components
    # - Span many frequencies
    # - Tests derivatives wrt sigma, mu, thicknesses and h
    def setup_class(self):
        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]
        height = 1e-5

        src_location = np.array([0.0, 0.0, 100.0 + height])
        rx_location = np.array([0.0, 0.0, 100.0 + height])
        frequencies = np.logspace(1, 8, 9)
        orientations = ["x", "y", "z"]
        components = ["real", "imag", "both"]
        I = 1.0
        a = 10.0

        # Define sources and receivers
        source_list = []
        for f in frequencies:
            receiver_list = []

            for rx_orientation in orientations:
                for comp in components:
                    receiver_list.append(
                        fdem.receivers.PointMagneticFieldSecondary(
                            rx_location, orientation=rx_orientation, component=comp
                        )
                    )

            source_list.append(
                fdem.sources.CircularLoop(
                    receiver_list, f, src_location, radius=a, current=I
                )
            )

        # Survey
        survey = fdem.Survey(source_list)

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.height = height
        self.frequencies = frequencies
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1

        nP = len(source_list)

        wire_map = maps.Wires(
            ("sigma", self.nlayers),
            ("mu", self.nlayers),
            ("thicknesses", self.nlayers - 1),
            ("h", 1),
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        self.mu_map = maps.ExpMap(nP=self.nlayers) * wire_map.mu
        self.thicknesses_map = maps.ExpMap(nP=self.nlayers - 1) * wire_map.thicknesses
        surject_mesh = TensorMesh([np.ones(nP)])
        self.h_map = maps.SurjectFull(surject_mesh) * maps.ExpMap(nP=1) * wire_map.h

        sim = fdem.Simulation1DLayered(
            survey=self.survey,
            sigmaMap=self.sigma_map,
            muMap=self.mu_map,
            thicknessesMap=self.thicknesses_map,
            hMap=self.h_map,
            topo=self.topo,
        )

        self.sim = sim

    def test_EM1DFDJvec_Layers(self):
        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_1D = np.r_[
            np.log(sig), np.log(mu), np.log(self.thicknesses), np.log(self.height)
        ]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        def derChk(m):
            return [fwdfun(m), lambda mx: jacfun(m, mx)]

        dm = m_1D * 0.5

        passed = tests.check_derivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15, random_seed=664
        )
        assert passed

    def test_EM1DFDJtvec_Layers(self):
        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 2 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[3] = mu_blk

        # General model
        m_true = np.r_[
            np.log(sig), np.log(mu), np.log(self.thicknesses), np.log(self.height)
        ]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * sigma_half),
            np.log(np.ones(self.nlayers) * 1.5 * mu_half),
            np.log(self.thicknesses) * 0.9,
            np.log(0.5 * self.height),
        ]
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini - dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = np.linalg.norm(dpred - dobs) ** 2
            dmisfit = 2 * self.sim.Jtvec(
                m, dr
            )  # derivative of ||dpred - dobs||^2 gives factor of 2
            return misfit, dmisfit

        def derChk(m):
            return misfit(m, dobs)

        passed = tests.check_derivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-27, random_seed=42
        )
        assert passed


class TestEM1D_FD_Jacobian_LineCurrent:
    # Tests 2nd order convergence of Jvec and Jtvec for piecewise linear loop.
    # - All rx orientations
    # - All rx components
    # - Span many frequencies
    # - Tests derivatives wrt sigma, mu, thicknesses and h
    def setup_class(self):
        x_path = np.array([-2, -2, 2, 2, -2])
        y_path = np.array([-1, 1, 1, -1, -1])
        frequencies = np.logspace(0, 4)

        wire_paths = np.c_[x_path, y_path, np.ones(5) * 0.5]
        source_list = []
        receiver_list = []
        receiver_location = np.array([9.28, 0.0, 0.45])
        orientations = ["x", "y", "z"]
        components = ["real", "imag", "both"]

        # Define sources and receivers
        source_list = []
        for f in frequencies:
            receiver_list = []

            for rx_orientation in orientations:
                for comp in components:
                    receiver_list.append(
                        fdem.receivers.PointMagneticFieldSecondary(
                            receiver_location,
                            orientation=rx_orientation,
                            component=comp,
                        )
                    )

            source_list.append(fdem.sources.LineCurrent(receiver_list, f, wire_paths))

        # Survey
        survey = fdem.Survey(source_list)
        self.thicknesses = np.array([20.0, 40.0])

        self.nlayers = len(self.thicknesses) + 1
        wire_map = maps.Wires(
            ("sigma", self.nlayers),
            ("mu", self.nlayers),
            ("thicknesses", self.nlayers - 1),
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        self.mu_map = maps.ExpMap(nP=self.nlayers) * wire_map.mu
        self.thicknesses_map = maps.ExpMap(nP=self.nlayers - 1) * wire_map.thicknesses

        sim = fdem.Simulation1DLayered(
            survey=survey,
            sigmaMap=self.sigma_map,
            muMap=self.mu_map,
            thicknessesMap=self.thicknesses_map,
        )

        self.sim = sim

    def test_EM1DFDJvec_Layers(self):
        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[1] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 1.1 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[1] = mu_blk

        # General model
        m_1D = np.r_[np.log(sig), np.log(mu), np.log(self.thicknesses)]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        dm = m_1D * 0.5

        def derChk(m):
            return [fwdfun(m), lambda mx: jacfun(m, mx)]

        passed = tests.check_derivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15, random_seed=1123
        )
        assert passed

    def test_EM1DFDJtvec_Layers(self):
        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[1] = sigma_blk

        # Permeability
        mu_half = mu_0
        mu_blk = 1.1 * mu_0
        mu = np.ones(self.nlayers) * mu_half
        mu[1] = mu_blk

        # General model
        m_true = np.r_[np.log(sig), np.log(mu), np.log(self.thicknesses)]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * sigma_half),
            np.log(np.ones(self.nlayers) * mu_half),
            np.log(self.thicknesses) * 0.9,
        ]
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini - dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = np.linalg.norm(dpred - dobs) ** 2
            dmisfit = 2 * self.sim.Jtvec(
                m, dr
            )  # derivative of ||dpred - dobs||^2 gives factor of 2
            return misfit, dmisfit

        def derChk(m):
            return misfit(m, dobs)

        passed = tests.check_derivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-27, random_seed=124
        )
        assert passed
