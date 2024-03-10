from __future__ import print_function
import unittest
import numpy as np
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG import maps, tests
from discretize import TensorMesh
from pymatsolver import PardisoSolver

np.random.seed(41)


class STITCHED_EM1D_TD_Jacobian_Test_MagDipole(unittest.TestCase):
    def setUp(self, parallel=False):
        times_hm = np.logspace(-6, -3, 31)
        times_lm = np.logspace(-5, -2, 31)

        # Waveforms
        waveform_hm = tdem.sources.TriangularWaveform(
            start_time=-0.01, peak_time=-0.005, off_time=0.0
        )
        waveform_lm = tdem.sources.TriangularWaveform(
            start_time=-0.01, peak_time=-0.0001, off_time=0.0
        )

        dz = 1
        geometric_factor = 1.1
        n_layer = 20
        thicknesses = dz * geometric_factor ** np.arange(n_layer - 1)
        n_layer = 20

        n_sounding = 5
        dx = 20.0
        hx = np.ones(n_sounding) * dx
        hz = np.r_[thicknesses, thicknesses[-1]]
        mesh = TensorMesh([hx, hz], x0="00")
        inds = mesh.cell_centers[:, 1] < 25
        inds_1 = mesh.cell_centers[:, 1] < 50
        sigma = np.ones(mesh.nC) * 1.0 / 100.0
        sigma[inds_1] = 1.0 / 10.0
        sigma[inds] = 1.0 / 50.0

        x = mesh.cell_centers_x
        y = np.zeros_like(x)
        z = np.ones_like(x) * 30.0
        source_locations = np.c_[x, y, z]
        source_orientation = "z"

        receiver_offset_r = 13.25
        receiver_offset_z = 2.0

        receiver_locations = np.c_[
            x + receiver_offset_r,
            np.zeros(n_sounding),
            30.0 * np.ones(n_sounding) + receiver_offset_z,
        ]
        receiver_orientation = "z"  # "x", "y" or "z"

        topo = np.c_[x, y, z - 30.0].astype(float)

        source_list = []

        for i_sounding in range(0, n_sounding):
            source_location = source_locations[i_sounding, :]
            receiver_location = receiver_locations[i_sounding, :]

            # Receiver list

            # Define receivers at each location.
            dbzdt_receiver_hm = tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times_hm, receiver_orientation
            )
            dbzdt_receiver_lm = tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times_lm, receiver_orientation
            )
            # Make a list containing all receivers even if just one

            # Must define the transmitter properties and associated receivers
            source_list.append(
                tdem.sources.MagDipole(
                    [dbzdt_receiver_hm],
                    location=source_location,
                    waveform=waveform_hm,
                    orientation=source_orientation,
                    i_sounding=i_sounding,
                )
            )

            source_list.append(
                tdem.sources.MagDipole(
                    [dbzdt_receiver_lm],
                    location=source_location,
                    waveform=waveform_lm,
                    orientation=source_orientation,
                    i_sounding=i_sounding,
                )
            )
        survey = tdem.Survey(source_list)
        wires = maps.Wires(("sigma", n_layer * n_sounding), ("h", n_sounding))
        sigmaMap = maps.ExpMap(nP=n_layer * n_sounding) * wires.sigma
        hMap = maps.ExpMap(nP=n_sounding) * wires.h

        simulation = tdem.Simulation1DLayeredStitched(
            survey=survey,
            thicknesses=thicknesses,
            sigmaMap=sigmaMap,
            hMap=hMap,
            topo=topo,
            parallel=False,
            n_cpu=2,
            verbose=False,
            solver=PardisoSolver,
        )

        self.sim = simulation
        self.mesh = mesh

    def test_EM1TDJvec_Layers(self):
        # Conductivity
        inds = self.mesh.cell_centers[:, 1] < 25
        inds_1 = self.mesh.cell_centers[:, 1] < 50
        sigma = np.ones(self.mesh.n_cells) * 1.0 / 100.0
        sigma[inds_1] = 1.0 / 10.0
        sigma[inds] = 1.0 / 50.0
        sigma_em1d = sigma.reshape(self.mesh.vnC, order="F").flatten()
        m_stitched = np.r_[
            np.log(sigma_em1d), np.ones(self.sim.n_sounding) * np.log(30.0)
        ]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        def derChk(m):
            return [fwdfun(m), lambda mx: jacfun(m, mx)]

        dm = m_stitched * 0.5

        passed = tests.check_derivative(
            derChk, m_stitched, num=4, dx=dm, plotIt=False, eps=1e-15
        )
        self.assertTrue(passed)
        if passed:
            print("STITCHED EM1DFM MagDipole Jvec test works")

    def test_EM1TDJtvec_Layers(self):
        # Conductivity
        inds = self.mesh.cell_centers[:, 1] < 25
        inds_1 = self.mesh.cell_centers[:, 1] < 50
        sigma = np.ones(self.mesh.n_cells) * 1.0 / 100.0
        sigma[inds_1] = 1.0 / 10.0
        sigma[inds] = 1.0 / 50.0
        sigma_em1d = sigma.reshape(self.mesh.vnC, order="F").flatten()
        m_stitched = m_stitched = np.r_[
            np.log(sigma_em1d), np.ones(self.sim.n_sounding) * np.log(30.0)
        ]

        dobs = self.sim.dpred(m_stitched)

        m_ini = np.r_[
            np.log(1.0 / 100.0) * np.ones(self.mesh.n_cells),
            np.ones(self.sim.n_sounding) * np.log(30.0) * 1.5,
        ]
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini - dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = 0.5 * np.linalg.norm(dpred - dobs) ** 2
            dmisfit = self.sim.Jtvec(m, dr)
            return misfit, dmisfit

        def derChk(m):
            return misfit(m, dobs)

        passed = tests.check_derivative(derChk, m_ini, num=4, plotIt=False, eps=1e-27)
        self.assertTrue(passed)
        if passed:
            print("STITCHED EM1DFM MagDipole Jtvec test works")


if __name__ == "__main__":
    unittest.main()
