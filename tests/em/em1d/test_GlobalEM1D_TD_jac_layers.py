from __future__ import print_function
import unittest
import numpy as np

import SimPEG.electromagnetics.time_domain as tdem
from SimPEG import *
from discretize import TensorMesh
from pymatsolver import PardisoSolver


np.random.seed(41)


class StitchedEM1DTM(unittest.TestCase):

    def setUp(self, parallel=True):

        np.random.seed(41)

        times = np.logspace(-5, -2, 31)
        dz = 1
        geometric_factor = 1.1
        n_layer = 20
        thicknesses = dz * geometric_factor ** np.arange(n_layer-1)
        n_layer = 20

        n_sounding = 5
        dx = 20.
        hx = np.ones(n_sounding) * dx
        hz = np.r_[thicknesses, thicknesses[-1]]
        mesh = TensorMesh([hx, hz], x0='00')
        inds = mesh.gridCC[:, 1] < 25
        inds_1 = mesh.gridCC[:, 1] < 50
        sigma = np.ones(mesh.nC) * 1./100.
        sigma[inds_1] = 1./10.
        sigma[inds] = 1./50.
        sigma_em1d = sigma.reshape(mesh.vnC, order='F').flatten()
        mSynth = np.log(sigma_em1d)

        x = mesh.vectorCCx
        y = np.zeros_like(x)
        z = np.ones_like(x) * 30.
        receiver_locations = np.c_[x, y, z]
        source_locations = np.c_[x, y, z]
        source_radius = 10.

        source_orientation = 'z'
        receiver_orientation = "z"  # "x", "y" or "z"

        topo = np.c_[x, y, z-30.].astype(float)

        sigma_map = maps.ExpMap(mesh)

        # Waveform
        waveform = tdem.sources.TriangularWaveform(
            startTime=-0.01, peakTime=-0.005, offTime=0.0
        )

        source_list = []

        for i_sounding in range(0, n_sounding):

            source_location = source_locations[i_sounding, :]
            receiver_location = receiver_locations[i_sounding, :]

            # Receiver list

            # Define receivers at each location.
            b_receiver = tdem.receivers.PointMagneticFluxDensity(
                receiver_location, times, receiver_orientation
            )
            dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times, receiver_orientation
            )
            receivers_list = [
                b_receiver, dbzdt_receiver
            ]  # Make a list containing all receivers even if just one

            # Must define the transmitter properties and associated receivers
            source_list.append(
                tdem.sources.CircularLoop(
                    receivers_list,
                    location=source_location,
                    waveform=waveform,
                    radius=source_radius,
                    i_sounding=i_sounding
                )
            )

        survey = tdem.Survey(source_list)

        simulation = tdem.Simulation1DLayeredStitched(
            survey=survey, thicknesses=thicknesses, sigmaMap=sigma_map,
            topo=topo, parallel=False, n_cpu=2, verbose=False, solver=PardisoSolver
        )

        dpred = simulation.dpred(mSynth)
        noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
        uncertainties = 0.1*np.abs(dpred)*np.ones(np.shape(dpred))
        dobs =  dpred + noise
        data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)

        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
        dmis.W = 1./uncertainties

        reg = regularization.Tikhonov(mesh)

        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0.)
        inv = inversion.BaseInversion(invProb)

        self.data = data_object
        self.dmis = dmis
        self.inv = inv
        self.reg = reg
        self.sim = simulation
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey


    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: (
                self.sim.dpred(m),
                lambda mx: self.sim.Jvec(self.m0, mx)
            ),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.data.dobs.shape[0])
        wtJv = w.dot(self.sim.Jvec(self.m0, v))
        vtJtw = v.dot(self.sim.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

# class StitchedEM1DTMHeight(unittest.TestCase):

#     def setUp(self, parallel=True):

#         times = np.logspace(-5, -2, 31)

#         source_radius = 1.
#         hz = 1.
#         n_sounding = 10
#         dx = 20.
#         hx = np.ones(n_sounding) * dx
#         e = np.ones(n_sounding)
#         mSynth = np.r_[e*np.log(1./100.), e*30]
#         mesh = TensorMesh([hx, hz], x0='00')

#         wires = maps.Wires(('sigma', n_sounding),('height', n_sounding))
#         expmap = maps.ExpMap(nP=n_sounding)
#         sigma_map = expmap * wires.sigma

#         x = mesh.vectorCCx
#         y = np.zeros_like(x)
#         z = np.ones_like(x) * 30.
#         receiver_locations = np.c_[x, y, z]
#         source_locations = np.c_[x, y, z]
#         topo = np.c_[x, y, z-30.].astype(float)

#         source_orientation = 'z'
#         receiver_orientation = "z"  # "x", "y" or "z"


#         # Waveform
#         waveform_times = np.r_[-np.logspace(-2, -5, 31), 0.]
#         waveform_current = triangular_waveform_current(
#             waveform_times, -0.01, -0.005, 0., 1.
#         )

#         waveform = tdem.sources.RawWaveform(
#                 waveform_times=waveform_times, waveform_current=waveform_current,
#                 n_pulse = 1, base_frequency = 25.,  high_cut_frequency=210*1e3
#         )

#         source_list = []

#         for i_sounding in range(0, n_sounding):

#             source_location = mkvc(source_locations[i_sounding, :])
#             receiver_location = mkvc(receiver_locations[i_sounding, :])

#             # Receiver list

#             # Define receivers at each location.
#             b_receiver = tdem.receivers.PointMagneticFluxDensity(
#                 receiver_location, times, receiver_orientation
#             )
#             dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
#                 receiver_location, times, receiver_orientation
#             )
#             receivers_list = [
#                 b_receiver, dbzdt_receiver
#             ]  # Make a list containing all receivers even if just one

#             # Must define the transmitter properties and associated receivers
#             source_list.append(
#                 tdem.sources.CircularLoop(
#                     receivers_list,
#                     location=source_location,
#                     waveform=waveform,
#                     radius=source_radius,
#                     i_sounding=i_sounding
#                 )
#             )

#         survey = tdem.Survey(source_list)

#         simulation = em1d.simulation.StitchedEM1DTMSimulation(
#             survey=survey, sigmaMap=sigma_map, hMap=wires.height,
#             topo=topo, parallel=False, n_cpu=2, verbose=False, solver=PardisoSolver
#         )

#         dpred = simulation.dpred(mSynth)
#         noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
#         uncertainties = 0.1*np.abs(dpred)*np.ones(np.shape(dpred))
#         dobs =  dpred + noise
#         data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)

#         dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
#         dmis.W = 1./uncertainties

#         reg_mesh = TensorMesh([int(n_sounding)])
#         reg_sigma = regularization.Tikhonov(reg_mesh, mapping=wires.sigma)
#         reg_height = regularization.Tikhonov(reg_mesh, mapping=wires.height)

#         reg = reg_sigma + reg_height

#         opt = optimization.InexactGaussNewton(
#             maxIterLS=20, maxIter=10, tolF=1e-6,
#             tolX=1e-6, tolG=1e-6, maxIterCG=6
#         )

#         invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0.)
#         inv = inversion.BaseInversion(invProb)

#         self.data = data_object
#         self.dmis = dmis
#         self.inv = inv
#         self.reg = reg
#         self.sim = simulation
#         self.mesh = reg_mesh
#         self.m0 = mSynth * 1.2
#         self.survey = survey


#     def test_misfit(self):
#         passed = tests.checkDerivative(
#             lambda m: (
#                 self.sim.dpred(m),
#                 lambda mx: self.sim.Jvec(self.m0, mx)
#             ),
#             self.m0,
#             plotIt=False,
#             num=3
#         )
#         self.assertTrue(passed)

#     def test_adjoint(self):
#         # Adjoint Test
#         v = np.random.rand(2*self.mesh.nC)
#         w = np.random.rand(self.data.dobs.shape[0])
#         wtJv = w.dot(self.sim.Jvec(self.m0, v))
#         vtJtw = v.dot(self.sim.Jtvec(self.m0, w))
#         passed = np.abs(wtJv - vtJtw) < 1e-10
#         print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
#         self.assertTrue(passed)

#     def test_dataObj(self):
#         passed = tests.checkDerivative(
#             lambda m: [self.dmis(m), self.dmis.deriv(m)],
#             self.m0,
#             plotIt=False,
#             num=3
#         )
#         self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
