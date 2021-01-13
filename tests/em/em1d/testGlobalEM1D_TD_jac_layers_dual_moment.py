from __future__ import print_function
import unittest
import numpy as np
import SimPEG.electromagnetics.em1d as em1d
from SimPEG.electromagnetics.em1d.utils import get_vertical_discretization_time
from SimPEG.electromagnetics.em1d.waveforms import TriangleFun
from SimPEG import *
from discretize import TensorMesh
from pymatsolver import PardisoSolver
from SimPEG.electromagnetics.em1d import skytem_HM_2015, skytem_LM_2015


np.random.seed(41)


class GlobalEM1DTD(unittest.TestCase):

    def setUp(self, parallel=True):
        wave_HM = skytem_HM_2015()
        wave_LM = skytem_LM_2015()
        time_HM = wave_HM.time_gate_center[0::2]
        time_LM = wave_LM.time_gate_center[0::2]

        time_input_currents_HM = wave_HM.current_times[-7:]
        input_currents_HM = wave_HM.currents[-7:]
        time_input_currents_LM = wave_LM.current_times[-13:]
        input_currents_LM = wave_LM.currents[-13:]

        n_layer = 20
        thicknesses = get_vertical_discretization_time(
            time_LM, facter_tmax=0.5, factor_tmin=10., n_layer=n_layer-1
        )

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
        source_locations = np.c_[x, y, z]
        source_current = 1.
        source_orientation = 'z'
        receiver_offset_r = 13.25
        receiver_offset_z = 2.

        receiver_locations = np.c_[x+receiver_offset_r, np.zeros(n_sounding), 30.*np.ones(n_sounding)+receiver_offset_z]
        receiver_orientation = "z"  # "x", "y" or "z"

        topo = np.c_[x, y, z-30.].astype(float)

        sigma_map = maps.ExpMap(mesh)

        source_list = []

        for ii in range(0, n_sounding):

            source_location = mkvc(source_locations[ii, :])
            receiver_location = mkvc(receiver_locations[ii, :])

            receiver_list = []

            receiver_list = [
                em1d.receivers.TimeDomainPointReceiver(
                    receiver_location,
                    times=time_HM,
                    times_dual_moment=time_LM,
                    orientation=receiver_orientation,
                    component="dbdt"
                )
            ]

            receiver_list = [
                em1d.receivers.TimeDomainPointReceiver(
                    receiver_location,
                    times=time_HM,
                    times_dual_moment=time_LM,
                    orientation=receiver_orientation,
                    component="b"
                )
            ]

            source_list.append(
                em1d.sources.TimeDomainMagneticDipoleSource(
                    receiver_list=receiver_list,
                    location=source_location,
                    moment_amplitude=source_current,
                    orientation=source_orientation,
                    wave_type="general",
                    moment_type='dual',
                    time_input_currents=time_input_currents_HM,
                    input_currents=input_currents_HM,
                    n_pulse = 1,
                    base_frequency = 25.,
                    time_input_currents_dual_moment = time_input_currents_LM,
                    input_currents_dual_moment = input_currents_LM,
                    base_frequency_dual_moment = 210
                )
            )

        survey = em1d.survey.EM1DSurveyTD(source_list)

        simulation = em1d.simulation.StitchedEM1DTMSimulation(
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

if __name__ == '__main__':
    unittest.main()
