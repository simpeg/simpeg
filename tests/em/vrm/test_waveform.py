import unittest
import numpy as np

from SimPEG.electromagnetics import viscous_remanent_magnetization as vrm


class VRM_waveform_tests(unittest.TestCase):
    def test_discrete(self):

        """
        Test ensures that if all different waveform classes are used to
        construct the same waveform, the characteristic decay they
        produce should be the same.
        """

        times = np.logspace(-4, -2, 3)

        t = np.r_[-0.00200001, -0.002, -0.0000000001, 0.0]
        I = np.r_[0.0, 1.0, 1.0, 0.0]

        waveObj1 = vrm.waveforms.SquarePulse(delt=0.002, t0=0.0)
        waveObj2 = vrm.waveforms.ArbitraryDiscrete(t_wave=t, I_wave=I)
        waveObj3 = vrm.waveforms.ArbitraryPiecewise(t_wave=t, I_wave=I)

        decay1b = waveObj1.getCharDecay("b", times)
        decay2b = waveObj2.getCharDecay("b", times)
        decay3b = waveObj3.getCharDecay("b", times)

        decay1dbdt = waveObj1.getCharDecay("dbdt", times)
        decay2dbdt = waveObj2.getCharDecay("dbdt", times)
        decay3dbdt = waveObj3.getCharDecay("dbdt", times)

        err1 = np.max(np.abs((decay2b - decay1b) / decay1b))
        err2 = np.max(np.abs((decay3b - decay1b) / decay1b))
        err3 = np.max(np.abs((decay2dbdt - decay1dbdt) / decay1dbdt))
        err4 = np.max(np.abs((decay3dbdt - decay1dbdt) / decay1dbdt))

        self.assertTrue(
            err1 < 0.01 and err2 < 0.01 and err3 < 0.025 and err4 < 0.01
        )

    def test_loguniform(self):

        """
        Tests to make sure log uniform decay and characteristic decay
        match of the range in which the approximation is valid.
        """

        times = np.logspace(-4, -2, 3)

        waveObj1 = vrm.waveforms.StepOff(t0=0.0)
        waveObj2 = vrm.waveforms.SquarePulse(delt=0.02)

        chi0 = np.array([0.0])
        dchi = np.array([0.01])
        tau1 = np.array([1e-10])
        tau2 = np.array([1e3])

        decay1b = (dchi / np.log(tau2 / tau1)) * waveObj2.getCharDecay("b", times)
        decay2b = waveObj2.getLogUniformDecay("b", times, chi0, dchi, tau1, tau2)

        decay1dbdt = (dchi / np.log(tau2 / tau1)) * waveObj1.getCharDecay("dbdt", times)
        decay2dbdt = waveObj1.getLogUniformDecay("dbdt", times, chi0, dchi, tau1, tau2)
        decay3dbdt = (dchi / np.log(tau2 / tau1)) * waveObj2.getCharDecay("dbdt", times)
        decay4dbdt = waveObj2.getLogUniformDecay("dbdt", times, chi0, dchi, tau1, tau2)

        err1 = np.max(np.abs((decay2b - decay1b) / decay1b))
        err2 = np.max(np.abs((decay2dbdt - decay1dbdt) / decay1dbdt))
        err3 = np.max(np.abs((decay4dbdt - decay3dbdt) / decay3dbdt))

        self.assertTrue(err1 < 0.01 and err2 < 0.01 and err3 < 0.01)


if __name__ == "__main__":
    unittest.main()
