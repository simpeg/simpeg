from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation
from ..frequency_domain_1d.supporting_functions.kernels import *
import numpy as np
from .sources import *
from .survey import EM1DSurveyTD
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from empymod.utils import check_time
from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel

from .supporting_functions.waveform_functions import (
    piecewise_pulse_fast,
    butterworth_type_filter, butter_lowpass_filter
)


class EM1DTMSimulation(BaseEM1DSimulation):
    """
    Simulation class for simulating the TEM response over a 1D layered Earth
    for a single sounding.
    """


    time_intervals_are_set = False
    frequencies_are_set = False
    time_filter = 'key_81_CosSin_2009'


    def __init__(self, **kwargs):
        BaseEM1DSimulation.__init__(self, **kwargs)
        if self.time_filter == 'key_81_CosSin_2009':
            self.fftfilt = filters.key_81_CosSin_2009()
        elif self.time_filter == 'key_201_CosSin_2012':
            self.fftfilt = filters.key_201_CosSin_2012()
        elif self.time_filter == 'key_601_CosSin_2012':
            self.fftfilt = filters.key_601_CosSin_2009()
        else:
            raise Exception()



    def set_time_intervals(self):
        """
        Define time interval for all source-receiver pairs.
        """

        for src in self.survey.source_list:
            waveform = src.waveform
            if src.waveform.wave_type != "stepoff":
                for rx in src.receiver_list:

                    if waveform.wave_type == "general":
                        time = rx.times
                        pulse_period = waveform.pulse_period
                        period = waveform.period
                    # Dual moment
                    else:
                        time = np.unique(np.r_[rx.times, rx.dual_times])
                        pulse_period = np.maximum(
                            waveform.pulse_period, waveform.dual_pulse_period
                        )
                        period = np.maximum(waveform.period, waveform.dual_period)
                    tmin = time[time>0.].min()
                    if waveform.n_pulse == 1:
                        tmax = time.max() + pulse_period
                    elif waveform.n_pulse == 2:
                        tmax = time.max() + pulse_period + period/2.
                    else:
                        raise NotImplementedError("n_pulse must be either 1 or 2")
                    n_time = int((np.log10(tmax)-np.log10(tmin))*10+1)

                    rx.time_interval = np.logspace(
                        np.log10(tmin), np.log10(tmax), n_time
                    )

        self.time_intervals_are_set = True


    def set_frequencies(self, pts_per_dec=-1):
        """
        Set frequencies required for Hankel transform computation and for accurate
        computation of the IFFT.
        """

        # Set range of time channels
        if self.time_intervals_are_set == False:
            self.set_time_intervals()

        for src in self.survey.source_list:
            for rx in src.receiver_list:

                if src.waveform.wave_type == "stepoff":
                    _, freq, ft, ftarg = check_time(
                        rx.times, -1, 'dlf',
                        {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0,
                    )

                else:
                    _, freq, ft, ftarg = check_time(
                        rx.time_interval, -1, 'dlf',
                        {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0
                    )

                rx.frequencies = freq
                rx.ftarg = ftarg

        self.frequencies_are_set = True


    def compute_integral(self, m, output_type='response'):
        """
        This method evaluates the Hankel transform for each source and
        receiver and outputs it as a list. Used for computing response
        or sensitivities.
        """

        self.model = m
        n_layer = self.n_layer
        n_filter = self.n_filter

        # For time-domain simulations, set frequencies for the evaluation
        # of the Hankel transform.
        if self.frequencies_are_set is False:
            self.set_frequencies()


        # Define source height above topography by mapping or from sources and receivers.
        if self.hMap is not None:
            h_vector = np.array(self.h)
        else:
            if self.topo is None:
                h_vector = np.array([src.location[2] for src in self.survey.source_list])
            else:
                h_vector = np.array(
                    [src.location[2]-self.topo[-1] for src in self.survey.source_list]
                )


        integral_output_list = []

        for ii, src in enumerate(self.survey.source_list):
            for jj, rx in enumerate(src.receiver_list):

                n_frequency = len(rx.frequencies)

                f = np.empty([n_frequency, n_filter], order='F')
                f[:, :] = np.tile(
                    rx.frequencies.reshape([-1, 1]), (1, n_filter)
                )

                # Create globally, not for each receiver in the future
                sig = self.compute_sigma_matrix(rx.frequencies)
                chi = self.compute_chi_matrix(rx.frequencies)

                # Compute receiver height
                h = h_vector[ii]
                if rx.use_source_receiver_offset:
                    z = h + rx.locations[2]
                else:
                    z = h + rx.locations[2] - src.location[2]

                # Hankel transform for x, y or z magnetic dipole source
                if isinstance(src, MagneticDipoleSource):

                    # Radial distance
                    if rx.use_source_receiver_offset:
                        r = rx.locations[0:2]
                    else:
                        r = rx.locations[0:2] - src.location[0:2]

                    r = np.sqrt(np.sum(r**2))
                    r_vec = r * np.ones(n_frequency)

                    # Use function from empymod to define Hankel coefficients.
                    # Size of lambd is (n_frequency x n_filter)

                    lambd = np.empty([n_frequency, n_filter], order='F')
                    lambd[:, :], _ = get_dlf_points(
                        self.fhtfilt, r_vec, self.hankel_pts_per_dec
                    )

                    # Get kernel function(s) at all lambda and frequencies
                    PJ = magnetic_dipole_kernel(
                        self, lambd, f, n_layer, sig, chi, h, z, r, src, rx, output_type
                    )

                    PJ = tuple(PJ)

                    if output_type=="sensitivity_sigma":
                        r_vec = np.tile(r_vec, (n_layer, 1))

                    # Evaluate Hankel transform using digital linear filter from empymod
                    integral_output = dlf(
                        PJ, lambd, r_vec, self.fhtfilt, self.hankel_pts_per_dec, ang_fact=None, ab=33
                    )

                # Hankel transform for horizontal loop source
                elif isinstance(src, HorizontalLoopSource):

                    # radial distance (r) and loop radius (a)
                    if rx.use_source_receiver_offset:
                        r = rx.locations[0:2]
                    else:
                        r = rx.locations[0:2] - src.location[0:2]

                    r_vec = np.sqrt(np.sum(r**2)) * np.ones(n_frequency)
                    a_vec = src.radius * np.ones(n_frequency)

                    # Use function from empymod to define Hankel coefficients.
                    # Size of lambd is (n_frequency x n_filter)
                    lambd = np.empty([n_frequency, n_filter], order='F')
                    lambd[:, :], _ = get_dlf_points(
                        self.fhtfilt, a_vec, self.hankel_pts_per_dec
                    )

                    # Get kernel function(s) at all lambda and frequencies
                    hz = horizontal_loop_kernel(
                        self, lambd, f, n_layer, sig, chi, a_vec, h, z, r,
                        src, rx, output_type
                    )

                    # kernels associated with each bessel function (j0, j1, j2)
                    PJ = (None, hz, None)  # PJ1

                    if output_type == "sensitivity_sigma":
                        a_vec = np.tile(a_vec, (n_layer, 1))

                    # Evaluate Hankel transform using digital linear filter from empymod
                    integral_output = dlf(
                        PJ, lambd, a_vec, self.fhtfilt, self.hankel_pts_per_dec, ang_fact=None, ab=33
                    )

                if output_type == "sensitivity_sigma":
                    integral_output_list.append(integral_output.T)
                else:
                    integral_output_list.append(integral_output)

        return integral_output_list


    def project_fields(self, u, output_type=None):
        """
        Project from the list of Hankel transform evaluations to the data or sensitivities.

        :param list u: list containing Hankel transform outputs for each unique
        source-receiver pair.
        :rtype: list: list containing predicted data for each unique
        source-receiver pair.
        :return: predicted data or sensitivities by source-receiver
        """

        COUNT = 0
        for ii, src in enumerate(self.survey.source_list):

            for jj, rx in enumerate(src.receiver_list):

                u_temp = u[COUNT]

                # use low-pass filter
                if src.waveform.use_lowpass_filter:
                    factor = src.waveform.lowpass_filter.copy()
                else:
                    factor = np.ones_like(rx.frequencies, dtype=complex)

                # Multiplication factors
                if rx.component in ["b", "h"]:
                    factor *= 1./(2j*np.pi*rx.frequencies)

                if rx.component in ["b", "dbdt"]:
                    factor *= mu_0

                # For stepoff waveform
                if src.waveform.wave_type == 'stepoff':

                    # Compute EM responses
                    if u_temp.ndim == 1:
                        resp, _ = fourier_dlf(
                            u_temp.flatten()*factor, rx.times, rx.frequencies, rx.ftarg
                        )

                    # Compute EM sensitivities
                    else:

                        resp = np.zeros(
                            (rx.n_time, self.n_layer), dtype=np.float64, order='F'
                        )
                        # TODO: remove for loop
                        for i in range(0, self.n_layer):
                            resp_i, _ = fourier_dlf(
                                u_temp[:, i]*factor, rx.times, rx.frequencies, rx.ftarg
                            )
                            resp[:, i] = resp_i

                # For general waveform.
                # Evaluate piecewise linear input current waveforms
                # Using Fittermann's approach (19XX) with Gaussian Quadrature
                else:

                    # Compute EM responses
                    if u_temp.ndim == 1:
                        resp_int, _ = fourier_dlf(
                            u_temp.flatten()*factor, rx.time_interval, rx.frequencies, rx.ftarg
                        )
                        # step_func = interp1d(
                        #     self.time_int, resp_int
                        # )
                        step_func = iuSpline(
                            np.log10(rx.time_interval), resp_int
                        )

                        resp = piecewise_pulse_fast(
                            step_func, rx.times,
                            src.waveform.waveform_times,
                            src.waveform.waveform_current,
                            src.waveform.period,
                            n_pulse=src.waveform.n_pulse
                        )

                        # Compute response for the dual moment
                        if src.waveform.wave_type == "dual":
                            resp_dual_moment = piecewise_pulse_fast(
                                step_func, rx.dual_times,
                                src.waveform.dual_waveform_times,
                                src.waveform.dual_waveform_current,
                                src.waveform.dual_period,
                                n_pulse=src.waveform.n_pulse
                            )
                            # concatenate dual moment response
                            # so, ordering is the first moment data
                            # then the second moment data.
                            resp = np.r_[resp, resp_dual_moment]

                    # Compute EM sensitivities
                    else:
                        # if src.moment_type == "single":
                        resp = np.zeros(
                            (rx.n_time, self.n_layer), dtype=np.float64, order='F'
                        )
                        # else:
                        #     # For dual moment
                        #     resp = np.zeros(
                        #         (rx.n_time, self.n_layer),
                        #         dtype=np.float64, order='F'
                        #     )

                        # TODO: remove for loop (?)
                        for i in range(self.n_layer):
                            resp_int_i, _ = fourier_dlf(
                                u_temp[:, i]*factor, rx.time_interval, rx.frequencies, rx.ftarg
                            )
                            # step_func = interp1d(
                            #     self.time_int, resp_int_i
                            # )

                            step_func = iuSpline(
                                np.log10(rx.time_interval), resp_int_i
                            )

                            resp_i = piecewise_pulse_fast(
                                step_func, rx.times,
                                src.waveform.waveform_times,
                                src.waveform.waveform_current,
                                src.waveform.period,
                                n_pulse=src.waveform.n_pulse
                            )

                            if src.waveform.wave_type != "dual":
                                resp[:, i] = resp_i
                            else:
                                resp_dual_moment_i = piecewise_pulse_fast(
                                    step_func,
                                    rx.dual_times,
                                    src.waveform.dual_waveform_times,
                                    src.waveform.dual_waveform_current,
                                    src.waveform.dual_period,
                                    n_pulse=src.waveform.n_pulse
                                )
                                resp[:, i] = np.r_[resp_i, resp_dual_moment_i]

                u[COUNT] = resp * (-2.0/np.pi)
                COUNT = COUNT + 1

        return u



#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################

def dot(args):
    return np.dot(args[0], args[1])

def run_simulation_TD(args):
    """
    This method simulates the EM response or computes the sensitivities for
    a single sounding. The method allows for parallelization of
    the stitched 1D problem.

    :param src: a EM1DTM source object
    :param topo: Topographic location (x, y, z)
    :param np.array thicknesses: np.array(N-1,) layer thicknesses for a single sounding
    :param np.array sigma: np.array(N,) layer conductivities for a single sounding
    :param np.array eta: np.array(N,) intrinsic chargeabilities for a single sounding
    :param np.array tau: np.array(N,) Cole-Cole time constant for a single sounding
    :param np.array c: np.array(N,) Cole-Cole frequency distribution constant for a single sounding
    :param np.array chi: np.array(N,) magnetic susceptibility for a single sounding
    :param np.array dchi: np.array(N,) DC susceptibility for magnetic viscosity for a single sounding
    :param np.array tau1: np.array(N,) lower time-relaxation constant for magnetic viscosity for a single sounding
    :param np.array tau2: np.array(N,) upper time-relaxation constant for magnetic viscosity for a single sounding
    :param float h: source height for a single sounding
    :param string output_type: "response", "sensitivity_sigma", "sensitivity_height"
    :param bool invert_height: boolean switch for inverting for source height
    :return: response or sensitivities

    """

    src, topo, thicknesses, sigma, eta, tau, c, chi, dchi, tau1, tau2, h, output_type, invert_height = args

    n_layer = len(thicknesses) + 1
    local_survey = EM1DSurveyTD([src])
    exp_map = maps.ExpMap(nP=n_layer)

    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment
        sim = EM1DTMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=exp_map, eta=eta, tau=tau, c=c, chi=chi, dchi=dchi, tau1=tau1, tau2=tau2,
            topo=topo, hankel_filter='key_101_2009'
        )

        if output_type == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(np.log(sigma))
            return utils.mkvc(drespdsig * sim.sigmaDeriv)
        else:
            resp = sim.dpred(np.log(sigma))
            return resp
    else:

        wires = maps.Wires(('sigma', n_layer), ('h', 1))
        sigma_map = exp_map * wires.sigma
        sim = EM1DTMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=sigma_map, hMap=wires.h, topo=topo,
            eta=eta, tau=tau, c=c, chi=chi, dchi=dchi, tau1=tau1, tau2=tau2,
            hankel_filter='key_101_2009'
        )

        m = np.r_[np.log(sigma), h]
        if output_type == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(m)
            return utils.mkvc(drespdsig * utils.sdiag(sigma))
        elif output_type == 'sensitivity_height':
            drespdh = sim.getJ_height(m)
            return utils.mkvc(drespdh)
        else:
            resp = sim.dpred(m)
            return resp


class StitchedEM1DTMSimulation(BaseStitchedEM1DSimulation):

    def run_simulation(self, args):
        if self.verbose:
            print(">> Time-domain")
        return run_simulation_TD(args)







