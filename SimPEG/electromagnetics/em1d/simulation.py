from SimPEG import maps, utils, props
from SimPEG.simulation import BaseSimulation
import numpy as np
from .sources import *
from .survey import BaseEM1DSurvey, EM1DSurveyFD, EM1DSurveyTD
from .supporting_functions.kernels import *
from scipy import sparse as sp
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.linalg import block_diag
import properties

from empymod.utils import check_time
from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel

from .known_waveforms import (
    piecewise_pulse_fast,
    butterworth_type_filter, butter_lowpass_filter
)

try:
    from multiprocessing import Pool
    from sys import platform
except ImportError:
    print("multiprocessing is not available")
    PARALLEL = False
else:
    PARALLEL = True
    import multiprocessing


#######################################################################
#               SIMULATION FOR A SINGLE SOUNDING
#######################################################################


class BaseEM1DSimulation(BaseSimulation):
    """
    Base simulation class for simulating the EM response over a 1D layered Earth
    for a single sounding. The simulation computes the fields by solving the
    Hankel transform solutions from Electromagnetic Theory for Geophysical
    Applications: Chapter 4 (Ward and Hohmann, 1988).
    """

    hankel_filter = 'key_101_2009'  # Default: Hankel filter
    hankel_pts_per_dec = None       # Default: Standard DLF
    verbose = False
    fix_Jmatrix = False
    _Jmatrix_sigma = None
    _Jmatrix_height = None
    _pred = None

    # Properties for electrical conductivity/resistivity
    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )

    rho, rhoMap, rhoDeriv = props.Invertible(
        "Electrical resistivity (Ohm m)"
    )

    props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = props.Invertible(
        "Intrinsic chargeability (V/V), 0 <= eta < 1",
        default=0.
    )

    tau, tauMap, tauDeriv = props.Invertible(
        "Time constant for Cole-Cole model (s)",
        default=1.
    )

    c, cMap, cDeriv = props.Invertible(
        "Frequency Dependency for Cole-Cole model, 0 < c < 1",
        default=0.5
    )

    # Properties for magnetic susceptibility
    chi, chiMap, chiDeriv = props.Invertible(
        "Magnetic susceptibility at infinite frequency (SI)",
        default=0.
    )

    dchi, dchiMap, dchiDeriv = props.Invertible(
        "DC magnetic susceptibility for viscous remanent magnetization contribution (SI)",
        default=0.
    )

    tau1, tau1Map, tau1Deriv = props.Invertible(
        "Lower bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
        default=1e-10
    )

    tau2, tau2Map, tau2Deriv = props.Invertible(
        "Upper bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
        default=10.
    )

    # Additional properties
    h, hMap, hDeriv = props.Invertible(
        "Receiver Height (m), h > 0",
    )

    survey = properties.Instance(
        "a survey object", BaseEM1DSurvey, required=True
    )

    topo = properties.Array("Topography (x, y, z)", dtype=float)

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "layer thicknesses (m)", default=np.array([])
    )

    def __init__(self, **kwargs):
        BaseSimulation.__init__(self, **kwargs)

        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).
        ht, htarg = check_hankel(
            'dlf',
            {
                'dlf': self.hankel_filter,
                'pts_per_dec': 0
            },
            1
        )

        self.fhtfilt = htarg['dlf']                 # Store filter
        self.hankel_pts_per_dec = htarg['pts_per_dec']      # Store pts_per_dec
        if self.verbose:
            print(">> Use "+self.hankel_filter+" filter for Hankel Transform")


    @property
    def halfspace_switch(self):
        """True = halfspace, False = layered Earth"""
        if (self.thicknesses is None) | (len(self.thicknesses)==0):
            return True
        else:
            return False

    @property
    def n_layer(self):
        """number of layers"""
        if self.halfspace_switch is False:
            return int(self.thicknesses.size + 1)
        elif self.halfspace_switch is True:
            return int(1)

    @property
    def n_filter(self):
        """ Length of filter """
        return self.fhtfilt.base.size

    @property
    def depth(self):
        """layer depths"""
        if self.thicknesses is not None:
            return np.r_[0., -np.cumsum(self.thicknesses)]


    def compute_sigma_matrix(self, frequencies):
        """
        Computes the complex conductivity matrix using Pelton's Cole-Cole model:

        .. math ::
            \\sigma (\\omega ) = \\sigma \\Bigg [
            1 - \\eta \\Bigg ( \\frac{1}{1 + (1-\\eta ) (1 + i\\omega \\tau)^c} \\Bigg )
            \\Bigg ]

        :param numpy.array frequencies: np.array(N,) containing frequencies
        :rtype: numpy.ndarray: np.array(n_layer, n_frequency)
        :return: complex conductivity matrix

        """
        n_layer = self.n_layer
        n_frequency = len(frequencies)
        # n_filter = self.n_filter

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))

        # No IP effect
        if np.all(self.eta) == 0.:
            return sigma

        # IP effect
        else:

            if np.isscalar(self.eta):
                eta = self.eta
                tau = self.tau
                c = self.c
            else:
                eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
                tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
                c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

            w = np.tile(
                2*np.pi*frequencies,
                (n_layer, 1)
            )

            sigma_complex = np.empty(
                [n_layer, n_frequency], dtype=np.complex128, order='F'
            )
            sigma_complex[:, :] = (
                sigma -
                sigma*eta/(1+(1-eta)*(1j*w*tau)**c)
            )

            return sigma_complex


    def compute_chi_matrix(self, frequencies):
        """
        Computes the complex magnetic susceptibility matrix assuming a log-uniform
        distribution of time-relaxation constants:

        .. math::
            \\chi (\\omega ) = \\chi + \\Delta \\chi \\Bigg [
            1 - \\Bigg ( \\frac{1}{ln (\\tau_2 / \\tau_1 )} \\Bigg )
            ln \\Bigg ( \\frac{1 + i\\omega \\tau_2}{1 + i\\omega tau_1} ) \\Bigg )
            \\Bigg ]

        :param numpy.array frequencies: np.array(N,) containing frequencies
        :rtype: numpy.ndarray: np.array(n_layer, n_frequency)
        :return: complex magnetic susceptibility matrix

        """

        if np.isscalar(self.chi):
            chi = np.ones_like(self.sigma) * self.chi
        else:
            chi = self.chi

        n_layer = self.n_layer
        n_frequency = len(frequencies)
        # n_filter = self.n_filter
     
        chi = np.tile(chi.reshape([-1, 1]), (1, n_frequency))

        # No magnetic viscosity
        if np.all(self.dchi) == 0.:

            

            return chi

        # Magnetic viscosity
        else:

            if np.isscalar(self.dchi):
                dchi = self.dchi * np.ones_like(self.chi)
                tau1 = self.tau1 * np.ones_like(self.chi)
                tau2 = self.tau2 * np.ones_like(self.chi)
            else:
                dchi = np.tile(self.dchi.reshape([-1, 1]), (1, n_frequency))
                tau1 = np.tile(self.tau1.reshape([-1, 1]), (1, n_frequency))
                tau2 = np.tile(self.tau2.reshape([-1, 1]), (1, n_frequency))

            w = np.tile(
                2*np.pi*frequencies,
                (n_layer, 1)
            )

            chi_complex = np.empty(
                [n_layer, n_frequency], dtype=np.complex128, order='F'
            )
            chi_complex[:, :] = chi + dchi*(
                1 - (np.log(tau2/tau1))**-1 * np.log(
                    (1 + 1j*w*tau2)/(1 + 1j*w*tau1)
                )
            )

            return chi_complex


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
        if isinstance(self.survey, EM1DSurveyTD):
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
                if isinstance(src, HarmonicMagneticDipoleSource) | isinstance(src, TimeDomainMagneticDipoleSource):

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
                elif isinstance(src, HarmonicHorizontalLoopSource) | isinstance(src, TimeDomainHorizontalLoopSource):

                    # radial distance (r) and loop radius (a)
                    if rx.use_source_receiver_offset:
                        r = rx.locations[0:2]
                    else:
                        r = rx.locations[0:2] - src.location[0:2]

                    r_vec = np.sqrt(np.sum(r**2)) * np.ones(n_frequency)
                    a_vec = src.a * np.ones(n_frequency)

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


    def fields(self, m):
        f = self.compute_integral(m, output_type='response')
        f = self.project_fields(f, output_type='response')
        return np.hstack(f)

    def dpred(self, m, f=None):
        """
        Computes predicted data.
        Here we do not store predicted data
        because projection (`d = P(f)`) is cheap.
        """

        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        return f

    def getJ_height(self, m, f=None):
        """
        Compute the sensitivity with respect to source height(s).
        """

        # Null if source height is not parameter of the simulation.
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height

        else:

            if self.verbose:
                print(">> Compute J height ")

            dudh = self.compute_integral(m, output_type="sensitivity_height")
            self._Jmatrix_height = np.hstack(self.project_fields(dudh, output_type="sensitivity_height"))
            if self.survey.nSrc == 1:
                self._Jmatrix_height = np.hstack(dudh).reshape([-1, 1])
            else:
                # COUNT = 0
                # dudh_by_source = []
                # for ii, src in enumerate(self.survey.source_list):
                #     temp = np.array([])
                #     for jj, rx in enumerate(src.receiver_list):
                #         temp = np.r_[temp, dudh[COUNT]]
                #         COUNT += 1

                #     dudh_by_source.append(temp.reshape([-1, 1]))

                # self._Jmatrix_height= block_diag(*dudh_by_source)
                self._Jmatrix_height= block_diag(*dudh_by_source)
            return self._Jmatrix_height


    def getJ_sigma(self, m, f=None):
        """
        Compute the sensitivity with respect to static conductivity.
        """

        # Null if sigma is not parameter of the simulation.
        if self.sigmaMap is None:
            return utils.Zero()

        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        else:

            if self.verbose:
                print(">> Compute J sigma")

            dudsig = self.compute_integral(m, output_type="sensitivity_sigma")
            self._Jmatrix_sigma = np.vstack(self.project_fields(dudsig,output_type="sensitivity_sigma"))
            if self._Jmatrix_sigma.ndim == 1:
                self._Jmatrix_sigma = self._Jmatrix_sigma.reshape([-1, 1])
            return self._Jmatrix_sigma

    def getJ(self, m, f=None):
        """
        Fetch Jacobian.
        """
        return (
            self.getJ_sigma(m, f=f) * self.sigmaDeriv +
            self.getJ_height(m, f=f) * self.hDeriv
        )

    def Jvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jv = np.dot(J_sigma, self.sigmaMap.deriv(m, v))
        if self.hMap is not None:
            Jv += np.dot(J_height, self.hMap.deriv(m, v))
        return Jv

    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jtv = self.sigmaDeriv.T*np.dot(J_sigma.T, v)
        if self.hMap is not None:
            Jtv += self.hDeriv.T*np.dot(J_height.T, v)
        return Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ['_Jmatrix_sigma']
            if self._Jmatrix_height is not None:
                toDelete += ['_Jmatrix_height']
        return toDelete

    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(utils.sdiag(1/delta_d/pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S-thres_hold > 0.
        doi = abs(self.depth[active]).max()
        return doi, active

    def get_threshold(self, uncert):
        _, active = self.depth_of_investigation(uncert)
        JtJdiag = self.get_JtJdiag(uncert)
        delta = JtJdiag[active].min()
        return delta

    def get_JtJdiag(self, uncert):
        J = self.getJ(self.model)
        JtJdiag = (np.power((utils.sdiag(1./uncert)*J), 2)).sum(axis=0)
        return JtJdiag



class EM1DFMSimulation(BaseEM1DSimulation):
    """
    Simulation class for simulating the FEM response over a 1D layered Earth
    for a single sounding.
    """

    def __init__(self, **kwargs):
        BaseEM1DSimulation.__init__(self, **kwargs)


    def project_fields(self, u, output_type='response'):
        """
        Project from the list of Hankel transform evaluations to the data or sensitivities.
        Data can be real or imaginary component of: total field, secondary field or ppm.

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

                if rx.component == 'real':
                    u_temp = np.real(u_temp)
                elif rx.component == 'imag':
                    u_temp = np.imag(u_temp)
                elif rx.component == 'both':
                    u_temp_r = np.real(u_temp)
                    u_temp_i = np.imag(u_temp)
                    if output_type == 'sensitivity_sigma':
                        u_temp = np.vstack((u_temp_r,u_temp_i))
                    else:
                        u_temp = np.r_[u_temp_r,u_temp_i]
                else:
                    raise Exception()

                # Either total or ppm
                if rx.field_type != "secondary":
                    u_primary = src.PrimaryField(rx.locations, rx.use_source_receiver_offset)
                    if rx.field_type == "ppm":
                        k = [comp == rx.orientation for comp in ["x", "y", "z"]]
                        u_temp = 1e6 * u_temp/u_primary[0, k]
                    else:
                        if rx.component == 'both':
                            if output_type == 'sensitivity_sigma':
                                u_temp = np.vstack((u_temp_r+u_primary,u_temp_i))
                            else:
                                u_temp = np.r_[u_temp_r+u_primary, u_temp_i]

                        else:
                            u_temp =+ u_primary

                u[COUNT] = u_temp
                COUNT = COUNT + 1

        return u


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
            if src.wave_type == "general":
                for rx in src.receiver_list:

                    if src.moment_type == "single":
                        time = rx.times
                        pulse_period = src.pulse_period
                        period = src.period
                    # Dual moment
                    else:
                        time = np.unique(np.r_[rx.times, rx.times_dual_moment])
                        pulse_period = np.maximum(
                            src.pulse_period, src.pulse_period_dual_moment
                        )
                        period = np.maximum(src.period, src.period_dual_moment)
                    tmin = time[time>0.].min()
                    if src.n_pulse == 1:
                        tmax = time.max() + pulse_period
                    elif src.n_pulse == 2:
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

                if src.wave_type == "general":
                    _, freq, ft, ftarg = check_time(
                        rx.time_interval, -1, 'dlf',
                        {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0
                    )
                elif src.wave_type == "stepoff":
                    _, freq, ft, ftarg = check_time(
                        rx.times, -1, 'dlf',
                        {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0,
                    )
                else:
                    raise Exception("wave_type must be either general or stepoff")

                rx.frequencies = freq
                rx.ftarg = ftarg

        self.frequencies_are_set = True


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
                if src.use_lowpass_filter:
                    factor = src.lowpass_filter.copy()
                else:
                    factor = np.ones_like(rx.frequencies, dtype=complex)

                # Multiplication factors
                if rx.component in ["b", "h"]:
                    factor *= 1./(2j*np.pi*rx.frequencies)

                if rx.component in ["b", "dbdt"]:
                    factor *= mu_0

                # For stepoff waveform
                if src.wave_type == 'stepoff':

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
                elif src.wave_type == 'general':

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
                            src.time_input_currents,
                            src.input_currents,
                            src.period,
                            n_pulse=src.n_pulse
                        )

                        # Compute response for the dual moment
                        if src.moment_type == "dual":
                            resp_dual_moment = piecewise_pulse_fast(
                                step_func, rx.times_dual_moment,
                                src.time_input_currents_dual_moment,
                                src.input_currents_dual_moment,
                                src.period_dual_moment,
                                n_pulse=src.n_pulse
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
                                src.time_input_currents, src.input_currents,
                                src.period, n_pulse=src.n_pulse
                            )

                            if src.moment_type == "single":
                                resp[:, i] = resp_i
                            else:
                                resp_dual_moment_i = piecewise_pulse_fast(
                                    step_func,
                                    rx.times_dual_moment,
                                    src.time_input_currents_dual_moment,
                                    src.input_currents_dual_moment,
                                    src.period_dual_moment,
                                    n_pulse=src.n_pulse
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


def run_simulation_FD(args):
    """
    This method simulates the EM response or computes the sensitivities for
    a single sounding. The method allows for parallelization of
    the stitched 1D problem.

    :param src: a EM1DFM source object
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
    local_survey = EM1DSurveyFD([src])
    exp_map = maps.ExpMap(nP=n_layer)

    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment

        sim = EM1DFMSimulation(
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

        sim = EM1DFMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=sigma_map, hMap=wires.h, topo=topo,
            eta=eta, tau=tau, c=c, chi=chi, dchi=dchi, tau1=tau1, tau2=tau2,
            hankel_filter='key_101_2009'
        )

        m = np.r_[np.log(sigma), h]
        if output_type == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(m)
            return utils.mkvc(drespdsig * utils.sdiag(sigma))
            # return utils.mkvc(drespdsig)
        elif output_type == 'sensitivity_height':
            drespdh = sim.getJ_height(m)
            return utils.mkvc(drespdh)
        else:
            resp = sim.dpred(m)
            return resp


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


class BaseStitchedEM1DSimulation(BaseSimulation):
    """
    Base class for the stitched 1D simulation. This simulation models the EM
    response for a set of 1D EM soundings.
    """

    _Jmatrix_sigma = None
    _Jmatrix_height = None
    run_simulation = None
    n_cpu = None
    parallel = False
    parallel_jvec_jtvec = False
    verbose = False
    fix_Jmatrix = False
    invert_height = None

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers",
        default=np.array([])
    )

    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    h, hMap, hDeriv = props.Invertible(
        "Receiver Height (m), h > 0",
    )

    eta = props.PhysicalProperty(
        "Electrical chargeability (V/V), 0 <= eta < 1"
    )

    tau = props.PhysicalProperty(
        "Time constant (s)"
    )

    c = props.PhysicalProperty(
        "Frequency Dependency, 0 < c < 1"
    )

    chi = props.PhysicalProperty(
        "Magnetic susceptibility (SI)"
    )

    dchi = props.PhysicalProperty(
        "DC magnetic susceptibility attributed to magnetic viscosity (SI)"
    )

    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants (s)"
    )

    tau2 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants (s)"
    )

    topo = properties.Array("Topography (x, y, z)", dtype=float, shape=('*', 3))

    survey = properties.Instance(
        "a survey object", BaseEM1DSurvey, required=True
    )

    def __init__(self, **kwargs):
        utils.setKwargs(self, **kwargs)

        if PARALLEL:
            if self.parallel:
                print(">> Use multiprocessing for parallelization")
                if self.n_cpu is None:
                    self.n_cpu = multiprocessing.cpu_count()
                print((">> n_cpu: %i") % (self.n_cpu))
            else:
                print(">> Serial version is used")
        else:
            print(">> Serial version is used")

        if self.hMap is None:
            self.invert_height = False
        else:
            self.invert_height = True

    # ------------- For survey ------------- #
    # @property
    # def dz(self):
    #     if self.mesh.dim==2:
    #         return self.mesh.dy
    #     elif self.mesh.dim==3:
    #         return self.mesh.dz

    @property
    def halfspace_switch(self):
        """True = halfspace, False = layered Earth"""
        if (self.thicknesses is None) | (len(self.thicknesses)==0):
            return True
        else:
            return False

    @property
    def n_layer(self):
        if self.thicknesses is None:
            return 1
        else:
            return len(self.thicknesses) + 1

    @property
    def n_sounding(self):
        return len(self.survey.source_list)


    @property
    def data_index(self):
        return self.survey.data_index


    # ------------- For physical properties ------------- #
    @property
    def Sigma(self):
        if getattr(self, '_Sigma', None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    @property
    def Eta(self):
        if getattr(self, '_Eta', None) is None:
            # Ordering: first z then x
            if self.eta is None:
                self._Eta = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Eta = self.eta.reshape((self.n_sounding, self.n_layer))
        return self._Eta

    @property
    def Tau(self):
        if getattr(self, '_Tau', None) is None:
            # Ordering: first z then x
            if self.tau is None:
                self._Tau = 1e-3*np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Tau = self.tau.reshape((self.n_sounding, self.n_layer))
        return self._Tau

    @property
    def C(self):
        if getattr(self, '_C', None) is None:
            # Ordering: first z then x
            if self.c is None:
                self._C = np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._C = self.c.reshape((self.n_sounding, self.n_layer))
        return self._C

    @property
    def Chi(self):
        if getattr(self, '_Chi', None) is None:
            # Ordering: first z then x
            if self.chi is None:
                self._Chi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Chi = self.chi.reshape((self.n_sounding, self.n_layer))
        return self._Chi

    @property
    def dChi(self):
        if getattr(self, '_dChi', None) is None:
            # Ordering: first z then x
            if self.dchi is None:
                self._dChi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._dChi = self.dchi.reshape((self.n_sounding, self.n_layer))
        return self._dChi

    @property
    def Tau1(self):
        if getattr(self, '_Tau1', None) is None:
            # Ordering: first z then x
            if self.tau1 is None:
                self._Tau1 = 1e-10 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Tau1 = self.tau1.reshape((self.n_sounding, self.n_layer))
        return self._Tau1

    @property
    def Tau2(self):
        if getattr(self, '_Tau2', None) is None:
            # Ordering: first z then x
            if self.tau2 is None:
                self._Tau2 = 100. * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Tau2 = self.tau2.reshape((self.n_sounding, self.n_layer))
        return self._Tau2

    @property
    def JtJ_sigma(self):
        return self._JtJ_sigma

    def JtJ_height(self):
        return self._JtJ_height

    @property
    def H(self):
        if self.hMap is None:
            return np.ones(self.n_sounding)
        else:
            return self.h


    # ------------- Etcetra .... ------------- #
    @property
    def IJLayers(self):
        if getattr(self, '_IJLayers', None) is None:
            # Ordering: first z then x
            self._IJLayers = self.set_ij_n_layer()
        return self._IJLayers

    @property
    def IJHeight(self):
        if getattr(self, '_IJHeight', None) is None:
            # Ordering: first z then x
            self._IJHeight = self.set_ij_n_layer(n_layer=1)
        return self._IJHeight

    # ------------- For physics ------------- #

    def input_args(self, i_sounding, output_type='forward'):
        output = (
            self.survey.source_list[i_sounding],
            self.topo[i_sounding, :],
            self.thicknesses,
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.Chi[i_sounding, :],
            self.dChi[i_sounding, :],
            self.Tau1[i_sounding, :],
            self.Tau2[i_sounding, :],
            self.H[i_sounding],
            output_type,
            self.invert_height
        )
        return output

    def fields(self, m):
        if self.verbose:
            print("Compute fields")

        return self.forward(m)

    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        # Set flat topo at zero
        if self.topo is None:
            self.set_null_topography()

        if self.survey.__class__ == EM1DSurveyFD:
            print("Correct Run Simulation")
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        # if (self.parallel) & (__name__=='__main__'):
        if self.parallel:
            if self.verbose:
                print ('parallel')
            pool = Pool(self.n_cpu)
            # This assumes the same # of layers for each of sounding
            result = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type='forward') for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
        else:
            result = [
                run_simulation(self.input_args(i, output_type='forward')) for i in range(self.n_sounding)
            ]
        return np.hstack(result)


    def set_null_topography(self):
        self.topo = np.vstack(
            [np.c_[src.location[0], src.location[1], 0.] for i, src in enumerate(self.survey.source_list)]
        )


    def set_ij_n_layer(self, n_layer=None):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        if n_layer is None:
            m = self.n_layer
        else:
            m = n_layer

        for i in range(self.n_sounding):
            n = self.survey.vnD_by_sounding[i]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order='F') +
                shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def set_ij_height(self):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        m = self.n_layer
        for i in range(self.n_sounding):
            n = self.survey.vnD[i]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order='F') +
                shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)


    def getJ_sigma(self, m):
        """
             Compute d F / d sigma
        """
        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        if self.verbose:
            print(">> Compute J sigma")
        self.model = m

        if self.survey.__class__ == EM1DSurveyFD:
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        # if (self.parallel) & (__name__=='__main__'):
        if self.parallel:

            pool = Pool(self.n_cpu)
            self._Jmatrix_sigma = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type='sensitivity_sigma') for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()

            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_sigma = sp.block_diag(self._Jmatrix_sigma).tocsr()
                self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
                # self._JtJ_sigma_diag =
                self._Jmatrix_sigma = sp.coo_matrix(
                    (self._Jmatrix_sigma, self.IJLayers), dtype=float
                ).tocsr()
        else:
            # _Jmatrix_sigma is block diagnoal matrix (sparse)
            # self._Jmatrix_sigma = sp.block_diag(
            #     [
            #         run_simulation(self.input_args(i, output_type='sensitivity_sigma')) for i in range(self.n_sounding)
            #     ]
            # ).tocsr()
            self._Jmatrix_sigma = [
                    run_simulation(self.input_args(i, output_type='sensitivity_sigma')) for i in range(self.n_sounding)
            ]
            self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()

        return self._Jmatrix_sigma

    def getJ_height(self, m):
        """
             Compute d F / d height
        """
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        if self.verbose:
            print(">> Compute J height")

        self.model = m

        if self.survey.__class__ == EM1DSurveyFD:
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        if (self.parallel) & (__name__=='__main__'):
            pool = Pool(self.n_cpu)
            self._Jmatrix_height = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type="sensitivity_height") for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_height = sp.block_diag(self._Jmatrix_height).tocsr()
                self._Jmatrix_height = np.hstack(self._Jmatrix_height)
                self._Jmatrix_height = sp.coo_matrix(
                    (self._Jmatrix_height, self.IJHeight), dtype=float
                ).tocsr()
        else:
            # self._Jmatrix_height = sp.block_diag(
            #     [
            #         run_simulation(self.input_args(i, output_type='sensitivity_height')) for i in range(self.n_sounding)
            #     ]
            # ).tocsr()
            self._Jmatrix_height = [
                    run_simulation(self.input_args(i, output_type='sensitivity_height')) for i in range(self.n_sounding)
            ]
            self._Jmatrix_height = np.hstack(self._Jmatrix_height)
            self._Jmatrix_height = sp.coo_matrix(
                (self._Jmatrix_height, self.IJHeight), dtype=float
            ).tocsr()

        return self._Jmatrix_height

    def Jvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        # This is deprecated at the moment
        # if self.parallel and self.parallel_jvec_jtvec:
        #     # Extra division of sigma is because:
        #     # J_sigma = dF/dlog(sigma)
        #     # And here sigmaMap also includes ExpMap
        #     v_sigma = utils.sdiag(1./self.sigma) * self.sigmaMap.deriv(m, v)
        #     V_sigma = v_sigma.reshape((self.n_sounding, self.n_layer))

        #     pool = Pool(self.n_cpu)
        #     Jv = np.hstack(
        #         pool.map(
        #             dot,
        #             [(J_sigma[i], V_sigma[i, :]) for i in range(self.n_sounding)]
        #         )
        #     )
        #     if self.hMap is not None:
        #         v_height = self.hMap.deriv(m, v)
        #         V_height = v_height.reshape((self.n_sounding, self.n_layer))
        #         Jv += np.hstack(
        #             pool.map(
        #                 dot,
        #                 [(J_height[i], V_height[i, :]) for i in range(self.n_sounding)]
        #             )
        #         )
        #     pool.close()
        #     pool.join()
        # else:
        Jv = J_sigma*(utils.sdiag(1./self.sigma)*(self.sigmaDeriv * v))
        if self.hMap is not None:
            Jv += J_height*(self.hDeriv * v)
        return Jv

    def Jtvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        # This is deprecated at the moment
        # if self.parallel and self.parallel_jvec_jtvec:
        #     pool = Pool(self.n_cpu)
        #     Jtv = np.hstack(
        #         pool.map(
        #             dot,
        #             [(J_sigma[i].T, v[self.data_index[i]]) for i in range(self.n_sounding)]
        #         )
        #     )
        #     if self.hMap is not None:
        #         Jtv_height = np.hstack(
        #             pool.map(
        #                 dot,
        #                 [(J_sigma[i].T, v[self.data_index[i]]) for i in range(self.n_sounding)]
        #             )
        #         )
        #         # This assumes certain order for model, m = (sigma, height)
        #         Jtv = np.hstack((Jtv, Jtv_height))
        #     pool.close()
        #     pool.join()
        #     return Jtv
        # else:
        # Extra division of sigma is because:
        # J_sigma = dF/dlog(sigma)
        # And here sigmaMap also includes ExpMap
        Jtv = self.sigmaDeriv.T * (utils.sdiag(1./self.sigma) * (J_sigma.T*v))
        if self.hMap is not None:
            Jtv += self.hDeriv.T*(J_height.T*v)
        return Jtv

    def getJtJdiag(self, m, W=None, threshold=1e-8):
        """
        Compute diagonal component of JtJ or
        trace of sensitivity matrix (J)
        """
        J_sigma = self.getJ_sigma(m)
        J_matrix = J_sigma*(utils.sdiag(1./self.sigma)*(self.sigmaDeriv))

        if self.hMap is not None:
            J_height = self.getJ_height(m)
            J_matrix += J_height*self.hDeriv

        if W is None:
            W = utils.speye(J_matrix.shape[0])

        J_matrix = W*J_matrix
        JtJ_diag = (J_matrix.T*J_matrix).diagonal()
        JtJ_diag /= JtJ_diag.max()
        JtJ_diag += threshold
        return JtJ_diag

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ['_Sigma']
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ['_Jmatrix_sigma']
            if self._Jmatrix_height is not None:
                toDelete += ['_Jmatrix_height']
        return toDelete


class StitchedEM1DFMSimulation(BaseStitchedEM1DSimulation):

    def run_simulation(self, args):
        if self.verbose:
            print(">> Frequency-domain")
        return run_simulation_FD(args)

    # @property
    # def frequency(self):
    #     return self.survey.frequency

    # @property
    # def switch_real_imag(self):
    #     return self.survey.switch_real_imag


class StitchedEM1DTMSimulation(BaseStitchedEM1DSimulation):

    # @property
    # def wave_type(self):
    #     return self.survey.wave_type

    # @property
    # def input_currents(self):
    #     return self.survey.input_currents

    # @property
    # def time_input_currents(self):
    #     return self.survey.time_input_currents

    # @property
    # def n_pulse(self):
    #     return self.survey.n_pulse

    # @property
    # def base_frequency(self):
    #     return self.survey.base_frequency

    # @property
    # def time(self):
    #     return self.survey.time

    # @property
    # def use_lowpass_filter(self):
    #     return self.survey.use_lowpass_filter

    # @property
    # def high_cut_frequency(self):
    #     return self.survey.high_cut_frequency

    # @property
    # def moment_type(self):
    #     return self.survey.moment_type

    # @property
    # def time_dual_moment(self):
    #     return self.survey.time_dual_moment

    # @property
    # def time_input_currents_dual_moment(self):
    #     return self.survey.time_input_currents_dual_moment

    # @property
    # def input_currents_dual_moment(self):
    #     return self.survey.input_currents_dual_moment

    # @property
    # def base_frequency_dual_moment(self):
    #     return self.survey.base_frequency_dual_moment

    def run_simulation(self, args):
        if self.verbose:
            print(">> Time-domain")
        return run_simulation_TD(args)

    # def forward(self, m, f=None):
    #     self.model = m

    #     if self.parallel:
    #         pool = Pool(self.n_cpu)
    #         # This assumes the same # of layer for each of soundings
    #         result = pool.map(
    #             run_simulation_TD,
    #             [
    #                 self.input_args(i, output_type=False) for i in range(self.n_sounding)
    #             ]
    #         )
    #         pool.close()
    #         pool.join()
    #     else:
    #         result = [
    #             run_simulation_TD(self.input_args(i, output_type=False)) for i in range(self.n_sounding)
    #         ]
    #     return np.hstack(result)

    # def getJ(self, m):
    #     """
    #          Compute d F / d sigma
    #     """
    #     if self._Jmatrix is not None:
    #         return self._Jmatrix
    #     if self.verbose:
    #         print(">> Compute J")
    #     self.model = m
    #     if self.parallel:
    #         pool = Pool(self.n_cpu)
    #         self._Jmatrix = pool.map(
    #             run_simulation_TD,
    #             [
    #                 self.input_args(i, output_type=True) for i in range(self.n_sounding)
    #             ]
    #         )
    #         pool.close()
    #         pool.join()
    #         if self.parallel_jvec_jtvec is False:
    #             self._Jmatrix = sp.block_diag(self._Jmatrix).tocsr()
    #     else:
    #         # _Jmatrix is block diagnoal matrix (sparse)
    #         self._Jmatrix = sp.block_diag(
    #             [
    #                 run_simulation_TD(self.input_args(i, output_type=True)) for i in range(self.n_sounding)
    #             ]
    #         ).tocsr()
    #     return self._Jmatrix








