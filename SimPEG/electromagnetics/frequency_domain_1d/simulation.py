from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation
import numpy as np
from .sources import *
from .survey import EM1DSurveyFD
from .supporting_functions.kernels import *

from empymod.utils import check_time
from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel


#######################################################################
#               SIMULATION FOR A SINGLE SOUNDING
#######################################################################

class EM1DFMSimulation(BaseEM1DSimulation):
    """
    Simulation class for simulating the FEM response over a 1D layered Earth
    for a single sounding.
    """

    def __init__(self, **kwargs):
        BaseEM1DSimulation.__init__(self, **kwargs)


    def compute_integral(self, m, output_type='response'):
        """
        This method evaluates the Hankel transform for each source and
        receiver and outputs it as a list. Used for computing response
        or sensitivities.
        """

        self.model = m
        n_layer = self.n_layer
        n_filter = self.n_filter

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




