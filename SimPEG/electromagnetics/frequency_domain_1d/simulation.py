from SimPEG import maps, utils
from SimPEG.electromagnetics.base import BaseEM1DSimulation
import numpy as np
from .sources import *
from .survey import EM1DSurveyFD
from .supporting_functions.kernels import *

from empymod.utils import check_time
from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel

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




