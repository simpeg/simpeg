from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation, Sensitivity
from ..frequency_domain.sources import MagDipole, CircularLoop
from ..frequency_domain.receivers import PointMagneticFieldSecondary
from ..frequency_domain.survey import Survey
import numpy as np
import properties
# from .sources import *
# from .survey import EM1DSurveyFD
from .supporting_functions.kernels import *
from .supporting_functions.kernels_by_sounding import magnetic_dipole_response_by_sounding, horizontal_loop_response_by_sounding
from SimPEG import Data
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
    
    survey = properties.Instance(
        "a survey object", Survey, required=True
    )

    def __init__(self, **kwargs):
        BaseEM1DSimulation.__init__(self, **kwargs)
        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                if rx.locations.shape[0] > 1:
                    raise Exception("A single location for a receiver object is assumed for the 1D EM code")

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
        # Issue: this only works for a single source.
        if self.hMap is not None:
            h_vector = self.h * np.ones(len(self.survey.source_list))
        else:
            if self.topo is None:
                h_vector = np.array([src.location[2] for src in self.survey.source_list])
            else:
                h_vector = np.array(
                    [src.location[2]-self.topo[-1] for src in self.survey.source_list]
                )


        integral_output_list = []

        # Issue: it would be better if have an internal sorting 
        # to combine all sources having the same location, 
        # but different frequencies as well as receiver locations 
        # having the same height. They do not needed to be in the for loop, 
        # we can compute all of them once. Which could save some time. 

        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                frequency = np.array([src.frequency])
                f = np.ones((1, n_filter), order='F') * frequency

                # Create globally, not for each receiver in the future
                sig = self.compute_sigma_matrix(frequency)
                chi = self.compute_chi_matrix(frequency)

                # Compute receiver height
                h = h_vector[i_src]
                if rx.use_source_receiver_offset:
                    z = h + rx.locations[0, 2]
                else:
                    z = h + rx.locations[0, 2] - src.location[2]
                
                # Hankel transform for horizontal loop source
                # Issue: isinstance(src, CircularLoop) is true even when 
                # src is MagDipole...
                if isinstance(src, CircularLoop):

                    # radial distance (r) and loop radius (a)
                    if rx.use_source_receiver_offset:
                        r = rx.locations[0, 0:2]
                    else:
                        r = rx.locations[0, 0:2] - src.location[0:2]
        
                    r_vec = np.sqrt(np.sum(r**2)) * np.ones(1)
                    a_vec = src.radius * np.ones(1)

                    # Use function from empymod to define Hankel coefficients.
                    # Size of lambd is (1 x n_filter)
                    lambd = np.empty([1, n_filter], order='F')
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
                    integral_output = src.current * dlf(
                        PJ, lambd, a_vec, self.fhtfilt, self.hankel_pts_per_dec, ang_fact=None, ab=33
                    )

                # Hankel transform for x, y or z magnetic dipole source
                elif isinstance(src, MagDipole):
                    # Radial distance
                    if rx.use_source_receiver_offset:
                        r = rx.locations[0, 0:2]
                    else:
                        r = rx.locations[0, 0:2] - src.location[0:2]

                    r = np.sqrt(np.sum(r**2))
                    r_vec = r * np.ones(1)

                    # Use function from empymod to define Hankel coefficients.
                    # Size of lambd is (1 x n_filter)

                    lambd = np.empty([1, n_filter], order='F')
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
                    integral_output = src.moment * dlf(
                        PJ, lambd, r_vec, self.fhtfilt, self.hankel_pts_per_dec, ang_fact=None, ab=33
                    )


                else:
                    raise Exception()
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
        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):

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

                if isinstance(rx, PointMagneticFieldSecondary):

                    if rx.data_type == "ppm":
                        u_primary = src.hPrimary1D(rx.locations, rx.use_source_receiver_offset)
                        k = [comp == rx.orientation for comp in ["x", "y", "z"]]
                        u_temp = 1e6 * u_temp/u_primary[0, k]
    
                elif isinstance(rx, PointMagneticField):
                    u_primary = src.hPrimary1D(rx.locations, rx.use_source_receiver_offset)
                    if rx.component == 'both':
                        if output_type == 'sensitivity_sigma':
                            u_temp = np.vstack((u_temp_r+u_primary,u_temp_i))
                        else:
                            u_temp = np.r_[u_temp_r+u_primary, u_temp_i]

                    else:
                        u_temp =+ u_primary
                else:
                    raise Exception()

                u[COUNT] = u_temp
                COUNT = COUNT + 1

        return u

    def compute_integral_by_sounding(self, m, output_type='response'):
        """
        This method evaluates the Hankel transform for each source and
        receiver and outputs it as a list. Used for computing response
        or sensitivities.
        """

        self.model = m
        n_layer = self.n_layer
        n_filter = self.n_filter
        
        # Define source height above topography by mapping or from sources and receivers.
        # Issue: this only works for a single source.

        integral_output_list = []

        source_location_by_sounding_dict = self.survey.source_location_by_sounding_dict
        if output_type == 'sensitivity_sigma':
            data_or_sensitivity = Sensitivity(self.survey, M=n_layer)
        else: 
            data_or_sensitivity = Data(self.survey)
            
        
        for i_sounding in source_location_by_sounding_dict:        
            src_locations = np.vstack(source_location_by_sounding_dict[i_sounding])
            rx_locations = self.survey.receiver_location_by_sounding_dict[i_sounding]
            rx_use_offset = self.survey.receiver_use_offset_by_sounding_dict[i_sounding]
            
            n_filter = self.n_filter
            n_frequency_rx = self.survey.vnrx_by_sounding_dict[i_sounding]
            f = np.empty([n_frequency_rx, n_filter], order='F')
            frequencies = self.survey.frequency_by_sounding_dict[i_sounding]
            f = np.tile(
                frequencies.reshape([-1, 1]), (1, n_filter)
            )

            # Create globally, not for each receiver in the future
            sig = self.compute_sigma_matrix(frequencies)
            chi = self.compute_chi_matrix(frequencies)

            # Compute receiver height
            # Assume all sources in i-th sounding have the same src.location
            if self.hMap is not None:
                h = self.h
            elif self.topo is None:
                h = src_locations[0, 2]
            else:
                h = src_locations[0, 2]-self.topo[-1]
            
            # Assume all receivers in i-th sounding have the same receiver height
            if rx_use_offset[0]:
                z = h + rx_locations[0, 2]
            else:
                z = h + rx_locations[0, 2] - src_locations[0, 2]            
           
            # Assume all receivers in i-th sounding have the same rx.use_source_receiver_offset.
            # But, their Radial distance can be different.
            if rx_use_offset[0]:
                r = rx_locations[:, 0:2]
            else:
                r = rx_locations[:, 0:2] - src_locations[0,0:2]
            r = np.sqrt(np.sum(r**2, axis=1))
           
            # Assume all sources in i-th sounding have the same type
            source_list = self.survey.get_sources_by_sounding_number(i_sounding)
            src = source_list[0]
            
            if isinstance(src, CircularLoop):
                # Assume all sources in i-th sounding have the same radius
                a = np.array([src.radius])
                # Use function from empymod to define Hankel coefficients.
                # Size of lambd is (1 x n_filter)
                lambd = np.empty([n_frequency_rx, n_filter], order='F')
                lambd[:, :], _ = get_dlf_points(
                    self.fhtfilt, a, self.hankel_pts_per_dec
                )      
                
                data_or_sensitivity = horizontal_loop_response_by_sounding(
                    self, lambd, f, n_layer, sig, chi, a, h, z, 
                    source_list, data_or_sensitivity,
                    output_type=output_type            
                )
            
            elif isinstance(src, MagDipole):                

                # Use function from empymod to define Hankel coefficients.
                # Size of lambd is (1 x n_filter)
                lambd = np.empty([n_frequency_rx, n_filter], order='F')
                lambd[:, :], _ = get_dlf_points(
                    self.fhtfilt, r, self.hankel_pts_per_dec
                )      
                
                data_or_sensitivity = magnetic_dipole_response_by_sounding(
                    self, lambd, f, n_layer, sig, chi, h, z, 
                    source_list, data_or_sensitivity, r,
                    output_type=output_type            
                )
            return data_or_sensitivity

    def project_fields_src_rx(self, u, i_sounding, src, rx, output_type='response'):
        """
        Project from the list of Hankel transform evaluations to the data or sensitivities.
        Data can be real or imaginary component of: total field, secondary field or ppm.

        :param list u: list containing Hankel transform outputs for each unique
        source-receiver pair.
        :rtype: list: list containing predicted data for each unique
        source-receiver pair.
        :return: predicted data or sensitivities by source-receiver
        """

        if rx.component == 'real':
            data = np.atleast_1d(np.real(u))
        elif rx.component == 'imag':
            data = np.atleast_1d(np.imag(u))
        elif rx.component == 'both':
            data_r = np.real(u)
            data_i = np.imag(u)
            if output_type == 'sensitivity_sigma':
                data = np.vstack((data_r,data_i))
            else:
                data = np.r_[data_r,data_i]
        else:
            raise Exception()

        if isinstance(rx, PointMagneticFieldSecondary):

            if rx.data_type == "ppm":
                data_primary = src.hPrimary1D(rx.locations, rx.use_source_receiver_offset)
                k = [comp == rx.orientation for comp in ["x", "y", "z"]]
                data = 1e6 * data/data_primary[0, k]

        elif isinstance(rx, PointMagneticField):
            data_primary = src.hPrimary1D(rx.locations, rx.use_source_receiver_offset)
            if rx.component == 'both':
                if output_type == 'sensitivity_sigma':
                    data = np.vstack((data_r+data_primary,data_i))
                else:
                    data = np.r_[data_r+data_primary, data_i]
            else:
                data =+ data_primary
        else:
            raise Exception()

        return data

    def fields(self, m):
        # f = self.compute_integral(m, output_type='response')
        # f = self.project_fields(f, output_type='response')
        data = self.compute_integral_by_sounding(m, output_type='response')
        return data.dobs

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

            dudh = self.compute_integral_by_sounding(m, output_type="sensitivity_height")
            self._Jmatrix_height = dudh.dobs.reshape([-1, 1])
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
            
            dudsig = self.compute_integral_by_sounding(m, output_type="sensitivity_sigma")
            self._Jmatrix_sigma = dudsig.sensitivity
            if self._Jmatrix_sigma.ndim == 1:
                self._Jmatrix_sigma = self._Jmatrix_sigma.reshape([-1, 1])
            return self._Jmatrix_sigma        

#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################

class StitchedEM1DFMSimulation(BaseStitchedEM1DSimulation):

    def run_simulation(self, args):
        if self.verbose:
            print(">> Frequency-domain")
        return self._run_simulation(args)

    def dot(self, args):
        return np.dot(args[0], args[1])

    def _run_simulation(self, args):
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

        src_list, topo, thicknesses, sigma, eta, tau, c, chi, dchi, tau1, tau2, h, output_type, invert_height = args

        n_layer = len(thicknesses) + 1
        local_survey = Survey(src_list)
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
