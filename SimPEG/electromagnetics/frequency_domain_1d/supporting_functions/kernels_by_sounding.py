import numpy as np
from scipy.constants import mu_0
from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient
import SimPEG.electromagnetics.frequency_domain as fdem
import SimPEG.electromagnetics.frequency_domain_1d as fdem_1d
import SimPEG.electromagnetics.time_domain_1d as tdem_1d


from empymod.transform import dlf

def magnetic_dipole_response_by_sounding(
    simulation, lamda, f, n_layer, sig, chi, h, z, 
    source_list, data_or_sensitivity, radial_distance,
    output_type='response'
):

    """
    Kernel for vertical (Hz) and radial (Hrho) magnetic component due to
    vertical magnetic diopole (VMD) source in (kx,ky) domain.

    For vertical magnetic dipole:

    .. math::

        H_z = \\frac{m}{4\\pi}
        \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
        \\lambda^2 J_0(\\lambda r) d \\lambda

    .. math::

        H_{\\rho} = - \\frac{m}{4\\pi}
        \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
        \\lambda^2 J_1(\\lambda r) d \\lambda

    For horizontal magnetic dipole:

    .. math::

        H_x = \\frac{m}{4\\pi} \\Bigg \\frac{1}{\\rho} -\\frac{2x^2}{\\rho^3} \\Bigg )
        \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
        \\lambda J_1(\\lambda r) d \\lambda
        + \\frac{m}{4\\pi} \\frac{x^2}{\\rho^2}
        \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
        \\lambda^2 J_0(\\lambda r) d \\lambda

    .. math::

        H_y = - \\frac{m}{4\\pi} \\frac{2xy}{\\rho^3}
        \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
        \\lambda J_1(\\lambda r) d \\lambda
        + \\frac{m}{4\\pi} \\frac{xy}{\\rho^2}
        \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
        \\lambda^2 J_0(\\lambda r) d \\lambda

    .. math::

        H_z = \\frac{m}{4\\pi} \\frac{x}{\\rho}
        \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
        \\lambda^2 J_1(\\lambda r) d \\lambda

    """

    # coefficient_wavenumber = 1/(4*np.pi)*lamda**2
   
    n_frequency = len(f)
    n_filter = simulation.n_filter
    thicknesses = simulation.thicknesses
    mu = (chi+1)*mu_0
    
    # COMPUTE TE-MODE REFLECTION COEFFICIENT
    if output_type == 'sensitivity_sigma':
        drTE = np.zeros(
            [n_layer, n_frequency, n_filter],
            dtype=np.complex128, order='F'
        )

        drTE, _, _ = rTE_gradient(
            f[:,0], lamda[:,:], sig, mu, thicknesses
            )

        temp = drTE * np.exp(-lamda*(z+h))
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        depth = simulation.depth
        rTE = rTE_forward(
            f[:,0], lamda[:,:], sig, mu, thicknesses
        )

        temp = rTE * np.exp(-lamda*(z+h))
        if output_type == 'sensitivity_height':
            temp *= -2*lamda
    
    integral_output_list = []
    if isinstance(simulation, fdem_1d.EM1DFMSimulation):
        i_count = 0
        for src in source_list:
            for rx in src.receiver_list:            
                # COMPUTE KERNEL FUNCTIONS FOR HANKEL TRANSFORM
                # Assume rx has only a single location
                C = src.moment/(4*np.pi)

                if rx.use_source_receiver_offset:
                    v_dist = rx.locations.ravel()
                else:
                    v_dist = rx.locations.ravel() - src.location
                
                if output_type == "sensitivity_sigma":
                    temp_slice = temp[:,i_count,:].reshape((n_layer, 1, n_filter), order='F')  
                else:
                    temp_slice = temp[i_count,:]

                if np.all(src.orientation==[0, 0, 1]):
                    if rx.orientation == "z":
                        kernels = [C * lamda[i_count,:]**2 * temp_slice, None, None]
                    elif rx.orientation == "x":
                        C *= -v_dist[0]/np.sqrt(np.sum(v_dist[0:-1]**2))
                        kernels = [None, C * lamda[i_count,:]**2 * temp_slice, None]
                    elif rx.orientation == "y":
                        C *= -v_dist[1]/np.sqrt(np.sum(v_dist[0:-1]**2))
                        kernels = [None, C * lamda[i_count,:]**2 * temp_slice, None]
                elif np.all(src.orientation==[1, 0, 0]):
                    rho = np.sqrt(np.sum(v_dist[0:-1]**2))
                    if rx.orientation == "z":
                        C *= v_dist[0]/rho
                        kernels = [None, C * lamda[i_count,:]**2 * temp_slice, None]
                    elif rx.orientation == "x":
                        C0 = C * v_dist[0]**2/rho**2
                        C1 = C * (1/rho - 2*v_dist[0]**2/rho**3)
                        kernels = [C0 * lamda[i_count,:]**2 * temp_slice, C1 * lamda[i_count,:] *temp_slice, None]
                    elif rx.orientation == "y":
                        C0 = C * v_dist[0]*v_dist[1]/rho**2
                        C1 = C * -2*v_dist[0]*v_dist[1]/rho**3
                        kernels = [C0 * lamda[i_count,:]**2 * temp_slice, C1 * lamda[i_count,:] *temp_slice, None]
                elif np.all(src.orientation==[0, 1, 0]):
                    rho = np.sqrt(np.sum(v_dist[0:-1]**2))
                    if rx.orientation == "z":
                        C *= v_dist[1]/rho
                        kernels = [None, C * lamda[i_count,:]**2 * temp_slice, None]
                    elif rx.orientation == "x":
                        C0 = C * -v_dist[0]*v_dist[1]/rho**2
                        C1 = C * 2*v_dist[0]*v_dist[1]/rho**3
                        kernels = [C0 * lamda[i_count,:]**2 * temp_slice, C1 * lamda[i_count,:] *temp_slice, None]
                    elif rx.orientation == "y":
                        C0 = C * v_dist[1]**2/rho**2
                        C1 = C * (1/rho - 2*v_dist[1]**2/rho**3)
                        kernels = [C0 * lamda[i_count,:]**2 * temp_slice, C1 * lamda[i_count,:] *temp_slice, None]
                
                kernels = tuple(kernels)

                if output_type=="sensitivity_sigma":
                    offset = np.tile(radial_distance[i_count], (n_layer, 1))
                else:
                    offset = radial_distance[i_count]
                # Evaluate Hankel transform using digital linear filter from empymod
                integral_output = src.moment * dlf(
                    kernels, lamda[i_count,:], offset, simulation.fhtfilt, simulation.hankel_pts_per_dec, ang_fact=None, ab=33
                )
                # Project fields
                if output_type == "sensitivity_sigma":
                    integral_output = integral_output.T

                data_or_sensitivity[src, rx] = simulation.project_fields_src_rx(
                    integral_output, src.i_sounding, src, rx, 
                    output_type=output_type
                )
                
                i_count += 1    

    elif isinstance(simulation, tdem_1d.EM1DTMSimulation): 
        for src in source_list:
            for rx in src.receiver_list:            
                # COMPUTE KERNEL FUNCTIONS FOR HANKEL TRANSFORM
                # Assume rx has only a single location
                C = src.moment/(4*np.pi)

                if rx.use_source_receiver_offset:
                    v_dist = rx.locations.ravel()
                else:
                    v_dist = rx.locations.ravel() - src.location

                if np.all(src.orientation==[0, 0, 1]):
                    if rx.orientation == "z":
                        kernels = [C * lamda[:,:]**2 * temp, None, None]
                    elif rx.orientation == "x":
                        C *= -v_dist[0]/np.sqrt(np.sum(v_dist[0:-1]**2))
                        kernels = [None, C * lamda[:,:]**2 * temp, None]
                    elif rx.orientation == "y":
                        C *= -v_dist[1]/np.sqrt(np.sum(v_dist[0:-1]**2))
                        kernels = [None, C * lamda[:,:]**2 * temp, None]
                elif np.all(src.orientation==[1, 0, 0]):
                    rho = np.sqrt(np.sum(v_dist[0:-1]**2))
                    if rx.orientation == "z":
                        C *= v_dist[0]/rho
                        kernels = [None, C * lamda[:,:]**2 * temp, None]
                    elif rx.orientation == "x":
                        C0 = C * v_dist[0]**2/rho**2
                        C1 = C * (1/rho - 2*v_dist[0]**2/rho**3)
                        kernels = [C0 * lamda[:,:]**2 * temp, C1 * lamda[:,:] * temp, None]
                    elif rx.orientation == "y":
                        C0 = C * v_dist[0]*v_dist[1]/rho**2
                        C1 = C * -2*v_dist[0]*v_dist[1]/rho**3
                        kernels = [C0 * lamda[:,:]**2 * temp, C1 * lamda[:,:] * temp, None]
                elif np.all(src.orientation==[0, 1, 0]):
                    rho = np.sqrt(np.sum(v_dist[0:-1]**2))
                    if rx.orientation == "z":
                        C *= v_dist[1]/rho
                        kernels = [None, C * lamda[:,:]**2 * temp, None]
                    elif rx.orientation == "x":
                        C0 = C * -v_dist[0]*v_dist[1]/rho**2
                        C1 = C * 2*v_dist[0]*v_dist[1]/rho**3
                        kernels = [C0 * lamda[:,:]**2 * temp, C1 * lamda[:,:] * temp, None]
                    elif rx.orientation == "y":
                        C0 = C * v_dist[1]**2/rho**2
                        C1 = C * (1/rho - 2*v_dist[1]**2/rho**3)
                        kernels = [C0 * lamda[:,:]**2 * temp, C1 * lamda[:,:] * temp, None]
                
                kernels = tuple(kernels)

                if output_type=="sensitivity_sigma":
                    offset = np.tile(radial_distance, (n_layer, 1))
                else:
                    offset = radial_distance
                # Evaluate Hankel transform using digital linear filter from empymod
                integral_output = src.moment * dlf(
                    kernels, lamda[:,:], offset, simulation.fhtfilt, simulation.hankel_pts_per_dec, ang_fact=None, ab=33
                )
                # Project fields
                if output_type == "sensitivity_sigma":
                    integral_output = integral_output.T

                data_or_sensitivity[src, rx] = simulation.project_fields_src_rx(
                    integral_output, src.i_sounding, src, rx, 
                    output_type=output_type
                )

    return data_or_sensitivity

def horizontal_loop_response_by_sounding(
    simulation, lamda, f, n_layer, sig, chi, a, h, z, 
    source_list, data_or_sensitivity,
    output_type='response'    
):

    """

    Kernel for vertical (Hz) and radial (Hrho) magnetic component due to
    horizontal cirular loop source in (kx,ky) domain.

    For the vertical component:

    .. math::
        H_z = \\frac{Ia}{2} \\int_0^{\\infty}
        \\r_{TE}e^{u_0|z-h|}] \\frac{\\lambda^2}{u_0}
        J_1(\\lambda a) J_0(\\lambda r) d \\lambda

    For the radial component:

    .. math::
        H_{\\rho} = - \\frac{Ia}{2} \\int_0^{\\infty}
        \\r_{TE}e^{u_0|z-h|}] \\lambda
        J_1(\\lambda a) J_1(\\lambda r) d \\lambda


    """
    n_frequency = len(f)
    n_filter = simulation.n_filter

    w = 2*np.pi*f
    u0 = lamda
    radius = np.empty([n_frequency, n_filter], order='F')
    radius[:, :] = np.tile(a.reshape([-1, 1]), (1, n_filter))

    coefficient_wavenumber = radius*0.5*lamda**2/u0
    thicknesses = simulation.thicknesses
    mu = (chi+1)*mu_0
    a_vec = np.array([a])        
    
    if output_type == 'sensitivity_sigma':
        a_vec = np.tile(a_vec, (n_layer, 1))
        drTE = np.empty(
            [n_layer, n_frequency, n_filter],
            dtype=np.complex128, order='F'
        )
        
        drTE, _, _ = rTE_gradient(
            f[:,0], lamda[:,:], sig, mu, thicknesses
        )
        kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )

        rTE = rTE_forward(
            f[:,0], lamda[:,:], sig, mu, thicknesses
        )

        kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber

        if output_type == 'sensitivity_height':
            kernel *= -2*u0
    
    integral_output_list = []
    if isinstance(simulation, fdem_1d.EM1DFMSimulation):
        i_count = 0
        for src in source_list:
            for rx in src.receiver_list:     
                if output_type == 'sensitivity_sigma':
                    kernels = (None, kernel[:,i_count,:].reshape((n_layer, 1, n_filter), order='F')  , None) 
                else:
                    kernels = (None, kernel[i_count,:], None) 

                # Evaluate Hankel transform using digital linear filter from empymod
                integral_output = src.current * dlf(
                    kernels, lamda[i_count, :], a_vec, simulation.fhtfilt, simulation.hankel_pts_per_dec, ang_fact=None, ab=33
                )
                
                # Project fields
                if output_type == "sensitivity_sigma":
                    integral_output = integral_output.T

                data_or_sensitivity[src, rx] = simulation.project_fields_src_rx(
                    integral_output, src.i_sounding, src, rx, 
                    output_type=output_type
                )
                
                i_count += 1    
    
    elif isinstance(simulation, tdem_1d.EM1DTMSimulation):
        for src in source_list:
            for rx in src.receiver_list:     
                kernels = (None, kernel, None) 

                # Evaluate Hankel transform using digital linear filter from empymod
                integral_output = src.current * dlf(
                    kernels, lamda, a_vec, simulation.fhtfilt, simulation.hankel_pts_per_dec, ang_fact=None, ab=33
                )
                
                # Project fields
                if output_type == "sensitivity_sigma":
                    integral_output = integral_output.T
                data_or_sensitivity[src, rx] = simulation.project_fields_src_rx(
                    integral_output, src.i_sounding, src, rx, 
                    output_type=output_type
                )
    else:
        raise Exception()

    return data_or_sensitivity
