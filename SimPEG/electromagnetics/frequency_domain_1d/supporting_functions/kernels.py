import numpy as np
from scipy.constants import mu_0
from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient

def magnetic_dipole_kernel(
    simulation, lamda, f, n_layer, sig, chi, h, z, r,
    src, rx, output_type='response'
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
    C = src.moment/(4*np.pi)

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
            f, lamda[:,:], sig, mu, thicknesses
            )

        temp = drTE * np.exp(-lamda*(z+h))
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        depth = simulation.depth
        rTE = rTE_forward(
            f, lamda[:,:], sig, mu, thicknesses
        )

        temp = rTE * np.exp(-lamda*(z+h))
        if output_type == 'sensitivity_height':
            temp *= -2*lamda

    # COMPUTE KERNEL FUNCTIONS FOR HANKEL TRANSFORM
    if rx.use_source_receiver_offset:
        v_dist = rx.locations.ravel()
    else:
        v_dist = rx.locations.ravel() - src.location

    if np.all(src.orientation==[0, 0, 1]):
        if rx.orientation == "z":
            kernels = [C * lamda**2 * temp, None, None]
        elif rx.orientation == "x":
            C *= -v_dist[0]/np.sqrt(np.sum(v_dist[0:-1]**2))
            kernels = [None, C * lamda**2 * temp, None]
        elif rx.orientation == "y":
            C *= -v_dist[1]/np.sqrt(np.sum(v_dist[0:-1]**2))
            kernels = [None, C * lamda**2 * temp, None]
    elif np.all(src.orientation==[1, 0, 0]):
        rho = np.sqrt(np.sum(v_dist[0:-1]**2))
        if rx.orientation == "z":
            C *= v_dist[0]/rho
            kernels = [None, C * lamda**2 * temp, None]
        elif rx.orientation == "x":
            C0 = C * v_dist[0]**2/rho**2
            C1 = C * (1/rho - 2*v_dist[0]**2/rho**3)
            kernels = [C0 * lamda**2 * temp, C1 * lamda *temp, None]
        elif rx.orientation == "y":
            C0 = C * v_dist[0]*v_dist[1]/rho**2
            C1 = C * -2*v_dist[0]*v_dist[1]/rho**3
            kernels = [C0 * lamda**2 * temp, C1 * lamda *temp, None]
    elif np.all(src.orientation==[0, 1, 0]):
        rho = np.sqrt(np.sum(v_dist[0:-1]**2))
        if rx.orientation == "z":
            C *= v_dist[1]/rho
            kernels = [None, C * lamda**2 * temp, None]
        elif rx.orientation == "x":
            C0 = C * -v_dist[0]*v_dist[1]/rho**2
            C1 = C * 2*v_dist[0]*v_dist[1]/rho**3
            kernels = [C0 * lamda**2 * temp, C1 * lamda *temp, None]
        elif rx.orientation == "y":
            C0 = C * v_dist[1]**2/rho**2
            C1 = C * (1/rho - 2*v_dist[1]**2/rho**3)
            kernels = [C0 * lamda**2 * temp, C1 * lamda *temp, None]


    return kernels


# def magnetic_dipole_fourier(
#     simulation, lamda, f, n_layer, sig, chi, I, h, z, r,
#     src, rx, output_type='response'
# ):

#     """
#     Kernel for vertical (Hz) and radial (Hrho) magnetic component due to
#     vertical magnetic diopole (VMD) source in (kx,ky) domain.

#     For vertical magnetic dipole:

#     .. math::

#         H_z = \\frac{m}{4\\pi}
#         \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
#         \\lambda^2 J_0(\\lambda r) d \\lambda

#     .. math::

#         H_{\\rho} = - \\frac{m}{4\\pi}
#         \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
#         \\lambda^2 J_1(\\lambda r) d \\lambda

#     For horizontal magnetic dipole:

#     .. math::

#         H_x = \\frac{m}{4\\pi} \\Bigg \\frac{1}{\\rho} -\\frac{2x^2}{\\rho^3} \\Bigg )
#         \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
#         \\lambda J_1(\\lambda r) d \\lambda
#         + \\frac{m}{4\\pi} \\frac{x^2}{\\rho^2}
#         \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
#         \\lambda^2 J_0(\\lambda r) d \\lambda

#     .. math::

#         H_y = - \\frac{m}{4\\pi} \\frac{2xy}{\\rho^3}
#         \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
#         \\lambda J_1(\\lambda r) d \\lambda
#         + \\frac{m}{4\\pi} \\frac{xy}{\\rho^2}
#         \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
#         \\lambda^2 J_0(\\lambda r) d \\lambda

#     .. math::

#         H_z = \\frac{m}{4\\pi} \\frac{x}{\\rho}
#         \\int_0^{\\infty} \\r_{TE} e^{u_0|z-h|}
#         \\lambda^2 J_1(\\lambda r) d \\lambda

#     """

#     # coefficient_wavenumber = 1/(4*np.pi)*lamda**2
#     C = I/(4*np.pi)

#     n_frequency = len(f)
#     n_filter = simulation.n_filter

#     # COMPUTE TE-MODE REFLECTION COEFFICIENT
#     if output_type == 'sensitivity_sigma':
#         drTE = np.zeros(
#             [n_layer, n_frequency, n_filter],
#             dtype=np.complex128, order='F'
#         )
#         if rte_fortran is None:
#             thick = simulation.thicknesses
#             drTE = rTEfunjac(
#                 n_layer, f, lamda[:,:], sig, chi, thick, simulation.halfspace_switch
#             )
#         else:
#             depth = simulation.depth
#             rte_fortran.rte_sensitivity(
#                 f, lamda[:,:], sig, chi, depth, simulation.halfspace_switch, drTE,
#                 n_layer, n_frequency, n_filter
#                 )

#         temp = drTE * np.exp(-lamda*(z+h))
#     else:
#         rTE = np.empty(
#             [n_frequency, n_filter], dtype=np.complex128, order='F'
#         )
#         if rte_fortran is None:
#             thick = simulation.thicknesses
#             rTE = rTEfunfwd(
#                 n_layer, f, lamda[:,:], sig, chi, thick, simulation.halfspace_switch
#             )
#         else:
#             depth = simulation.depth
#             rte_fortran.rte_forward(
#                 f, lamda[:,:], sig, chi, depth, simulation.halfspace_switch,
#                 rTE, n_layer, n_frequency, n_filter
#             )

#         if output_type == 'sensitivity_height':
#             rTE *= -2*lamda

#     # COMPUTE KERNEL FUNCTIONS FOR FOURIER TRANSFORM
#     return C * lamda**2 * rTE



# TODO: make this to take a vector rather than a single frequency
def horizontal_loop_kernel(
    simulation, lamda, f, n_layer, sig, chi, a, h, z, r,
    src, rx, output_type='response'
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

    coefficient_wavenumber = src.current*radius*0.5*lamda**2/u0
    thicknesses = simulation.thicknesses
    mu = (chi+1)*mu_0
    
    if output_type == 'sensitivity_sigma':
        drTE = np.empty(
            [n_layer, n_frequency, n_filter],
            dtype=np.complex128, order='F'
        )
        
        drTE, _, _ = rTE_gradient(
            f, lamda[0,:], sig, mu, thicknesses
        )
        kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )

        rTE = rTE_forward(
            f, lamda[0,:], sig, mu, thicknesses
        )

        kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber

        if output_type == 'sensitivity_height':
            kernel *= -2*u0

    return kernel

def hz_kernel_horizontal_electric_dipole(
    simulation, lamda, f, n_layer, sig, chi, h, z,
    flag, output_type='response'
):

    """
        Kernel for vertical magnetic field (Hz) due to
        horizontal electric diopole (HED) source in (kx,ky) domain

    """
    n_frequency = len(f)
    n_filter = simulation.n_filter

    u0 = lamda
    coefficient_wavenumber = 1/(4*np.pi)*lamda**2/u0
    thicknesses = simulation.thicknesses
    mu = (chi+1)*mu_0

    if output_type == 'sensitivity_sigma':
        drTE = np.zeros(
            [n_layer, n_frequency, n_filter], dtype=np.complex128,
            order='F'
        )
        
        drTE, _, _ = rTE_gradient(
            f, lamda[0,:], sig, mu, thicknesses
        )

        kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        rTE = rTE_forward(
            f, lamda[0,:], sig, mu, thicknesses
        )

        kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        if output_type == 'sensitivity_height':
            kernel *= -2*u0

    return kernel

