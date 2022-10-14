import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from geoana.em.fdem.base import skin_depth
from geoana.em.tdem import diffusion_distance

from SimPEG import utils


def plot_layer(sig, mesh, xscale="log", ax=None, showlayers=False, xlim=None, **kwargs):
    """
    Plot Conductivity model for the layered earth model
    """

    z_grid = mesh.vectorNx
    n_sig = sig.size
    sigma = np.repeat(sig, 2)
    z = []
    for i in range(n_sig):
        z.append(np.r_[z_grid[i], z_grid[i + 1]])
    z = np.hstack(z)
    if xlim == None:
        sig_min = sig[~np.isnan(sig)].min() * 0.5
        sig_max = sig[~np.isnan(sig)].max() * 2
    else:
        sig_min, sig_max = xlim

    if xscale == "linear" and sig.min() == 0.0:
        if xlim == None:
            sig_min = -sig[~np.isnan(sig)].max() * 0.5
            sig_max = sig[~np.isnan(sig)].max() * 2

    if ax == None:
        plt.xscale(xscale)
        plt.xlim(sig_min, sig_max)
        plt.ylim(z.min(), z.max())
        plt.xlabel("Conductivity (S/m)", fontsize=14)
        plt.ylabel("Depth (m)", fontsize=14)
        plt.ylabel("Depth (m)", fontsize=14)
        if showlayers == True:
            for locz in z_grid:
                plt.plot(
                    np.linspace(sig_min, sig_max, 100),
                    np.ones(100) * locz,
                    "b--",
                    lw=0.5,
                )
        return plt.plot(sigma, z, "k-", **kwargs)

    else:
        ax.set_xscale(xscale)
        ax.set_xlim(sig_min, sig_max)
        ax.set_ylim(z.min(), z.max())
        ax.set_xlabel("Conductivity (S/m)", fontsize=14)
        ax.set_ylabel("Depth (m)", fontsize=14)
        if showlayers == True:
            for locz in z_grid:
                ax.plot(
                    np.linspace(sig_min, sig_max, 100),
                    np.ones(100) * locz,
                    "b--",
                    lw=0.5,
                )
        return ax.plot(sigma, z, "k-", **kwargs)


def get_vertical_discretization(n_layer, minimum_dz, geomtric_factor):
    """
    Creates a list of vertical discretizations generate from a geometric series.

    >>> minimum_dz * geomtric_factor ** np.arange(n_layer)

    Parameters
    ----------
    n_layer : int
        The number of discretizations
    minimum_dz : float
        The smallest discretization
    geometric_factor : float
        The expansion factor of the discretizations

    Returns
    (n_layer) : numpy.ndarray
        The cell widths
    """
    hz = minimum_dz * (geomtric_factor) ** np.arange(n_layer)
    print(
        ">> Depth from the surface to the base of the bottom layer is {:.1f}m".format(
            hz[:].sum()
        )
    )
    return hz


def get_vertical_discretization_frequency(
    frequency,
    sigma_background=0.01,
    n_layer=19,
    hz_min=None,
    z_max=None,
    factor_fmax=4,
    factor_fmin=1.0,
):
    """
    Creates a list of recommended vertical discretizations.

    The vertical discretizations are determined based on the background conductivity
    and the frequency. They are intended to be used in a Layered one dimensional
    simulation.

    Parameters
    ----------
    frequency : numpy.ndarray
        The frequencies needed to represent.
    sigma_background : float, optional
        The background conductivity
    n_layer : int, optional
        Number of layers to generate
    hz_min : optional
        The minimum cell size. By default, it is estimated to be the maximum
        frequency skin depth divided by `factor_fmax`.
    z_max : optional
        The maximum depth of the cells. By default it is estimated as the minimum
        frequency skin depth times `factor_fmin`.
    factor_fmax, factor_fmin : float, optional
        The scaling factors to scale the minimum and maximum skin depth
        estimate, respectively

    Returns
    -------
    (n_layer) numpy.ndarray
        The cell widths.
    """
    if hz_min is None:
        hz_min = skin_depth(frequency.max(), sigma_background) / factor_fmax
    if z_max is None:
        z_max = skin_depth(frequency.min(), sigma_background) * factor_fmin
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
    z_sum = hz.sum()

    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
        z_sum = hz.sum()
    return hz


def get_vertical_discretization_time(
    time,
    sigma_background=0.01,
    n_layer=19,
    hz_min=None,
    z_max=None,
    factor_tmin=4,
    facter_tmax=1.0,
):
    """
    Creates a list of recommended vertical discretizations.

    The vertical discretizations are determined based on the background conductivity
    and the times. They are intended to be used in a Layered one dimensional
    simulation.

    Parameters
    ----------
    time : numpy.ndarray
        The times in seconds needed to represent.
    sigma_background : float, optional
        The background conductivity in S/m.
    n_layer : int, optional
        Number of layers to generate
    hz_min : optional
        The minimum cell size in meters. By default, it is estimated to be the minimum diffusion
        distance divided by `factor_tmin`.
    z_max : optional
        The maximum depth of the cells in meters. By default it is estimated as the maximum
        diffusion distance times `factor_tmax`.
    factor_tmin, factor_tmax : float, optional
        The scaling factors to scale the minimum and maximum diffusion distances
        estimate.

    Returns
    -------
    (n_layer) numpy.ndarray
        The cell widths.
    """
    if hz_min is None:
        hz_min = diffusion_distance(time.min(), sigma_background) / factor_tmin
    if z_max is None:
        z_max = diffusion_distance(time.max(), sigma_background) * facter_tmax
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
    z_sum = hz.sum()
    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
        z_sum = hz.sum()
    return hz


#############################################################
#       PHYSICAL PROPERTIES
#############################################################


def ColeCole(f, sig_inf=1e-2, eta=0.1, tau=0.1, c=1):
    """
    Computing Cole-Cole model in frequency domain

    Parameters
    ----------
    f : float or (n_freq) numpy.ndarray
        Frequency in Hz.
    sig_inf : float or (n_sig) numpy.ndarray, optional
        Conductivity at infinite time.
        If `numpy.ndarray` `f` must also be a `numpy.ndarray`
    eta, tau, c : float, optional
        Cole Cole control parameters

    Returns
    -------
    float, (n_freq) numpy.ndarray, or (n_freq, n_sig) numpy.ndarray

    Notes
    -----
    Defined as

    .. math ::
        \\sigma (\\omega ) = \\sigma_{\\infty} \\Bigg [
        1 - \\eta \\Bigg ( \\frac{1}{1 + (1-\\eta ) (1 + i\\omega \\tau)^c} \\Bigg )
        \\Bigg ]

    """

    if np.isscalar(sig_inf):
        w = 2 * np.pi * f
        sigma = sig_inf - sig_inf * eta / (1 + (1 - eta) * (1j * w * tau) ** c)
    else:
        sigma = np.zeros((f.size, sig_inf.size), dtype=complex)
        for i in range(f.size):
            w = 2 * np.pi * f[i]
            sigma[i, :] = utils.mkvc(
                sig_inf - sig_inf * eta / (1 + (1 - eta) * (1j * w * tau) ** c)
            )
    return sigma


def LogUniform(f, chi_inf=0.05, del_chi=0.05, tau1=1e-5, tau2=1e-2):
    """
    Computing relaxation model in the frequency domain for a log-uniform
    distribution of time-relaxation constants.

    Parameters
    ----------
    f : float or numpy.ndarray
        Frequency in Hz.
    chi_inf, del_chi, tau1, tau2, optional
        Relaxation model control parameters.

    Returns
    -------
    float or numpy.ndarray

    Notes
    -----

    .. math::
        \\chi (\\omega ) = \\chi_{\\infty} + \\Delta \\chi \\Bigg [
        1 - \\Bigg ( \\frac{1}{ln (\\tau_2 / \\tau_1 )} \\Bigg )
        ln \\Bigg ( \\frac{1 + i\\omega \\tau_2}{1 + i\\omega tau_1} ) \\Bigg )
        \\Bigg ]
    """

    w = 2 * np.pi * f
    return chi_inf + del_chi * (
        1 - np.log((1 + 1j * w * tau2) / (1 + 1j * w * tau1)) / np.log(tau2 / tau1)
    )
