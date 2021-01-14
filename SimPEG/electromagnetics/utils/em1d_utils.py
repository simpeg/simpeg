import numpy as np
import matplotlib.pyplot as plt
import scipy
from discretize import TensorMesh
from SimPEG import maps, utils
from ..analytics import skin_depth, diffusion_distance
from ..simulation import EM1DFMSimulation, EM1DTMSimulation
from ..survey import EM1DSurveyFD, EM1DSurveyTD

def plotLayer(sig, mesh, xscale='log', ax=None, showlayers=False, xlim=None,**kwargs):
    """
        Plot Conductivity model for the layered earth model
    """

    z_grid = mesh.vectorNx
    n_sig = sig.size
    sigma = np.repeat(sig, 2)
    z = []
    for i in range(n_sig):
        z.append(np.r_[z_grid[i], z_grid[i+1]])
    z = np.hstack(z)
    if xlim == None:
        sig_min = sig[~np.isnan(sig)].min()*0.5
        sig_max = sig[~np.isnan(sig)].max()*2
    else:
        sig_min, sig_max = xlim

    if xscale == 'linear' and sig.min() == 0.:
        if xlim == None:
            sig_min = -sig[~np.isnan(sig)].max()*0.5
            sig_max = sig[~np.isnan(sig)].max()*2

    if ax==None:
        plt.xscale(xscale)
        plt.xlim(sig_min, sig_max)
        plt.ylim(z.min(), z.max())
        plt.xlabel('Conductivity (S/m)', fontsize = 14)
        plt.ylabel('Depth (m)', fontsize = 14)
        plt.ylabel('Depth (m)', fontsize = 14)
        if showlayers == True:
            for locz in z_grid:
                plt.plot(np.linspace(sig_min, sig_max, 100), np.ones(100)*locz, 'b--', lw = 0.5)
        return plt.plot(sigma, z, 'k-', **kwargs)

    else:
        ax.set_xscale(xscale)
        ax.set_xlim(sig_min, sig_max)
        ax.set_ylim(z.min(), z.max())
        ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
        ax.set_ylabel('Depth (m)', fontsize = 14)
        if showlayers == True:
            for locz in z_grid:
                ax.plot(np.linspace(sig_min, sig_max, 100), np.ones(100)*locz, 'b--', lw = 0.5)
        return ax.plot(sigma, z, 'k-', **kwargs)

def plotComplexData(frequency, val, xscale='log', ax=None, **kwargs):
    """
        Plot Complex EM responses
        * Complex value val should be sorted as:
            val = [val0.real, val1.real, val2.real ..., val0.imag, val1.imag, ...]
    """
    Nfreq = frequency.size
    if ax==None:

        plt.semilogx(frequency, val[:Nfreq], 'b', **kwargs)
        plt.xlabel('Frequency (Hz)', fontsize = 14)
        plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        return plt.semilogx(frequency, val[Nfreq:], 'r', **kwargs)
    else:

        ax.semilogx(frequency, val[:Nfreq], 'b', **kwargs)
        ax.set_xlabel('Frequency (Hz)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

        return ax.semilogx(frequency, val[Nfreq:], 'r', **kwargs)

def get_vertical_discretization_frequency(
    frequency, sigma_background=0.01,
    factor_fmax=4, factor_fmin=1., n_layer=19,
    hz_min=None, z_max=None
):
    if hz_min is None:
        hz_min = skin_depth(frequency.max(), sigma_background) / factor_fmax
    if z_max is None:
        z_max = skin_depth(frequency.min(), sigma_background) * factor_fmin
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
    z_sum = hz.sum()

    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
        z_sum = hz.sum()
    return hz


def get_vertical_discretization_time(
    time, sigma_background=0.01,
    factor_tmin=4, facter_tmax=1., n_layer=19,
    hz_min=None, z_max=None
):
    if hz_min is None:
        hz_min = diffusion_distance(time.min(), sigma_background) / factor_tmin
    if z_max is None:
        z_max = diffusion_distance(time.max(), sigma_background) * facter_tmax
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
    z_sum = hz.sum()
    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
        z_sum = hz.sum()
    return hz


def set_mesh_1d(hz):
    return TensorMesh([hz], x0=[0])
