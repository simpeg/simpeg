import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from geoana.em.fdem.base import skin_depth
from geoana.em.tdem import diffusion_distance

from SimPEG import utils
from scipy.constants import mu_0, pi
import scipy.special as spec

import properties
from scipy.spatial import cKDTree as kdtree
import scipy.sparse as sp
from matplotlib.colors import LogNorm
import warnings


def plot_layer(sig, mesh, xscale='log', ax=None, showlayers=False, xlim=None,**kwargs):
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

def get_vertical_discretization(n_layer, minimum_dz, geomtric_factor):
    hz = minimum_dz*(geomtric_factor)**np.arange(n_layer)
    print (">> Depth from the surface to the base of the bottom layer is {:.1f}m".format(hz[:].sum()))
    return hz

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



def get_2d_mesh(n_sounding, hz):
    """
        Generate 2D mesh for regularization

        hx:
        hz:

    """
    hx = np.ones(n_sounding)
    return TensorMesh([hz, hx])




#############################################################
#       PHYSICAL PROPERTIES
#############################################################

def ColeCole(f, sig_inf=1e-2, eta=0.1, tau=0.1, c=1):
    """
        Computing Cole-Cole model in frequency domain

        .. math ::
            \\sigma (\\omega ) = \\sigma_{\\infty} \\Bigg [
            1 - \\eta \\Bigg ( \\frac{1}{1 + (1-\\eta ) (1 + i\\omega \\tau)^c} \\Bigg )
            \\Bigg ]


    """

    if np.isscalar(sig_inf):
        w = 2*np.pi*f
        sigma = sig_inf - sig_inf*eta/(1+(1-eta)*(1j*w*tau)**c)
    else:
        sigma = np.zeros((f.size,sig_inf.size), dtype=complex)
        for i in range(f.size):
            w = 2*np.pi*f[i]
            sigma[i,:] = utils.mkvc(sig_inf - sig_inf*eta/(1+(1-eta)*(1j*w*tau)**c))
    return sigma


def LogUniform(f, chi_inf=0.05, del_chi=0.05, tau1=1e-5, tau2=1e-2):
    """
        Computing relaxation model in the frequency domain for a log-uniform
        distribution of time-relaxation constants.

        .. math::
            \\chi (\\omega ) = \\chi_{\\infty} + \\Delta \\chi \\Bigg [
            1 - \\Bigg ( \\frac{1}{ln (\\tau_2 / \\tau_1 )} \\Bigg )
            ln \\Bigg ( \\frac{1 + i\\omega \\tau_2}{1 + i\\omega tau_1} ) \\Bigg )
            \\Bigg ]


    """

    w = 2*np.pi*f
    return chi_inf + del_chi*(1 - np.log((1 + 1j*w*tau2)/(1 + 1j*w*tau1))/np.log(tau2/tau1))



def diffusion_distance(t, sigma):
    """
        Compute diffusion distance

        .. math::

            d = \sqrt{\\frac{2t}{\mu_0\sigma}}

        assume \\\\(\\\\ \mu = \mu_0\\\\) is chargeability
    """

    return np.sqrt(2*t/mu_0/sigma)


def skin_depth(f, sigma):
    """
        Compute skin depth

        .. math::

            \delta = \sqrt{\\frac{2}{\omega\mu_0\sigma}}

        where \\\\(\\\\ \omega = 2\pi f \\\\) is chargeability
    """

    return np.sqrt(2/mu_0/sigma/f/(2*pi))


#############################################################
#       VMD SOURCE SOLUTIONS
#############################################################

def Hz_vertical_magnetic_dipole(f, r, sig, flag="secondary"):

    """

        Hz component of analytic solution for half-space (VMD source)
        Src and Rx are on the surface

        .. math::

            H_z  = \\frac{m}{2\pi k^2 r^5} \
                    \left( 9 -(9+\imath\ kr - 4 k^2r^2 - \imath k^3r^3)e^{-\imath kr}\\right)

        * r: Src-Rx offset
        * m: magnetic dipole moment
        * k: propagation constant

        .. math::

            k = \omega^2\epsilon\mu - \imath\omega\mu\sigma


    """

    mu0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu0*sig)
    Hz = 1./(2*np.pi*k**2*r**5)*(9-(9+9*1j*k*r-4*k**2*r**2-1j*k**3*r**3)*np.exp(-1j*k*r))
    
    if flag == 'secondary':
        Hzp = -1/(4*np.pi*r**3)
        Hz = Hz-Hzp
    return Hz


def Hr_vertical_magnetic_dipole(f, r, sig):

    """

        Hz component of analytic solution for half-space (VMD source)
        Src and Rx are on the surface

        .. math::

            H_z  = \\frac{-mk^2}{4\pi \\rho} \
            \\Bigg [ I_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg )
            - I_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) \\Bigg ]

        * r: Src-Rx offset
        * m: magnetic dipole moment
        * k: propagation constant
        * :math:`I_n`: modified Bessel function of the 1st kind of order *n*
        * :math:`K_n`: modified Bessel function of the 2nd kind of order *n*

        .. math::

            k = \omega^2\epsilon\mu - \imath\omega\mu\sigma \\


    """

    mu0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu0*sig)
    alpha = 1j*k*r/2.

    IK1 = spec.iv(1, alpha)*spec.kv(1, alpha)
    IK2 = spec.iv(2, alpha)*spec.kv(2, alpha)

    Hr = (-k**2/(4*np.pi*r))*(IK1 - IK2)
    
    return Hr




def Hz_horizontal_magnetic_dipole(f, r, x, sig):

    """

        Hz component of analytic solution for half-space (HMD source)
        Src and Rx are on the surface

        .. math::

            H_z  = \\frac{mxk^2}{4\pi \\rho^2} \
            \\Bigg [ I_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_1 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg )
            - I_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) K_2 \\Bigg ( \\frac{\i k \\rho }{2} \\Bigg ) \\Bigg ]

        * x: Src-Rx x offset
        * r: Src-Rx offset
        * m: magnetic dipole moment
        * k: propagation constant
        * :math:`I_n`: modified Bessel function of the 1st kind of order *n*
        * :math:`K_n`: modified Bessel function of the 2nd kind of order *n*

        .. math::

            k = \omega^2\epsilon\mu - \imath\omega\mu\sigma \\


    """

    mu0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu0*sig)
    alpha = 1j*k*r/2.

    IK1 = spec.iv(1, alpha)*spec.kv(1, alpha)
    IK2 = spec.iv(2, alpha)*spec.kv(2, alpha)

    Hr = (x*k**2/(4*np.pi*r**2))*(IK1 - IK2)
    return Hr




def Bz_vertical_magnetic_dipole(r, t, sigma):

    theta = np.sqrt((sigma*mu_0)/(4*t))
    tr = theta*r
    etr = spec.erf(tr)
    t1 = (9/(2*tr**2) - 1)*etr
    t2 = (1/np.sqrt(pi))*(9/tr + 4*tr)*np.exp(-tr**2)
    hz = (t1 - t2)/(4*pi*r**3)
    return mu_0*hz



#############################################################
#       HORIZONTAL LOOP SOURCE SOLUTIONS
#############################################################


def Hz_horizontal_circular_loop(f, I, a, sig, flag="secondary"):

    """

        Hz component of analytic solution for half-space (Circular-loop source)
        Src and Rx are on the surface and receiver is located at the center of the loop.

        .. math::

            H_z  = -\\frac{I}{k^2a^3} \
                    \left( 3 -(3+\imath\ ka - k^2a^2 )e^{-\imath ka}\\right)

        * a: Src-loop radius
        * I: Current intensity

    """

    mu_0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu_0*sig)
    Hz = -I/(k**2*a**3)*(3-(3+3*1j*k*a-k**2*a**2)*np.exp(-1j*k*a))
    
    if flag == 'secondary':
        Hzp = I/2./a
        Hz = Hz-Hzp
    
    return Hz





def dHzdsiganalCirc(sig, f, I, a, flag):

    """
        Compute sensitivity for HzanalCirc by using perturbation

        .. math::

            \\frac{\partial H_z}{\partial \sigma}
            = \\frac{H_z(\sigma+\\triangle\sigma)- H_z(\sigma-\\triangle\sigma)}
                {2\\triangle\sigma}
    """
    mu_0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu_0*sig)
    perc = 0.001
    Hzfun = lambda m: HzanalCirc(m, f, I, a, flag)
    dHzdsig = (Hzfun(sig+perc*sig)-Hzfun(sig-perc*sig))/(2*perc*sig)
    return dHzdsig









def Bz_horizontal_circular_loop(a, t, sigma):
    """
        Hz component of analytic solution for half-space (Circular-loop source)
        Src and Rx are on the surface and receiver is located at the center of the loop.
        Src waveform here is step-off.

        .. math::

            h_z  = \\frac{I}{2a} \
                    \left( \\frac{3}{\sqrt{\pi}\\theta a}e^{-\\theta^2a^2}
                    +(1-\\frac{3}{2\\theta^2a^2})erf(\\theta a)\\right)

        .. math::

            \\theta = \sqrt{\\frac{\sigma\mu}{4t}}
    """

    theta = np.sqrt((sigma*mu_0)/(4*t))
    ta = theta*a
    eta = spec.erf(ta)
    t1 = (3/(np.sqrt(pi)*ta))*np.exp(-ta**2)
    t2 = (1 - (3/(2*ta**2)))*eta
    hz = (t1 + t2)/(2*a)
    return mu_0*hz


def dBzdt_horizontal_circular_loop(a, t, sigma):
    """
        Hz component of analytic solution for half-space (Circular-loop source)
        Src and Rx are on the surface and receiver is located at the center of the loop.
        Src waveform here is step-off.

        .. math::

            \\frac{\partial h_z}{\partial t}  = -\\frac{I}{\mu_0\sigma a^3} \
                    \left( 3erf(\\theta a) - \\frac{2}{\sqrt{\pi}}\\theta a (3+2\\theta^2 a^2) e^{-\\theta^2a^2}\\right)

        .. math::

            \\theta = \sqrt{\\frac{\sigma\mu}{4t}}
    """
    theta = np.sqrt((sigma*mu_0)/(4*t))
    const = -1/(mu_0*sigma*a**3)
    ta = theta*a
    eta = spec.erf(ta)
    t1 = 3*eta
    t2 = -2/(np.pi**0.5)*ta*(3+2*ta**2)*np.exp(-ta**2)
    dhzdt = const*(t1+t2)
    return mu_0*dhzdt


def Bz_horizontal_circular_loop_ColeCole(a, t, sigma):

    wt, tbase, omega_int = setFrequency(t)
    hz = Hz_horizontal_circular_loop(omega_int/2/np.pi, 1., a, sigma, 'secondary')
    # Treatment for inaccuracy in analytic solutions
    ind = omega_int < 0.2
    hz[ind] = 0.
    hzTD, f0 = transFilt(hz, wt, tbase, omega_int, t)
    return hzTD*mu_0


def dBzdt_horizontal_circular_loop_ColeCole(a, t, sigma):

    wt, tbase, omega_int = setFrequency(t)
    hz = Hz_horizontal_circular_loop(omega_int/2/np.pi, 1., a, sigma, 'secondary')
    # Treatment for inaccuracy in analytic solutions
    ind = omega_int < 0.2
    hz[ind] = 0.
    dhzdtTD = -transFiltImpulse(hz, wt, tbase, omega_int, t)

    return dhzdtTD*mu_0


def Bz_horizontal_circular_loop_VRM(a, z, h, t, dchi, tau1, tau2):

    mu0 = 4*np.pi*1e-7
    F = - (1/np.log(tau2/tau1)) * (spec.expi(t/tau2) + spec.expi(-t/tau1))
    B0 = (0.5*mu0*a**2) * (dchi/(2 + dchi)) * ((z + h)**2 + a**2)**-1.5
    return B0*F


def dBzdt_horizontal_circular_loop_VRM(a, z, h, t, dchi, tau1, tau2):

    mu0 = 4*np.pi*1e-7
    dFdt = (1/np.log(tau2/tau1)) * (np.exp(-t/tau1) - np.exp(-t/tau2)) / t
    B0 = (0.5*mu0*a**2) * (dchi/(2 + dchi)) * ((z + h)**2 + a**2)**-1.5
    return B0*dFdt

#############################################################
#       PLOTTING RESTIVITY MODEL
#############################################################
#TODO: revisit, and replace hz with thickness
#... clean up the code

class Stitched1DModel(properties.HasProperties):

    topography = properties.Array(
        "topography (x, y, z)", dtype=float,
        shape=('*', '*')
    )

    physical_property = properties.Array(
        "Physical property", dtype=float
    )

    line = properties.Array(
        "Line", dtype=float, default=None
    )

    time_stamp = properties.Array(
        "Time stamp", dtype=float, default=None
    )

    hz = properties.Array(
        "Vertical thickeness of 1D mesh", dtype=float
    )

    n_layer = properties.Integer("Number of layers")

    def __init__(self, **kwargs):
        super(Stitched1DModel, self).__init__(**kwargs)
        warnings.warn(
            "code under construction - API might change in the future"
        )
    @property
    def n_sounding(self):
        if getattr(self, '_n_sounding', None) is None:
            self._n_sounding = self.topography.shape[0]
        return self._n_sounding

    @property
    def unique_line(self):
        if getattr(self, '_unique_line', None) is None:
            if self.line is None:
                raise Exception("line information is required!")
            self._unique_line = np.unique(self.line)
        return self._unique_line

    @property
    def xyz(self):
        if getattr(self, '_xyz', None) is None:
            xyz = np.empty(
                (self.n_layer, self.topography.shape[0], 3), order='F'
            )
            for i_xy in range(self.topography.shape[0]):
                z = -self.mesh_1d.vectorCCx + self.topography[i_xy, 2]
                x = np.ones_like(z) * self.topography[i_xy, 0]
                y = np.ones_like(z) * self.topography[i_xy, 1]
                xyz[:, i_xy, :] = np.c_[x, y, z]
            self._xyz = xyz
        return self._xyz

    @property
    def mesh_1d(self):
        if getattr(self, '_mesh_1d', None) is None:
            if self.hz is None:
                raise Exception("hz information is required!")
            self._mesh_1d = set_mesh_1d(np.r_[self.hz[:self.n_layer]])
        return self._mesh_1d

    @property
    def mesh_3d(self):
        if getattr(self, '_mesh_3d', None) is None:
            if self.hz is None:
                raise Exception("hz information is required!")
            self._mesh_3d = set_mesh_3d(np.r_[self.hz[:self.n_layer-1], 1e20])
        return self._mesh_3d

    @property
    def physical_property_matrix(self):
        if getattr(self, '_physical_property_matrix', None) is None:
            if self.physical_property is None:
                raise Exception("physical_property information is required!")
            self._physical_property_matrix = self.physical_property.reshape((self.n_layer, self.n_sounding), order='F')
        return self._physical_property_matrix

    @property
    def depth_matrix(self):
        if getattr(self, '_depth_matrix', None) is None:
            if self.hz.size == self.n_layer:
                depth = np.cumsum(np.r_[0, self.hz])
                self._depth_matrix = np.tile(depth, (self.n_sounding, 1)).T
            else:
                self._depth_matrix =np.hstack(
                    (np.zeros((self.n_sounding,1)), np.cumsum(self.hz.reshape((self.n_sounding, self.n_layer)), axis=1))
                ).T
        return self._depth_matrix

    @property
    def distance(self):
        if getattr(self, '_distance', None) is None:
            self._distance = np.zeros(self.n_sounding, dtype=float)
            for line_tmp in self.unique_line:
                ind_line = self.line == line_tmp
                xy_line = self.topography[ind_line,:2]
                distance_line = np.r_[0, np.cumsum(np.sqrt((np.diff(xy_line, axis=0)**2).sum(axis=1)))]
                self._distance[ind_line] = distance_line
        return self._distance

    def plot_section(
        self, i_layer=0, i_line=0, x_axis='x',
        plot_type="contour",
        physical_property=None, clim=None,
        ax=None, cmap='viridis', ncontour=20, scale='log',
        show_colorbar=True, aspect=1, zlim=None, dx=20.,
        invert_xaxis=False,
        alpha=0.7,
        pcolorOpts={}
    ):
        ind_line = self.line == self.unique_line[i_line]
        if physical_property is not None:
            physical_property_matrix = physical_property.reshape(
                (self.n_layer, self.n_sounding), order='F'
            )
        else:
            physical_property_matrix = self.physical_property_matrix

        if x_axis.lower() == 'y':
            x_ind = 1
            xlabel = 'Northing (m)'
        elif x_axis.lower() == 'x':
            x_ind = 0
            xlabel = 'Easting (m)'
        elif x_axis.lower() == 'distance':
            xlabel = 'Distance (m)'

        if ax is None:
            fig = plt.figure(figsize=(15, 10))
            ax = plt.subplot(111)

        if clim is None:
            vmin = np.percentile(physical_property_matrix, 5)
            vmax = np.percentile(physical_property_matrix, 95)
        else:
            vmin, vmax = clim

        if scale == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
            vmin=None
            vmax=None
        else:
            norm=None

        ind_line = np.arange(ind_line.size)[ind_line]

        for i in ind_line:
            inds_temp = [i]
            if x_axis == 'distance':
                x_tmp = self.distance[i]
            else:
                x_tmp = self.topography[i, x_ind]

            topo_temp = np.c_[
                x_tmp-dx,
                x_tmp+dx
            ]
            out = ax.pcolormesh(
                topo_temp, -self.depth_matrix[:,i]+self.topography[i, 2], physical_property_matrix[:, inds_temp],
                cmap=cmap, alpha=alpha,
                vmin=vmin, vmax=vmax, norm=norm, shading='auto', **pcolorOpts
            )

        if show_colorbar:
            from mpl_toolkits import axes_grid1
            cb = plt.colorbar(out, ax=ax, fraction=0.01)
            cb.set_label("Conductivity (S/m)")

        ax.set_aspect(aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Elevation (m)')
        if zlim is not None:
            ax.set_ylim(zlim)

        if x_axis == 'distance':
            xlim = self.distance[ind_line].min()-dx, self.distance[ind_line].max()+dx
        else:
            xlim = self.topography[ind_line, x_ind].min()-dx, self.topography[ind_line, x_ind].max()+dx
        if invert_xaxis:
            ax.set_xlim(xlim[1], xlim[0])
        else:
            ax.set_xlim(xlim)

        plt.tight_layout()

        if show_colorbar:
            return out, ax, cb
        else:
            return out, ax
        return ax,

    def get_3d_mesh(
        self, dx=None, dy=None, dz=None,
        npad_x=0, npad_y=0, npad_z=0,
        core_z_length=None,
        nx=100,
        ny=100,
    ):

        xmin, xmax = self.topography[:, 0].min(), self.topography[:, 0].max()
        ymin, ymax = self.topography[:, 1].min(), self.topography[:, 1].max()
        zmin, zmax = self.topography[:, 2].min(), self.topography[:, 2].max()
        zmin -= self.mesh_1d.vectorNx.max()

        lx = xmax-xmin
        ly = ymax-ymin
        lz = zmax-zmin

        if dx is None:
            dx = lx/nx
            print ((">> dx:%.1e")%(dx))
        if dy is None:
            dy = ly/ny
            print ((">> dy:%.1e")%(dy))
        if dz is None:
            dz = np.median(self.mesh_1d.hx)

        nx = int(np.floor(lx/dx))
        ny = int(np.floor(ly/dy))
        nz = int(np.floor(lz/dz))

        if nx*ny*nz > 1e6:
            warnings.warn(
                ("Size of the mesh (%i) will greater than 1e6")%(nx*ny*nz)
            )
        hx = [(dx, npad_x, -1.2), (dx, nx), (dx, npad_x, -1.2)]
        hy = [(dy, npad_y, -1.2), (dy, ny), (dy, npad_y, -1.2)]
        hz = [(dz, npad_z, -1.2), (dz, nz)]

        zmin = self.topography[:, 2].max() - utils.meshTensor(hz).sum()
        self._mesh_3d = TensorMesh([hx, hy, hz], x0=[xmin, ymin, zmin])

        return self.mesh_3d

    @property
    def P(self):
        if getattr(self, '_P', None) is None:
            raise Exception("Run get_interpolation_matrix first!")
        return self._P

    def get_interpolation_matrix(
        self,
        npts=20,
        epsilon=None
    ):

        tree_2d = kdtree(self.topography[:, :2])
        xy = utils.ndgrid(self.mesh_3d.vectorCCx, self.mesh_3d.vectorCCy)

        distance, inds = tree_2d.query(xy, k=npts)
        if epsilon is None:
            epsilon = np.min([self.mesh_3d.hx.min(), self.mesh_3d.hy.min()])

        w = 1. / (distance + epsilon)**2
        w = utils.sdiag(1./np.sum(w, axis=1)) * (w)
        I = utils.mkvc(
            np.arange(inds.shape[0]).reshape([-1, 1]).repeat(npts, axis=1)
        )
        J = utils.mkvc(inds)

        self._P = sp.coo_matrix(
            (utils.mkvc(w), (I, J)),
            shape=(inds.shape[0], self.topography.shape[0])
        )

        mesh_1d = TensorMesh([np.r_[self.hz[:-1], 1e20]])

        z = self.P*self.topography[:, 2]

        self._actinds = utils.surface2ind_topo(self.mesh_3d, np.c_[xy, z])

        Z = np.empty(self.mesh_3d.vnC, dtype=float, order='F')
        Z = self.mesh_3d.gridCC[:, 2].reshape(
            (self.mesh_3d.nCx*self.mesh_3d.nCy, self.mesh_3d.nCz), order='F'
        )
        ACTIND = self._actinds.reshape(
            (self.mesh_3d.nCx*self.mesh_3d.nCy, self.mesh_3d.nCz), order='F'
        )

        self._Pz = []

        # This part can be cythonized or parallelized
        for i_xy in range(self.mesh_3d.nCx*self.mesh_3d.nCy):
            actind_temp = ACTIND[i_xy, :]
            z_temp = -(Z[i_xy, :] - z[i_xy])
            self._Pz.append(mesh_1d.getInterpolationMat(z_temp[actind_temp]))

    def interpolate_from_1d_to_3d(self, physical_property_1d):
        physical_property_2d = self.P*(
            physical_property_1d.reshape(
                (self.n_layer, self.n_sounding), order='F'
            ).T
        )
        physical_property_3d = np.ones(
            (self.mesh_3d.nCx*self.mesh_3d.nCy, self.mesh_3d.nCz),
            order='C', dtype=float
        ) * np.nan

        ACTIND = self._actinds.reshape(
            (self.mesh_3d.nCx*self.mesh_3d.nCy, self.mesh_3d.nCz), order='F'
        )

        for i_xy in range(self.mesh_3d.nCx*self.mesh_3d.nCy):
            actind_temp = ACTIND[i_xy, :]
            physical_property_3d[i_xy, actind_temp] = (
                self._Pz[i_xy]*physical_property_2d[i_xy, :]
            )

        return physical_property_3d
