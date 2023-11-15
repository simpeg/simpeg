import numpy as np
from geoana.em.fdem.base import skin_depth
from geoana.em.tdem import diffusion_distance
import matplotlib.pyplot as plt

from SimPEG import utils
from discretize import TensorMesh

from SimPEG.utils.code_utils import (
    validate_ndarray_with_shape,
)

from scipy.spatial import cKDTree as kdtree
import scipy.sparse as sp
from matplotlib.colors import LogNorm


def set_mesh_1d(hz):
    return TensorMesh([hz], x0=[0])


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
    r"""
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

        \sigma (\omega ) = \sigma_{\infty} \Bigg [
        1 - \eta \Bigg ( \frac{1}{1 + (1-\eta ) (1 + i\omega \tau)^c} \Bigg )
        \Bigg ]

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
    r"""
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

        \chi (\omega ) = \chi_{\infty} + \Delta \chi \Bigg [
        1 - \Bigg ( \frac{1}{ln (\tau_2 / \tau_1 )} \Bigg )
        ln \Bigg ( \frac{1 + i\omega \tau_2}{1 + i\omega tau_1} ) \Bigg )
        \Bigg ]
    """

    w = 2 * np.pi * f
    return chi_inf + del_chi * (
        1 - np.log((1 + 1j * w * tau2) / (1 + 1j * w * tau1)) / np.log(tau2 / tau1)
    )


#############################################################
#       PLOTTING RESTIVITY MODEL
#############################################################


class Stitched1DModel:
    def __init__(
        self,
        topography=None,
        physical_property=None,
        line=None,
        time_stamp=None,
        thicknesses=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.topography = topography
        self.physical_property = physical_property
        self.line = line
        self.time_stamp = time_stamp
        self.thicknesses = thicknesses

    @property
    def topography(self):
        """Topography

        Returns
        -------
        (n_sounding, n_dim) np.ndarray
            Topography.
        """
        return self._topography

    @topography.setter
    def topography(self, locs):
        self._topography = validate_ndarray_with_shape(
            "topography", locs, shape=("*", "*"), dtype=float
        )

    @property
    def physical_property(self):
        """Physical property

        Returns
        -------
        (n_sounding x n_layer,) np.ndarray
            physical_property.
        """
        return self._physical_property

    @physical_property.setter
    def physical_property(self, values):
        self._physical_property = validate_ndarray_with_shape(
            "physical_property", values, shape=("*"), dtype=float
        )

    @property
    def line(self):
        """Line number

        Returns
        -------
        (n_sounding,) np.ndarray
            line.
        """
        return self._line

    @line.setter
    def line(self, values):
        self._line = validate_ndarray_with_shape("line", values, shape=("*"), dtype=int)

    @property
    def timestamp(self):
        """Time stamp

        Returns
        -------
        (n_sounding,) np.ndarray
            timestamp.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, values):
        self._timestamp = validate_ndarray_with_shape(
            "timestamp", values, shape=("*"), dtype=float
        )

    @property
    def thicknesses(self):
        """Layer thicknesses

        Returns
        -------
        (n_sounding,) np.ndarray
            thicknesses.
        """
        return self._thicknesses

    @thicknesses.setter
    def thicknesses(self, values):
        self._thicknesses = validate_ndarray_with_shape(
            "thicknesses", values, shape=("*"), dtype=float
        )

    @property
    def n_layer(self):
        return len(self.hz)

    @property
    def hz(self):
        if getattr(self, "_hz", None) is None:
            self._hz = np.r_[self.thicknesses, self.thicknesses[-1]]
        return self._hz

    @property
    def n_sounding(self):
        if getattr(self, "_n_sounding", None) is None:
            self._n_sounding = self.topography.shape[0]
        return self._n_sounding

    @property
    def unique_line(self):
        if getattr(self, "_unique_line", None) is None:
            if self.line is None:
                raise Exception("line information is required!")
            self._unique_line = np.unique(self.line)
        return self._unique_line

    @property
    def xyz(self):
        if getattr(self, "_xyz", None) is None:
            xyz = np.empty((self.n_layer, self.topography.shape[0], 3), order="F")
            for i_xy in range(self.topography.shape[0]):
                z = -self.mesh_1d.vectorCCx + self.topography[i_xy, 2]
                x = np.ones_like(z) * self.topography[i_xy, 0]
                y = np.ones_like(z) * self.topography[i_xy, 1]
                xyz[:, i_xy, :] = np.c_[x, y, z]
            self._xyz = xyz
        return self._xyz

    @property
    def mesh_1d(self):
        if getattr(self, "_mesh_1d", None) is None:
            if self.thicknesses is None:
                raise Exception("thicknesses information is required!")
            self._mesh_1d = set_mesh_1d(np.r_[self.hz[: self.n_layer]])
        return self._mesh_1d

    @property
    def mesh_3d(self):
        if getattr(self, "_mesh_3d", None) is None:
            if self.mesh_3d is None:
                raise Exception("Run get_mesh_3d!")
        return self._mesh_3d

    @property
    def physical_property_matrix(self):
        if getattr(self, "_physical_property_matrix", None) is None:
            if self.physical_property is None:
                raise Exception("physical_property information is required!")
            self._physical_property_matrix = self.physical_property.reshape(
                (self.n_layer, self.n_sounding), order="F"
            )
        return self._physical_property_matrix

    @property
    def depth_matrix(self):
        if getattr(self, "_depth_matrix", None) is None:
            if self.hz.size == self.n_layer:
                depth = np.cumsum(np.r_[0, self.hz])
                self._depth_matrix = np.tile(depth, (self.n_sounding, 1)).T
            else:
                self._depth_matrix = np.hstack(
                    (
                        np.zeros((self.n_sounding, 1)),
                        np.cumsum(
                            self.hz.reshape((self.n_sounding, self.n_layer)), axis=1
                        ),
                    )
                ).T
        return self._depth_matrix

    @property
    def distance(self):
        if getattr(self, "_distance", None) is None:
            self._distance = np.zeros(self.n_sounding, dtype=float)
            for line_tmp in self.unique_line:
                ind_line = self.line == line_tmp
                xy_line = self.topography[ind_line, :2]
                distance_line = np.r_[
                    0, np.cumsum(np.sqrt((np.diff(xy_line, axis=0) ** 2).sum(axis=1)))
                ]
                self._distance[ind_line] = distance_line
        return self._distance

    def plot_section(
        self,
        i_layer=0,
        i_line=0,
        x_axis="x",
        plot_type="contour",
        physical_property=None,
        clim=None,
        ax=None,
        cmap="viridis",
        ncontour=20,
        scale="log",
        show_colorbar=True,
        aspect=1,
        zlim=None,
        dx=20.0,
        invert_xaxis=False,
        alpha=0.7,
        pcolorOpts={},
    ):
        ind_line = self.line == self.unique_line[i_line]
        if physical_property is not None:
            physical_property_matrix = physical_property.reshape(
                (self.n_layer, self.n_sounding), order="F"
            )
        else:
            physical_property_matrix = self.physical_property_matrix

        if x_axis.lower() == "y":
            x_ind = 1
            xlabel = "Northing (m)"
        elif x_axis.lower() == "x":
            x_ind = 0
            xlabel = "Easting (m)"
        elif x_axis.lower() == "distance":
            xlabel = "Distance (m)"

        if ax is None:
            plt.figure(figsize=(15, 10))
            ax = plt.subplot(111)

        if clim is None:
            vmin = np.percentile(physical_property_matrix, 5)
            vmax = np.percentile(physical_property_matrix, 95)
        else:
            vmin, vmax = clim

        if scale == "log":
            norm = LogNorm(vmin=vmin, vmax=vmax)
            vmin = None
            vmax = None
        else:
            norm = None

        ind_line = np.arange(ind_line.size)[ind_line]

        for i in ind_line:
            inds_temp = [i]
            if x_axis == "distance":
                x_tmp = self.distance[i]
            else:
                x_tmp = self.topography[i, x_ind]

            topo_temp = np.c_[x_tmp - dx, x_tmp + dx]
            out = ax.pcolormesh(
                topo_temp,
                -self.depth_matrix[:, i] + self.topography[i, 2],
                physical_property_matrix[:, inds_temp],
                cmap=cmap,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                shading="auto",
                **pcolorOpts
            )

        if show_colorbar:
            cb = plt.colorbar(out, ax=ax, fraction=0.01)
            cb.set_label("Conductivity (S/m)")

        ax.set_aspect(aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Elevation (m)")
        if zlim is not None:
            ax.set_ylim(zlim)

        if x_axis == "distance":
            xlim = (
                self.distance[ind_line].min() - dx,
                self.distance[ind_line].max() + dx,
            )
        else:
            xlim = (
                self.topography[ind_line, x_ind].min() - dx,
                self.topography[ind_line, x_ind].max() + dx,
            )
        if invert_xaxis:
            ax.set_xlim(xlim[1], xlim[0])
        else:
            ax.set_xlim(xlim)

        plt.tight_layout()

        if show_colorbar:
            return out, ax, cb
        else:
            return out, ax
        return (ax,)

    def get_3d_mesh(
        self,
        dx=None,
        dy=None,
        dz=None,
        npad_x=0,
        npad_y=0,
        npad_z=0,
        core_z_length=None,
        nx=100,
        ny=100,
    ):
        xmin, xmax = self.topography[:, 0].min(), self.topography[:, 0].max()
        ymin, ymax = self.topography[:, 1].min(), self.topography[:, 1].max()
        zmin, zmax = self.topography[:, 2].min(), self.topography[:, 2].max()
        zmin -= self.mesh_1d.vectorNx.max()

        lx = xmax - xmin
        ly = ymax - ymin
        lz = zmax - zmin

        if dx is None:
            dx = lx / nx
            print((">> dx:%.1e") % (dx))
        if dy is None:
            dy = ly / ny
            print((">> dy:%.1e") % (dy))
        if dz is None:
            dz = np.median(self.mesh_1d.hx)

        nx = int(np.floor(lx / dx))
        ny = int(np.floor(ly / dy))
        nz = int(np.floor(lz / dz))

        if nx * ny * nz > 1e6:
            warnings.warn(
                ("Size of the mesh (%i) will greater than 1e6") % (nx * ny * nz)
            )
        hx = [(dx, npad_x, -1.2), (dx, nx), (dx, npad_x, -1.2)]
        hy = [(dy, npad_y, -1.2), (dy, ny), (dy, npad_y, -1.2)]
        hz = [(dz, npad_z, -1.2), (dz, nz)]

        zmin = self.topography[:, 2].max() - utils.meshTensor(hz).sum()
        self._mesh_3d = TensorMesh([hx, hy, hz], x0=[xmin, ymin, zmin])

        return self.mesh_3d

    @property
    def P(self):
        if getattr(self, "_P", None) is None:
            raise Exception("Run get_interpolation_matrix first!")
        return self._P

    def get_interpolation_matrix(self, npts=20, epsilon=None):
        tree_2d = kdtree(self.topography[:, :2])
        xy = utils.ndgrid(self.mesh_3d.vectorCCx, self.mesh_3d.vectorCCy)

        distance, inds = tree_2d.query(xy, k=npts)
        if epsilon is None:
            epsilon = np.min([self.mesh_3d.hx.min(), self.mesh_3d.hy.min()])

        w = 1.0 / (distance + epsilon) ** 2
        w = utils.sdiag(1.0 / np.sum(w, axis=1)) * (w)
        I = utils.mkvc(np.arange(inds.shape[0]).reshape([-1, 1]).repeat(npts, axis=1))
        J = utils.mkvc(inds)

        self._P = sp.coo_matrix(
            (utils.mkvc(w), (I, J)), shape=(inds.shape[0], self.topography.shape[0])
        )

        mesh_1d = TensorMesh([np.r_[self.hz[:-1], 1e20]])

        z = self.P * self.topography[:, 2]

        self._actinds = utils.surface2ind_topo(self.mesh_3d, np.c_[xy, z])

        Z = np.empty(self.mesh_3d.vnC, dtype=float, order="F")
        Z = self.mesh_3d.gridCC[:, 2].reshape(
            (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz), order="F"
        )
        ACTIND = self._actinds.reshape(
            (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz), order="F"
        )

        self._Pz = []

        # This part can be cythonized or parallelized
        for i_xy in range(self.mesh_3d.nCx * self.mesh_3d.nCy):
            actind_temp = ACTIND[i_xy, :]
            z_temp = -(Z[i_xy, :] - z[i_xy])
            self._Pz.append(mesh_1d.getInterpolationMat(z_temp[actind_temp]))

    def interpolate_from_1d_to_3d(self, physical_property_1d):
        physical_property_2d = self.P * (
            physical_property_1d.reshape((self.n_layer, self.n_sounding), order="F").T
        )
        physical_property_3d = (
            np.ones(
                (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz),
                order="C",
                dtype=float,
            )
            * np.nan
        )

        ACTIND = self._actinds.reshape(
            (self.mesh_3d.nCx * self.mesh_3d.nCy, self.mesh_3d.nCz), order="F"
        )

        for i_xy in range(self.mesh_3d.nCx * self.mesh_3d.nCy):
            actind_temp = ACTIND[i_xy, :]
            physical_property_3d[i_xy, actind_temp] = (
                self._Pz[i_xy] * physical_property_2d[i_xy, :]
            )

        return physical_property_3d
