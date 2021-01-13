import properties
import numpy as np
from scipy.spatial import cKDTree as kdtree
import scipy.sparse as sp
from SimPEG import utils
from discretize import TensorMesh
from .utils import set_mesh_1d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings


class DataIO(properties.HasProperties):

    def __init__(self, **kwargs):
        super(DataIO, self).__init__(**kwargs)
        warnings.warn(
            "code under construction - API might change in the future"
        )

    def read_geosoft_xyz(self, data, header):
        pass

    def plot_profile_data(self, line_index=None):
        pass

    def plot_map_data(self, time_index=None, time=None):
        pass

    def plot_time_decay(self, time_index=None, time=None):
        pass


class ModelIO(properties.HasProperties):

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

    hz = properties.Array(
        "Vertical thickeness of 1D mesh", dtype=float
    )

    def __init__(self, **kwargs):
        super(ModelIO, self).__init__(**kwargs)
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
                (self.hz.size, self.topography.shape[0], 3), order='F'
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
            self._mesh_1d = set_mesh_1d(np.r_[self.hz])
        return self._mesh_1d

    @property
    def mesh_3d(self):
        if getattr(self, '_mesh_3d', None) is None:
            if self.hz is None:
                raise Exception("hz information is required!")
            self._mesh_3d = set_mesh_3d(np.r_[self.hz[:-1], 1e20])
        return self._mesh_3d

    @property
    def physical_property_matrix(self):
        if getattr(self, '_physical_property_matrix', None) is None:
            if self.physical_property is None:
                raise Exception("physical_property information is required!")
            self._physical_property_matrix = self.physical_property.reshape((self.hz.size, self.n_sounding), order='F')
        return self._physical_property_matrix

    def plot_plan(
            self, i_layer=0, i_line=0, show_line=False,
            physical_property=None, clim=None,
            ax=None, cmap='viridis', ncontour=20, scale='log',
            show_colorbar=True, aspect=1,
            contourOpts={}
    ):
        ind_line = self.line == self.unique_line[i_line]
        if physical_property is not None:
            physical_property_matrix = physical_property.reshape(
                (self.hz.size, self.n_sounding), order='F'
            )
        else:
            physical_property_matrix = self.physical_property_matrix

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(111)

        if clim is None:
            vmin = np.percentile(physical_property_matrix, 5)
            vmax = np.percentile(physical_property_matrix, 95)
        else:
            vmin, vmax = clim

        if scale == 'log':
            contourOpts['vmin'] = np.log10(vmin)
            contourOpts['vmax'] = np.log10(vmax)
            norm = LogNorm()
        else:
            norm = None

        contourOpts['cmap'] = cmap

        im = utils.plot2Ddata(
            self.topography[:, :2], utils.mkvc(physical_property_matrix[i_layer, :]), scale=scale,
            ncontour=ncontour, ax=ax,
            contourOpts=contourOpts, dataloc=False,
        )

        out = ax.scatter(
            self.topography[:, 0], self.topography[:, 1],
            c=physical_property_matrix[i_layer, :], s=0.5, vmin=vmin, vmax=vmax,
            cmap=cmap, alpha=1, norm=norm
        )

        if show_line:
            ax.plot(self.topography[ind_line,0], self.topography[ind_line,1], 'k.')

        if show_colorbar:
            from mpl_toolkits import axes_grid1
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = plt.colorbar(out, cax=cax)
#             cb.set_label("Conductivity (S/m)")
        ax.set_aspect(aspect)
        ax.set_title(("At %.1f m below surface")%(self.mesh_1d.vectorCCx[i_layer]))
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(True)
        plt.tight_layout()
#         plt.show()
        if show_colorbar:
            return out, ax, cb
        else:
            return out, ax

    def plot_section(
        self, i_layer=0, i_line=0, line_direction='x',
        show_layer=False,
        plot_type="contour",
        physical_property=None, clim=None,
        ax=None, cmap='viridis', ncontour=20, scale='log',
        show_colorbar=True, aspect=1, zlim=None, dx=20.,
        contourOpts={}
    ):
        ind_line = self.line == self.unique_line[i_line]
        if physical_property is not None:
            physical_property_matrix = physical_property.reshape(
                (self.hz.size, self.n_sounding), order='F'
            )
        else:
            physical_property_matrix = self.physical_property_matrix

        if line_direction.lower() == 'y':
            x_ind = 1
            xlabel = 'Northing (m)'
        elif line_direction.lower() == 'x':
            x_ind = 0
            xlabel = 'Easting (m)'

        yz = self.xyz[:, ind_line, :][:,:,[x_ind,2]].reshape(
            (int(self.hz.size*ind_line.sum()), 2), order='F'
        )

        if ax is None:
            fig = plt.figure(figsize=(15, 10))
            ax = plt.subplot(111)

        if clim is None:
            vmin = np.percentile(physical_property_matrix, 5)
            vmax = np.percentile(physical_property_matrix, 95)
        else:
            vmin, vmax = clim
        if plot_type == "contour":
            if scale == 'log':
                contourOpts['vmin'] = np.log10(vmin)
                contourOpts['vmax'] = np.log10(vmax)
                norm = LogNorm()
            else:
                norm=None

            contourOpts['cmap'] = cmap
            im = utils.plot2Ddata(
                yz, utils.mkvc(physical_property_matrix[:, ind_line]), scale='log', ncontour=40, dataloc=False, ax=ax,
                contourOpts=contourOpts
            )
            ax.fill_between(self.topography[ind_line, 1], self.topography[ind_line, 2], y2=yz[:,1].max(), color='w')

            out = ax.scatter(
                yz[:, 0], yz[:, 1],
                c=utils.mkvc(physical_property_matrix[:, ind_line]), s=0.1, vmin=vmin, vmax=vmax,
                cmap=cmap, alpha=1, norm=norm
            )
        elif plot_type == "pcolor":
            if scale == 'log':
                norm = LogNorm()
            else:
                norm=None
            ind_line = np.arange(ind_line.size)[ind_line]
            for i in ind_line:
                inds_temp = [i, i]
                topo_temp = np.c_[
                    self.topography[i, x_ind]-dx,
                    self.topography[i, x_ind]+dx
                ]
                out = ax.pcolormesh(
                    topo_temp, -self.mesh_1d.vectorCCx+self.topography[i, 2], physical_property_matrix[:, inds_temp],
                    cmap=cmap, alpha=0.7,
                    vmin=vmin, vmax=vmax, norm=norm
                )
        if show_layer:
            ax.plot(
                self.topography[ind_line, x_ind], self.topography[ind_line, 2]-self.mesh_1d.vectorCCx[i_layer],
                '--', lw=1, color='grey'
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

        xlim = self.topography[ind_line, x_ind].min(), self.topography[ind_line, x_ind].max()
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
            print ((">> dx:%.1e")%(dy))
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
                (self.hz.size, self.n_sounding), order='F'
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
