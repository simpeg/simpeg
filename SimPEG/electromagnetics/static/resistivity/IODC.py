import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import properties
import warnings

from discretize import TensorMesh, TreeMesh
from discretize.base import BaseMesh
from discretize.utils import refine_tree_xyz, mkvc, meshTensor

from ....data import Data
from ....utils import sdiag, uniqueRows, surface2ind_topo, plot2Ddata
from ..utils import geometric_factor
from . import sources as Src
from . import receivers as Rx
from .survey import Survey


class IO(properties.HasProperties):
    """

    """

    # Survey
    survey_layout = properties.StringChoice(
        "Survey geometry of DC surveys",
        default="SURFACE",
        choices=["SURFACE", "BOREHOLE", "GENERAL"],
    )

    survey_type = properties.StringChoice(
        "DC-IP Survey type",
        default="dipole-dipole",
        choices=["dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"],
    )

    dimension = properties.Integer(
        "Dimension of electrode locations", default=2, required=True
    )

    a_locations = properties.Array(
        "locations of the positive (+) current electrodes",
        required=True,
        shape=("*", "*"),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float,  # data are floats
    )

    b_locations = properties.Array(
        "locations of the negative (-) current electrodes",
        required=True,
        shape=("*", "*"),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float,  # data are floats
    )

    m_locations = properties.Array(
        "locations of the positive (+) potential electrodes",
        required=True,
        shape=("*", "*"),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float,  # data are floats
    )

    n_locations = properties.Array(
        "locations of the negative (-) potential electrodes",
        required=True,
        shape=("*", "*"),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float,  # data are floats
    )

    electrode_locations = properties.Array(
        "unique locations of a, b, m, n electrodes",
        required=True,
        shape=("*", "*"),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float,  # data are floats
    )

    # Data
    data_dc_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=["volt", "apparent_resistivity", "apparent_conductivity",],
    )

    data_dc = properties.Array(
        "Measured DC data", shape=("*",), dtype=float  # data are floats
    )

    data_ip_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=["volt", "apparent_chargeability",],
    )

    data_ip = properties.Array(
        "Measured IP data", shape=("*",), dtype=float  # data are floats
    )

    data_sip_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=["volt", "apparent_chargeability",],
    )

    data_sip = properties.Array(
        "Measured Spectral IP data", shape=("*", "*"), dtype=float  # data are floats
    )

    times_ip = properties.Array(
        "Time channels of measured Spectral IP voltages (s)",
        required=True,
        shape=("*",),
        dtype=float,  # data are floats
    )

    G = properties.Array(
        "Geometric factor of DC-IP survey", shape=("*",), dtype=float  # data are floats
    )

    grids = properties.Array(
        "Spatial grids for plotting pseudo-section",
        shape=("*", "*"),
        dtype=float,  # data are floats
    )

    space_type = properties.StringChoice(
        "Assumption to compute apparent resistivity",
        default="half-space",
        choices=["half-space", "whole-space"],
    )

    line_inds = properties.Array(
        "Line indices", required=True, shape=("*",), dtype=int  # data are floats
    )
    sort_inds = properties.Array(
        "Sorting indices from ABMN",
        required=True,
        shape=("*",),
        dtype=int,  # data are floats
    )

    # Related to Physics and Discretization
    mesh = properties.Instance("Mesh for discretization", BaseMesh, required=True)

    dx = properties.Float("Length of corecell in x-direction", required=True,)
    dy = properties.Float("Length of corecell in y-direction", required=True)
    dz = properties.Float("Length of corecell in z-direction", required=True)

    npad_x = properties.Integer(
        "The number of padding cells x-direction", required=True, default=5
    )

    npad_y = properties.Integer(
        "The number of padding cells y-direction", required=True, default=5
    )

    npad_z = properties.Integer(
        "The number of padding cells z-direction", required=True, default=5
    )

    pad_rate_x = properties.Float(
        "Expansion rate of padding cells in  x-direction", required=True, default=1.3
    )

    pad_rate_y = properties.Float(
        "Expansion rate of padding cells in  y-direction", required=True, default=1.3
    )

    pad_rate_z = properties.Float(
        "Expansion rate of padding cells in  z-direction", required=True, default=1.3
    )

    ncell_per_dipole = properties.Integer(
        "The number of cells between dipole electrodes", required=True, default=4
    )

    corezlength = properties.Float("Core depth (m)", required=True,)

    # For synthetic surveys
    x0 = None
    lineLength = None
    a = None
    n_spacing = None
    n_data = None

    def __init__(self, **kwargs):
        super(IO, self).__init__(**kwargs)
        warnings.warn("code under construction - API might change in the future")

    # Properties
    @property
    def voltages(self):
        """
        Votages (V)
        """
        if self.data_dc_type.lower() == "volt":
            return self.data_dc
        elif self.data_dc_type.lower() == "apparent_resistivity":
            return self.data_dc * self.G
        elif self.data_dc_type.lower() == "apparent_conductivity":
            return self.apparent_conductivity / (self.data_dc * self.G)
        else:
            raise NotImplementedError()

    @property
    def apparent_resistivity(self):
        """
        Apparent Resistivity (Ohm-m)
        """
        if self.data_dc_type.lower() == "apparent_resistivity":
            return self.data_dc
        elif self.data_dc_type.lower() == "volt":
            return self.data_dc / self.G
        elif self.data_dc_type.lower() == "apparent_conductivity":
            return 1.0 / self.data_dc
        else:
            print(self.data_dc_type.lower())
            raise NotImplementedError()

    @property
    def apparent_conductivity(self):
        """
        Apparent Conductivity (S/m)
        """
        if self.data_dc_type.lower() == "apparent_conductivity":
            return self.data_dc
        elif self.data_dc_type.lower() == "apparent_resistivity":
            return 1.0 / self.data_dc
        elif self.data_dc_type.lower() == "volt":
            return 1.0 / self.data_dc * self.G

    # For IP
    @property
    def voltages_ip(self):
        """
        IP votages (V)
        """
        if self.data_ip_type.lower() == "volt":
            return self.data_ip
        elif self.data_ip_type.lower() == "apparent_chargeability":
            if self.voltages is None:
                raise Exception("DC voltages must be set to compute IP voltages")
            return self.data_ip * self.voltages
        else:
            raise NotImplementedError()

    # For SIP
    @property
    def voltages_sip(self):
        """
        IP votages (V)
        """
        if self.data_sip_type.lower() == "volt":
            return self.data_sip
        elif self.data_sip_type.lower() == "apparent_chargeability":
            if self.voltages is None:
                raise Exception("DC voltages must be set to compute IP voltages")
            return sdiag(self.voltages) * self.data_sip
        else:
            raise NotImplementedError()

    @property
    def apparent_chargeability(self):
        """
        Apparent Conductivity (S/m)
        """
        if self.data_ip_type.lower() == "apparent_chargeability":
            return self.data_ip
        elif self.data_ip_type.lower() == "volt":
            if self.voltages is None:
                raise Exception(
                    "DC voltages must be set to compute Apparent Chargeability"
                )
            return self.data_ip / self.voltages
        else:
            raise NotImplementedError()

    # For SIP
    @property
    def apparent_chargeability_sip(self):
        """
        Apparent Conductivity (S/m)
        """
        if self.data_sip_type.lower() == "apparent_chargeability":
            return self.data_sip
        elif self.data_sip_type.lower() == "volt":
            if self.voltages is None:
                raise Exception(
                    "DC voltages must be set to compute Apparent Chargeability"
                )
            return sdiag(1.0 / self.voltages) * self.data_sip
        else:
            raise NotImplementedError()

    def geometric_factor(self, survey):
        """
        Compute geometric factor, G, using locational informaition
        in survey object
        """
        G = geometric_factor(
            survey, survey_type=self.survey_type, space_type=self.space_type
        )
        return G

    def from_ambn_locations_to_survey(
        self,
        a_locations,
        b_locations,
        m_locations,
        n_locations,
        survey_type=None,
        data_dc=None,
        data_ip=None,
        data_sip=None,
        data_dc_type="volt",
        data_ip_type="volt",
        data_sip_type="volt",
        fname=None,
        dimension=2,
        line_inds=None,
        times_ip=None,
    ):
        """
        read A, B, M, N electrode location and data (V or apparent_resistivity)
        """
        self.a_locations = a_locations.copy()
        self.b_locations = b_locations.copy()
        self.m_locations = m_locations.copy()
        self.n_locations = n_locations.copy()
        self.survey_type = survey_type
        self.dimension = dimension
        self.data_dc_type = data_dc_type
        self.data_ip_type = data_ip_type
        self.data_sip_type = data_sip_type
        if times_ip is not None:
            self.times_ip = times_ip

        uniqSrc = uniqueRows(np.c_[self.a_locations, self.b_locations])
        uniqElec = uniqueRows(
            np.vstack(
                (self.a_locations, self.b_locations, self.m_locations, self.n_locations)
            )
        )
        self.electrode_locations = uniqElec[0]
        nSrc = uniqSrc[0].shape[0]
        ndata = self.a_locations.shape[0]

        if self.survey_layout == "SURFACE":
            # 2D locations
            srcLists = []
            sort_inds = []
            for iSrc in range(nSrc):
                inds = uniqSrc[2] == iSrc
                sort_inds.append(np.arange(ndata)[inds])

                locsM = self.m_locations[inds, :]
                locsN = self.n_locations[inds, :]
                if survey_type in ["dipole-dipole", "pole-dipole"]:
                    rx = Rx.Dipole(locsM, locsN)
                elif survey_type in ["dipole-pole", "pole-pole"]:
                    rx = Rx.Pole(locsM)
                else:
                    raise NotImplementedError()

                if dimension == 2:
                    locA = uniqSrc[0][iSrc, :2]
                    locB = uniqSrc[0][iSrc, 2:]
                elif dimension == 3:
                    locA = uniqSrc[0][iSrc, :3]
                    locB = uniqSrc[0][iSrc, 3:]

                if survey_type in ["dipole-dipole", "dipole-pole"]:
                    src = Src.Dipole([rx], locA, locB)
                elif survey_type in ["pole-dipole", "pole-pole"]:
                    src = Src.Pole([rx], locA)

                srcLists.append(src)

            self.sort_inds = np.hstack(sort_inds)

            if dimension == 2:
                survey = Survey(srcLists)
            elif dimension == 3:
                survey = Survey(srcLists)
            else:
                raise NotImplementedError()

            self.a_locations = self.a_locations[self.sort_inds, :]
            self.b_locations = self.b_locations[self.sort_inds, :]
            self.m_locations = self.m_locations[self.sort_inds, :]
            self.n_locations = self.n_locations[self.sort_inds, :]
            self.G = self.geometric_factor(survey)

            if data_dc is not None:
                self.data_dc = data_dc[self.sort_inds]
            if data_ip is not None:
                self.data_ip = data_ip[self.sort_inds]
            if data_sip is not None:
                self.data_sip = data_sip[self.sort_inds, :]
            if line_inds is not None:
                self.line_inds = line_inds[self.sort_inds]
            # Here we ignore ... z-locations
            self.n_data = survey.nD

            midABx = (self.a_locations[:, 0] + self.b_locations[:, 0]) * 0.5
            midMNx = (self.m_locations[:, 0] + self.n_locations[:, 0]) * 0.5

            if dimension == 2:
                z = abs(midABx - midMNx) * 1.0 / 3.0
                x = (midABx + midMNx) * 0.5
                zmax = z.max()
                a = abs(np.diff(np.sort(self.electrode_locations[:, 0]))).min()
                # Consider the case of 1D types of array
                if np.all(zmax < a):
                    z = abs(self.a_locations[:, 0] - self.b_locations[:, 0]) / 3.0
                self.grids = np.c_[x, z]

            elif dimension == 3:
                midABy = (self.a_locations[:, 1] + self.b_locations[:, 1]) * 0.5
                midMNy = (self.m_locations[:, 1] + self.n_locations[:, 1]) * 0.5
                z = np.sqrt((midABx - midMNx) ** 2 + (midABy - midMNy) ** 2) * 1.0 / 3.0
                x = (midABx + midMNx) * 0.5
                y = (midABy + midMNy) * 0.5
                self.grids = np.c_[x, y, z]
            else:
                raise Exception()
        else:
            raise NotImplementedError()
        return survey

    def set_mesh(
        self,
        topo=None,
        dx=None,
        dy=None,
        dz=None,
        corezlength=None,
        npad_x=None,
        npad_y=None,
        npad_z=None,
        pad_rate_x=None,
        pad_rate_y=None,
        pad_rate_z=None,
        ncell_per_dipole=None,
        mesh_type="TensorMesh",
        dimension=2,
        method="nearest",
    ):
        """
        Set up a mesh for a given DC survey
        """

        # Update properties
        if npad_x is None:
            npad_x = self.npad_x
        self.npad_x = npad_x

        if npad_z is None:
            npad_z = self.npad_z
        self.npad_z = npad_z

        if pad_rate_x is None:
            pad_rate_x = self.pad_rate_x
        self.pad_rate_x = pad_rate_x

        if pad_rate_z is None:
            pad_rate_z = self.pad_rate_z
        self.pad_rate_z = pad_rate_z

        if ncell_per_dipole is None:
            ncell_per_dipole = self.ncell_per_dipole
        self.ncell_per_dipole = ncell_per_dipole

        # 2D or 3D mesh
        if dimension not in [2, 3]:
            raise NotImplementedError(
                "Set mesh has not been implemented for a 1D system"
            )

        if dimension == 2:
            z_ind = 1
        else:
            z_ind = 2
        a = abs(np.diff(np.sort(self.electrode_locations[:, 0]))).min()
        lineLength = abs(
            self.electrode_locations[:, 0].max() - self.electrode_locations[:, 0].min()
        )
        dx_ideal = a / ncell_per_dipole
        if dx is None:
            dx = dx_ideal
            print(
                "dx is set to {} m (samllest electrode spacing ({}) / {})".format(
                    dx, a, ncell_per_dipole
                )
            )
        if dz is None:
            dz = dx * 0.5
            print("dz ({} m) is set to dx ({} m) / {}".format(dz, dx, 2))
        if dimension == 3:
            if dy is None:
                print("dy is set equal to dx")
                dy = dx
            self.dy = dy

            if npad_y is None:
                npad_y = self.npad_y
            self.npad_y = npad_y

            if pad_rate_y is None:
                pad_rate_y = self.pad_rate_y
            self.pad_rate_y = pad_rate_y

        x0 = self.electrode_locations[:, 0].min()
        if topo is None:
            # For 2D mesh
            if dimension == 2:
                # sort by x
                row_idx = np.lexsort((self.electrode_locations[:, 0],))
            # For 3D mesh
            else:
                # sort by x, then by y
                row_idx = np.lexsort(
                    (self.electrode_locations[:, 1], self.electrode_locations[:, 0])
                )
            locs = self.electrode_locations[row_idx, :]
        else:
            # For 2D mesh
            if dimension == 2:
                mask = np.isin(self.electrode_locations[:, 0], topo[:, 0])
                if np.any(mask):
                    warnings.warn(
                        "Because the x coordinates of some topo and electrodes are the same,"
                        " we excluded electrodes with the same coordinates.",
                        RuntimeWarning,
                    )
                locs_tmp = np.vstack((topo, self.electrode_locations[~mask, :]))
                row_idx = np.lexsort((locs_tmp[:, 0],))
            else:
                dtype = [("x", np.float64), ("y", np.float64)]
                mask = np.isin(
                    self.electrode_locations[:, [0, 1]].copy().view(dtype),
                    topo[:, [0, 1]].copy().view(dtype),
                ).flatten()
                if np.any(mask):
                    warnings.warn(
                        "Because the x and y coordinates of some topo and electrodes are the same,"
                        " we excluded electrodes with the same coordinates.",
                        RuntimeWarning,
                    )
                locs_tmp = np.vstack((topo, self.electrode_locations[~mask, :]))
                row_idx = np.lexsort((locs_tmp[:, 1], locs_tmp[:, 0]))
            locs = locs_tmp[row_idx, :]

        if dx > dx_ideal:
            # warnings.warn(
            #     "Input dx ({}) is greater than expected \n We recommend using {:0.1e} m cells, that is, {} cells per {0.1e} m dipole length".format(dx, dx_ideal, ncell_per_dipole, a)
            # )
            pass

        # Set mesh properties to class instance
        self.dx = dx
        self.dz = dz

        zmax = locs[:, z_ind].max()
        zmin = locs[:, z_ind].min()

        # 3 cells each for buffer
        corexlength = lineLength + dx * 6
        if corezlength is None:
            dz_topo = locs[:, 1].max() - locs[:, 1].min()
            corezlength = self.grids[:, z_ind].max() + dz_topo
            self.corezlength = corezlength

        if mesh_type == "TensorMesh":
            ncx = np.round(corexlength / dx)
            ncz = np.round(corezlength / dz)
            hx = [(dx, npad_x, -pad_rate_x), (dx, ncx), (dx, npad_x, pad_rate_x)]
            hz = [(dz, npad_z, -pad_rate_z), (dz, ncz)]
            x0_mesh = -(
                (dx * pad_rate_x ** (np.arange(npad_x) + 1)).sum() + dx * 3 - x0
            )
            z0_mesh = (
                -((dz * pad_rate_z ** (np.arange(npad_z) + 1)).sum() + dz * ncz) + zmax
            )

            # For 2D mesh
            if dimension == 2:
                h = [hx, hz]
                x0_for_mesh = [x0_mesh, z0_mesh]
                self.xyzlim = np.vstack(
                    (np.r_[x0, x0 + lineLength], np.r_[zmax - corezlength, zmax])
                )
                fill_value = "extrapolate"

            # For 3D mesh
            else:

                ylocs = np.unique(self.electrode_locations[:, 1])
                ymin, ymax = ylocs.min(), ylocs.max()
                # 3 cells each for buffer in y-direction
                coreylength = ymax - ymin + dy * 6
                ncy = np.round(coreylength / dy)
                hy = [(dy, npad_y, -pad_rate_y), (dy, ncy), (dy, npad_y, pad_rate_y)]
                y0 = ylocs.min() - dy / 2.0
                y0_mesh = -(
                    (dy * pad_rate_y ** (np.arange(npad_y) + 1)).sum() + dy * 3 - y0
                )

                h = [hx, hy, hz]
                x0_for_mesh = [x0_mesh, y0_mesh, z0_mesh]
                self.xyzlim = np.vstack(
                    (
                        np.r_[x0, x0 + lineLength],
                        np.r_[ymin - dy * 3, ymax + dy * 3],
                        np.r_[zmax - corezlength, zmax],
                    )
                )
            mesh = TensorMesh(h, x0=x0_for_mesh)

        elif mesh_type == "TREE":
            # Quadtree mesh
            if dimension == 2:

                pad_length_x = np.sum(meshTensor([(dx, npad_x, pad_rate_x)]))
                pad_length_z = np.sum(meshTensor([(dz, npad_z, pad_rate_z)]))

                dom_width_x = lineLength + 2 * pad_length_x  # domain width x
                dom_width_z = corezlength + pad_length_z  # domain width z

                nbcx = 2 ** int(
                    np.ceil(np.log(dom_width_x / dx) / np.log(2.0))
                )  # num. base cells x
                nbcz = 2 ** int(
                    np.ceil(np.log(dom_width_z / dz) / np.log(2.0))
                )  # num. base cells z

                length = 0.0
                dz_tmp = dz
                octree_levels = []
                while length < corezlength:
                    length += 5 * dz_tmp
                    octree_levels.append(5)
                    dz_tmp *= 2

                # Define the base mesh
                hx = [(dx, nbcx)]
                hz = [(dz, nbcz)]

                mesh_width = np.sum(meshTensor(hx))
                mesh_height = np.sum(meshTensor(hz))

                array_midpoint = 0.5 * (
                    self.electrode_locations[:, 0].min()
                    + self.electrode_locations[:, 0].max()
                )
                mesh = TreeMesh(
                    [hx, hz], x0=[array_midpoint - mesh_width / 2, zmax - mesh_height]
                )
                # mesh = TreeMesh([hx, hz], x0='CN')

                # Mesh refinement based on topography
                mesh = refine_tree_xyz(
                    mesh,
                    self.electrode_locations,
                    octree_levels=octree_levels,
                    method="radial",
                    finalize=False,
                )
                mesh.finalize()

                self.xyzlim = np.vstack(
                    (
                        np.r_[
                            self.electrode_locations[:, 0].min(),
                            self.electrode_locations[:, 0].max(),
                        ],
                        np.r_[zmax - corezlength, zmax],
                    )
                )

            # Octree mesh
            elif dimension == 3:
                raise NotImplementedError(
                    "set_mesh has not implemented 3D TreeMesh (yet)"
                )

        else:
            raise NotImplementedError(
                "set_mesh currently generates TensorMesh or TreeMesh"
            )

        actind = surface2ind_topo(mesh, locs, method=method, fill_value=np.nan)

        return mesh, actind

    def plotPseudoSection(
        self,
        data_type="apparent_resistivity",
        data=None,
        dataloc=True,
        aspect_ratio=2,
        scale="log",
        cmap="viridis",
        ncontour=10,
        ax=None,
        figname=None,
        clim=None,
        label=None,
        iline=0,
        orientation="vertical",
    ):
        """
            Plot 2D pseudo-section for DC-IP data
        """
        matplotlib.rcParams["font.size"] = 12

        if ax is None:
            fig = plt.figure(figsize=(10, 5))
            ax = plt.subplot(111)

        if self.dimension == 2:
            inds = np.ones(self.n_data, dtype=bool)
            grids = self.grids.copy()
        elif self.dimension == 3:
            inds = self.line_inds == iline
            grids = self.grids[inds, :][:, [0, 2]]
        else:
            raise NotImplementedError()

        if data_type == "apparent_resistivity":
            if data is None:
                val = self.apparent_resistivity[inds]
            else:
                val = data.copy()[inds]
            label_tmp = "Apparent Res. ($\Omega$m)"
        elif data_type == "volt":
            if data is None:
                val = self.voltages[inds]
            else:
                val = data.copy()[inds]
            label_tmp = "Voltage (V)"
        elif data_type == "apparent_conductivity":
            if data is None:
                val = self.apparent_conductivity[inds]
            else:
                val = data.copy()[inds]
            label_tmp = "Apparent Cond. (S/m)"
        elif data_type == "apparent_chargeability":
            if data is not None:
                val = data.copy()[inds]
            else:
                val = self.apparent_chargeability.copy()[inds] * 1e3
            label_tmp = "Apparent Charg. (mV/V)"
        elif data_type == "volt_ip":
            if data is not None:
                val = data.copy()[inds]
            else:
                val = self.voltages_ip.copy()[inds] * 1e3
            label_tmp = "Secondary voltage. (mV)"
        else:
            print(data_type)
            raise NotImplementedError()

        if label is None:
            label = label_tmp

        out = plot2Ddata(
            grids,
            val,
            contourOpts={"cmap": cmap},
            ax=ax,
            dataloc=dataloc,
            scale=scale,
            ncontour=ncontour,
            clim=clim,
        )
        ax.invert_yaxis()
        ax.set_xlabel("x (m)")
        ax.set_yticklabels([])
        ax.set_ylabel("n-spacing")
        if orientation == "vertical":
            frac = 0.01
        elif orientation == "horizontal":
            frac = 0.03
        else:
            raise ValueError(
                "Orientation must be either vertical or horizontal, not {}".format(
                    orientation
                )
            )
        cb = plt.colorbar(
            out[0], format="%.1e", ax=ax, orientation=orientation, fraction=frac
        )
        cb.set_label(label)
        cb.set_ticks(out[0].levels)
        ax.set_aspect(aspect_ratio)
        plt.tight_layout()
        if figname is not None:
            fig.savefig(figname, dpi=200)

    def read_ubc_dc2d_obs_file(self, filename, input_type="simple", toponame=None):
        obsfile = np.genfromtxt(filename, delimiter=" \n", dtype=np.str, comments="!")
        if input_type == "general":
            topo = None
            n_src = 0
            n_rxs = []
            src_info = []
            abmn = []
            for obs in obsfile:
                temp = np.fromstring(obs, dtype=float, sep=" ").T
                if len(temp) == 5:
                    n_src += 1
                    src_info = temp[:4]
                    n_rxs.append(int(temp[-1]))
                else:
                    abmn.append(np.r_[src_info, temp])

            abmn = np.vstack(abmn)
            a = np.c_[abmn[:, 0], -abmn[:, 1]]
            b = np.c_[abmn[:, 2], -abmn[:, 3]]
            m = np.c_[abmn[:, 4], -abmn[:, 5]]
            n = np.c_[abmn[:, 6], -abmn[:, 7]]
            voltage = abmn[:, 8]
            standard_deviation = abmn[:, 9]

        elif input_type == "simple":
            if toponame is not None:
                tmp_topo = np.loadtxt(toponame)
                n_topo = tmp_topo[0, 0]
                z_ref = tmp_topo[0, 1]
                topo = tmp_topo[1:, :]
                if topo.shape[0] != n_topo:
                    print(
                        ">> # of points for the topography is not {0}, but {0}".format(
                            n_topo, topo.shape[0]
                        )
                    )
            tmp = np.loadtxt(filename, comments="!").astype(float)
            e = np.zeros(tmp.shape[0], dtype=float)
            a = np.c_[tmp[:, 0], e]
            b = np.c_[tmp[:, 1], e]
            m = np.c_[tmp[:, 2], e]
            n = np.c_[tmp[:, 3], e]
            voltage = tmp[:, 4]
            standard_deviation = tmp[:, 5]

        if np.all(a == b):
            if np.all(m == n):
                survey_type = "pole-pole"
            else:
                survey_type = "pole-dipole"
        else:
            if np.all(m == n):
                survey_type = "dipole-pole"
            else:
                survey_type = "dipole-dipole"

        survey = self.from_ambn_locations_to_survey(
            a, b, m, n, survey_type=survey_type, data_dc=voltage
        )
        survey.dobs = voltage[self.sort_inds]
        survey.std = standard_deviation[self.sort_inds]
        survey.topo = topo
        return survey

    def write_to_csv(self, fname, dobs, standard_deviation=None, **kwargs):
        uncert = kwargs.pop("uncertainty", None)
        if uncert is not None:
            warnings.warn(
                "The uncertainty option has been deprecated and will be removed"
                " in SimPEG 0.15.0. Please use standard_deviation.",
                DeprecationWarning,
            )
            standard_deviation = uncert

        if standard_deviation is None:
            standard_deviation = np.ones(dobs.size) * np.nan
        data = np.c_[
            self.a_locations,
            self.b_locations,
            self.m_locations,
            self.n_locations,
            dobs,
            standard_deviation,
        ]
        df = pd.DataFrame(
            data=data,
            columns=[
                "Ax",
                "Az",
                "Bx",
                "Bz",
                "Mx",
                "Mz",
                "Nx",
                "Nz",
                "Voltage",
                "Uncertainty",
            ],
        )
        df.to_csv(fname)

    def read_dc_data_csv(self, fname, dim=2):
        df = pd.read_csv(fname)
        if dim == 2:
            a_locations = df[["Ax", "Az"]].values
            b_locations = df[["Bx", "Bz"]].values
            m_locations = df[["Mx", "Mz"]].values
            n_locations = df[["Nx", "Nz"]].values
            dobs = df["Voltage"].values
            standard_deviation = df["Uncertainty"].values

            if np.all(a_locations == b_locations):
                src_type = "pole-"
            else:
                src_type = "dipole-"

            if np.all(m_locations == n_locations):
                rx_type = "pole"
            else:
                rx_type = "dipole"
            survey_type = src_type + rx_type
            survey = self.from_ambn_locations_to_survey(
                a_locations,
                b_locations,
                m_locations,
                n_locations,
                survey_type,
                data_dc=dobs,
                data_dc_type="volt",
            )
            survey.std = standard_deviation[self.sort_inds]
            survey.dobs = dobs[self.sort_inds]
        else:
            raise NotImplementedError()
        return survey

    def read_topo_csv(self, fname, dim=2):
        if dim == 2:
            df = pd.read_csv(fname)
            topo = df[["X", "Z"]].values
        return topo
