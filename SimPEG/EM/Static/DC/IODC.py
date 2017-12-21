import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import properties
import warnings

import SimPEG
from SimPEG import Utils, Mesh
from . import SrcDC as Src
from . import RxDC as Rx
from .SurveyDC import Survey_ky


class IO(properties.HasProperties):
    """

    """

    # Survey
    survey_geometry = properties.StringChoice(
        "Survey geometry of DC surveys",
        default="SURFACE",
        choices=["SURFACE", "BOREHOLE", "GENERAL"]
    )

    survey_type = properties.StringChoice(
        "DC-IP Survey type",
        default="dipole-dipole",
        choices=[
            "dipole-dipole", "pole-dipole",
            "dipole-pole", "pole-pole"
        ]
    )

    dim = properties.Integer(
        "Dimension of locations",
        default=2,
        required=True
    )

    a_locations = properties.Array(
        "locations of the positive (+) current electrodes",
        required=True,
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    b_locations = properties.Array(
        "locations of the negative (-) current electrodes",
        required=True,
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    m_locations = properties.Array(
        "locations of the positive (+) potential electrodes",
        required=True,
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    n_locations = properties.Array(
        "locations of the negative (-) potential electrodes",
        required=True,
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    electrode_locations = properties.Array(
        "unique locations of a, b, m, n electrodes",
        required=True,
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    # Data
    data_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=[
           "volt", "appResistivity", "appConductivity",
           "voltIP", "appChargeability"
        ]
    )

    dobs = properties.Array(
        "Measured data",
        shape=('*',),
        dtype=float  # data are floats
    )

    volt = properties.Array(
        "Measured DC voltages (V)",
        shape=('*',),
        dtype=float  # data are floats
    )

    appResistivity = properties.Array(
        "Measured apparent resistivity (Ohm-m)",
        shape=('*',),
        dtype=float  # data are floats
    )

    appConductivity = properties.Array(
        "Measured apparent conductivity (S/m)",
        shape=('*',),
        dtype=float  # data are floats
    )

    G = properties.Array(
        "Geometric factor of DC-IP survey",
        shape=('*',),
        dtype=float  # data are floats
    )

    grids = properties.Array(
        "Spatial grids for plotting pseudo-section",
        shape=('*', '*'),
        dtype=float  # data are floats
    )

    space_type = properties.StringChoice(
        "Assumption to compute apparent resistivity",
        default="half-space",
        choices=[
            "half-space", "whole-space"
        ]
    )

    voltIP = properties.Array(
        "Measured IP voltages (V)",
        shape=('*',),
        dtype=float  # data are floats
    )

    appChargeability = properties.Array(
        "Measured apparent Chargeability (V/V)",
        shape=('*',),
        dtype=float  # data are floats
    )

    timesIP = properties.Array(
        "Time channels of measured IP voltages (s)",
        required=True,
        shape=('*',),
        dtype=float  # data are floats
    )

    sortinds = properties.Array(
        "Sorting indices from ABMN",
        required=True,
        shape=('*',),
        dtype=int  # data are floats
    )

    # Related to Physics and Discretization
    mesh = properties.Instance(
        "Mesh for discretization", Mesh.BaseMesh, required=True
    )

    dx = properties.Float(
        "Length of corecell in x-direction", required=True,
    )
    dy = properties.Float(
        "Length of corecell in y-direction", required=True
    )
    dy = properties.Float(
        "Length of corecell in z-direction", required=True
    )

    npad_x = properties.Integer(
        "The number of padding cells x-direction",
        required=True,
        default=5
    )

    npad_y = properties.Integer(
        "The number of padding cells y-direction",
        required=True,
        default=5
    )

    npad_z = properties.Integer(
        "The number of padding cells z-direction",
        required=True,
        default=5
    )

    pad_rate_x = properties.Float(
        "The number of padding cells x-direction",
        required=True,
        default=5
    )

    pad_rate_y = properties.Float(
        "The number of padding cells y-direction",
        required=True,
        default=5
    )

    pad_rate_z = properties.Float(
        "The number of padding cells z-direction",
        required=True,
        default=5
    )

    ncell_per_dipole = properties.Integer(
        "The number of cells between dipole electrodes",
        required=True,
        default=5
    )

    # For synthetic surveys
    x0 = None
    lineLength = None
    a = None
    n_spacing = None

    def from_ambn_locations_to_survey(
        self, a_locations, b_locations, m_locations, n_locations,
        survey_type=None, dobs=None,
        data_type="volt", fname=None, dim=2
    ):
        """
        read A, B, M, N electrode location and data (V or appResistivity)
        """
        geometric_factor = SimPEG.EM.Static.Utils.StaticUtils.geometric_factor

        self.a_locations = a_locations.copy()
        self.b_locations = b_locations.copy()
        self.m_locations = m_locations.copy()
        self.n_locations = n_locations.copy()
        self.survey_type = survey_type
        self.data_type = data_type
        self.dim = dim

        uniqSrc = Utils.uniqueRows(np.c_[self.a_locations, self.b_locations])
        uniqElec = SimPEG.Utils.uniqueRows(
            np.vstack((self.a_locations, self.b_locations, self.m_locations, self.n_locations))
            )
        self.electrode_locations = uniqElec[0]
        nSrc = uniqSrc[0].shape[0]
        ndata = self.a_locations.shape[0]

        if self.survey_geometry == "SURFACE":
            # 2D locations
            srcLists = []
            sortinds = []
            for iSrc in range (nSrc):
                inds = uniqSrc[2] == iSrc
                sortinds.append(np.arange(ndata)[inds])

                locsM = self.m_locations[inds, :]
                locsN = self.n_locations[inds, :]

                if (survey_type == 'dipole-dipole') or (survey_type == 'pole-dipole'):
                    rx = Rx.Dipole_ky(locsM, locsN)
                elif (survey_type == 'dipole-pole') or (survey_type == 'pole-pole'):
                    rx = Rx.Pole_ky(locsM)

                if dim == 2:
                    locA = uniqSrc[0][iSrc, :2]
                    locB = uniqSrc[0][iSrc, 2:]
                elif dim == 3:
                    locA = uniqSrc[0][iSrc, :3]
                    locB = uniqSrc[0][iSrc, 3:]

                if (survey_type == 'dipole-dipole') or (survey_type == 'dipole-pole'):
                    src = Src.Dipole([rx], locA, locB)
                elif (survey_type == 'pole-dipole') or (survey_type == 'pole-pole'):
                    src = Src.Pole([rx], locA)

                srcLists.append(src)

            self.sortinds = np.hstack(sortinds)
            survey = Survey_ky(srcLists)
            self.a_locations = self.a_locations[self.sortinds, :]
            self.b_locations = self.b_locations[self.sortinds, :]
            self.m_locations = self.m_locations[self.sortinds, :]
            self.n_locations = self.n_locations[self.sortinds, :]
            self.G = geometric_factor(
                survey, survey_type=self.survey_type,
                space_type=self.space_type
                )
            if dobs is None:
                self.dobs = 100.*self.G
            else:
                self.dobs = dobs[self.sortinds]

            if self.data_type == "volt":
                self.volt = self.dobs.copy()
                self.appResistivity = self.volt / self.G
            elif self.data_type == "appResistivity":
                self.appResistivity = self.dobs.copy()
                self.volt = self.appResistivity * self.G
                self.dobs = self.volt.copy()
            else:
                raise NotImplementedError()

            # Here we ignore ... z-locations

            midABx = (self.a_locations[:, 0] + self.b_locations[:, 0])*0.5
            midMNx = (self.m_locations[:, 0] + self.n_locations[:, 0])*0.5

            if dim == 2:
                z = abs(midABx-midMNx)*1./3.
                x = (midABx+midMNx)*0.5
                self.grids = np.c_[x, z]

            elif dim == 3:
                midABy = (self.a_locations[:, 1] + self.b_locations[:, 1])*0.5
                midMNy = (self.m_locations[:, 1] + self.n_locations[:, 1])*0.5
                z = np.sqrt((midABx-midMNx)**2 + (midABy-midMNy)**2) * 1./3.
                x = (midABx+midMNx)*0.5
                y = (midABy+midMNy)*0.5
                self.grids = np.c_[x, y, z]
            else:
                raise Exception()
        else:
            raise NotImplementedError()
        return survey

    def set_dc_data(self, dc_data, data_type='volt'):
        if data_type == 'volt':
            self.volt = dc_data.copy()
            self.appResistivity = self.volt / self.G
            self.appConductivity = 1./self.appResistivity
        elif data_type == 'appResistivity':
            self.appResistivity = dc_data.copy()
            self.volt = self.appResistivity * self.G
            self.appConductivity = 1./self.appResistivity
        elif data_type == 'appConductivity':
            self.appConductivity = dc_data.copy()
            self.appResistivity = 1./self.appConductivity
            self.volt = self.appResistivity * self.G
        else:
            raise NotImplementedError()

    def set_ip_data(self, ip_data, data_type='appChargeability', dc_data=None):
        if data_type == 'voltIP':
            self.voltIP = ip_data.copy()
            self.appChargeability = self.voltIP / self.volt
        elif data_type == 'appChargeability':
            self.appChargeability = ip_data.copy()
            self.voltIP = self.appChargeability * self.volt
        else:
            raise NotImplementedError()

    def set_mesh(self, topo=None,
                dx=None, dz=None,
                n_spacing=None, corezlength=None,
                npad_x=7, npad_z=7,
                pad_rate_x=1.3, pad_rate_y=1.3, pad_rate_z=1.3,
                ncell_per_dipole=4, mesh_type='TensorMesh',
                dim=2
                ):
        if mesh_type == 'TreeMesh':
            raise NotImplementedError()

        electrode_separations = SimPEG.EM.Static.Utils.StaticUtils.electrode_separations

        if dim == 2:
            a = abs(np.diff(np.sort(self.electrode_locations[:, 0]))).min()
            lineLength = abs(
                self.electrode_locations[:, 0].max() -
                self.electrode_locations[:, 0].min()
            )
            dx_ideal = a/ncell_per_dipole
            if dx is None:
                dx = dx_ideal
                warnings.warn(
                    "dx is set to {} m (samllest electrode spacing ({}) / {})".format(dx, a, ncell_per_dipole)
                )
            if dz is None:
                dz = dx*0.5
                warnings.warn(
                    "dz ({} m) is set to dx ({} m) / {}".format(dz, dx, 2)
                )
            x0 = self.electrode_locations[:, 0].min()
            if topo is None:
                locs = self.electrode_locations
            else:
                locs = np.vstack((topo, self.electrode_locations))

            zmax = locs[:, 1].max()
            zmin = locs[:, 1].min()
            if dx > dx_ideal:
                warnings.warn(
                    "Input dx ({}) is greater than expected \n We recommend using {:0.1e} m cells, that is, {:i} cells per {0.1e} m dipole length".format(dx, dx_ideal, ncell_per_dipole, a)
                )

            self.dx = dx
            self.dz = dz
            self.npad_x = npad_x
            self.npad_z = npad_z
            self.pad_rate_x = pad_rate_x
            self.pad_rate_z = pad_rate_z
            self.ncell_per_dipole = ncell_per_dipole
            # 3 cells each for buffer
            corexlength = lineLength + dx * 6
            if corezlength is None:
                corezlength = self.grids[:, 1].max() + zmax - zmin

            ncx = np.floor(corexlength/dx)
            ncz = np.floor(corezlength/dz)
            hx = [(dx, npad_x, -pad_rate_x), (dx, ncx), (dx, npad_x, pad_rate_x)]
            hz = [(dz, npad_z, -pad_rate_z), (dz, ncz)]
            x0_mesh = -(
                (dx * 1.3 ** (np.arange(npad_x)+1)).sum() + dx * 3 - x0
            )
            z0_mesh = -((dz * 1.3 ** (np.arange(npad_z)+1)).sum() + dz * ncz) + zmax
            mesh = Mesh.TensorMesh([hx, hz], x0=[x0_mesh, z0_mesh])
            actind = Utils.surface2ind_topo(mesh, locs)
            print (mesh)
        elif dim == 3:
            raise NotImplementedError()

        else:
            raise NotImplementedError()

        return mesh, actind

    def plot_pseudosection(
        self, data_type="appResistivity",
        data=None,
        dataloc=True, aspect_ratio=2,
        scale="log",
        cmap="viridis", ncontour=10, ax=None, figname=None,
        label=None
    ):
        """
            Plot 2D pseudo-section for DC-IP data
        """
        matplotlib.rcParams['font.size'] = 12

        if self.dim == 2:
            if ax is None:
                fig = plt.figure(figsize=(10, 5))
                ax = plt.subplot(111)

            if data_type == "appResistivity":
                if data is not None:
                    val = data.copy()
                else:
                    val = self.appResistivity.copy()
                label = "Apparent Res. ($\Omega$m)"
            elif data_type == "volt":
                if data is not None:
                    val = data.copy()
                else:
                    val = self.volt.copy()
                label = "Voltage (V)"
            elif data_type == "appChargeability":
                if data is not None:
                    val = data.copy()
                else:
                    val = self.appChargeability.copy() * 1e3
                label = "Apparent Charg. (mV/V)"
            elif data_type == "voltIP":
                if data is not None:
                    val = data.copy()
                else:
                    val = self.voltIP.copy() * 1e3
                label = "Secondary voltage. (mV)"
            else:
                raise NotImplementedError()
            if scale == "log":
                fmt = "10$^{%.1f}$"
            elif scale == "linear":
                fmt = "%.1e"
            else:
                raise NotImplementedError()

            out = Utils.plot2Ddata(
                self.grids, val,
                contourOpts={'cmap': cmap},
                ax=ax,
                dataloc=dataloc,
                scale=scale,
                ncontour=ncontour
            )
            ax.invert_yaxis()
            ax.set_xlabel("x (m)")
            ax.set_yticklabels([])
            ax.set_ylabel("n-spacing")
            cb = plt.colorbar(out[0], fraction=0.01, format=fmt, ax=ax)
            cb.set_label(label)
            ax.set_aspect(aspect_ratio)
            plt.tight_layout()
            if figname is not None:
                fig.savefig(figname, dpi=200)
