import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import properties
import warnings

import SimPEG
from SimPEG import Utils, Mesh
from . import SrcDC as Src
from . import RxDC as Rx
from .SurveyDC import Survey_ky, Survey

warnings.warn("code under construction - API might change in the future")


class IO(properties.HasProperties):
    """

    """

    # Survey
    survey_layout = properties.StringChoice(
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

    dimension = properties.Integer(
        "Dimension of electrode locations",
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
    data_dc_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=[
           "volt", "apparent_resistivity", "apparent_conductivity",
        ]
    )

    data_dc = properties.Array(
        "Measured DC data",
        shape=('*',),
        dtype=float  # data are floats
    )

    data_ip_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=[
           "volt", "apparent_chargeability",
        ]
    )

    data_ip = properties.Array(
        "Measured IP data",
        shape=('*',),
        dtype=float  # data are floats
    )

    data_sip_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=[
           "volt", "apparent_chargeability",
        ]
    )

    data_sip = properties.Array(
        "Measured Spectral IP data",
        shape=('*', '*'),
        dtype=float  # data are floats
    )

    times_ip = properties.Array(
        "Time channels of measured Spectral IP voltages (s)",
        required=True,
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

    line_inds = properties.Array(
        "Line indices",
        required=True,
        shape=('*',),
        dtype=int  # data are floats
    )
    sort_inds = properties.Array(
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
        "Expansion rate of padding cells in  x-direction",
        required=True,
        default=1.3
    )

    pad_rate_y = properties.Float(
        "Expansion rate of padding cells in  y-direction",
        required=True,
        default=1.3
    )

    pad_rate_z = properties.Float(
        "Expansion rate of padding cells in  z-direction",
        required=True,
        default=1.3
    )

    ncell_per_dipole = properties.Integer(
        "The number of cells between dipole electrodes",
        required=True,
        default=4
    )

    # For synthetic surveys
    x0 = None
    lineLength = None
    a = None
    n_spacing = None
    n_data = None

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
            return 1./self.data_dc
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
            return 1./self.data_dc
        elif self.data_dc_type.lower() == "volt":
            return 1./self.data_dc * self.G

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
                raise Exception(
                    "DC voltages must be set to compute IP voltages"
                    )
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
                raise Exception(
                    "DC voltages must be set to compute IP voltages"
                    )
            return Utils.sdiag(self.voltages) * self.data_sip
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
            return Utils.sdiag(1./self.voltages) * self.data_sip
        else:
            raise NotImplementedError()

    def geometric_factor(self, survey):
        """
        Compute geometric factor, G, using locational informaition
        in survey object
        """
        geometric_factor = SimPEG.EM.Static.Utils.StaticUtils.geometric_factor
        G = geometric_factor(
            survey, survey_type=self.survey_type, space_type=self.space_type
            )
        return G

    def from_ambn_locations_to_survey(
        self, a_locations, b_locations, m_locations, n_locations,
        survey_type=None, data_dc=None, data_ip=None, data_sip=None,
        data_dc_type="volt", data_ip_type="volt", data_sip_type="volt",
        fname=None, dimension=2, line_inds=None,
        times_ip=None
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

        uniqSrc = Utils.uniqueRows(np.c_[self.a_locations, self.b_locations])
        uniqElec = SimPEG.Utils.uniqueRows(
            np.vstack(
                (
                    self.a_locations, self.b_locations,
                    self.m_locations, self.n_locations
                )
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
                if dimension == 2:
                    if survey_type in ['dipole-dipole', 'pole-dipole']:
                        rx = Rx.Dipole_ky(locsM, locsN)
                    elif survey_type in ['dipole-pole', 'pole-pole']:
                        rx = Rx.Pole_ky(locsM)
                elif dimension == 3:
                    if survey_type in ['dipole-dipole', 'pole-dipole']:
                        rx = Rx.Dipole(locsM, locsN)
                    elif survey_type in ['dipole-pole', 'pole-pole']:
                        rx = Rx.Pole(locsM)
                else:
                    raise NotImplementedError()

                if dimension == 2:
                    locA = uniqSrc[0][iSrc, :2]
                    locB = uniqSrc[0][iSrc, 2:]
                elif dimension == 3:
                    locA = uniqSrc[0][iSrc, :3]
                    locB = uniqSrc[0][iSrc, 3:]

                if survey_type in ['dipole-dipole', 'dipole-pole']:
                    src = Src.Dipole([rx], locA, locB)
                elif survey_type in ['pole-dipole', 'pole-pole']:
                    src = Src.Pole([rx], locA)

                srcLists.append(src)

            self.sort_inds = np.hstack(sort_inds)

            if dimension == 2:
                survey = Survey_ky(srcLists)
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

            midABx = (self.a_locations[:, 0] + self.b_locations[:, 0])*0.5
            midMNx = (self.m_locations[:, 0] + self.n_locations[:, 0])*0.5

            if dimension == 2:
                z = abs(midABx-midMNx)*1./3.
                x = (midABx+midMNx)*0.5
                self.grids = np.c_[x, z]

            elif dimension == 3:
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

    def set_mesh(self, topo=None,
                dx=None, dy=None, dz=None,
                n_spacing=None, corezlength=None,
                npad_x=7, npad_y=7, npad_z=7,
                pad_rate_x=1.3, pad_rate_y=1.3, pad_rate_z=1.3,
                ncell_per_dipole=4, mesh_type='TensorMesh',
                dimension=2,
                method='linear'
                ):
        """
        Set up a mesh for a given DC survey
        """
        if mesh_type == 'TreeMesh':
            raise NotImplementedError()

        # 2D or 3D mesh
        if dimension in [2, 3]:
            if dimension == 2:
                z_ind = 1
            else:
                z_ind = 2
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

            if dx > dx_ideal:
                # warnings.warn(
                #     "Input dx ({}) is greater than expected \n We recommend using {:0.1e} m cells, that is, {} cells per {0.1e} m dipole length".format(dx, dx_ideal, ncell_per_dipole, a)
                # )
                pass
            self.dx = dx
            self.dz = dz
            self.npad_x = npad_x
            self.npad_z = npad_z
            self.pad_rate_x = pad_rate_x
            self.pad_rate_z = pad_rate_z
            self.ncell_per_dipole = ncell_per_dipole
            zmax = locs[:, z_ind].max()
            zmin = locs[:, z_ind].min()

            # 3 cells each for buffer
            corexlength = lineLength + dx * 6
            if corezlength is None:
                corezlength = self.grids[:, z_ind].max()

            ncx = np.round(corexlength/dx)
            ncz = np.round(corezlength/dz)
            hx = [
                (dx, npad_x, -pad_rate_x), (dx, ncx), (dx, npad_x, pad_rate_x)
            ]
            hz = [(dz, npad_z, -pad_rate_z), (dz, ncz)]
            x0_mesh = -(
                (dx * pad_rate_x ** (np.arange(npad_x)+1)).sum() + dx * 3 - x0
            )
            z0_mesh = -((dz * pad_rate_z ** (np.arange(npad_z)+1)).sum() + dz * ncz) + zmax

            # For 2D mesh
            if dimension == 2:
                h = [hx, hz]
                x0_for_mesh = [x0_mesh, z0_mesh]
                self.xyzlim = np.vstack((
                    np.r_[x0, x0+lineLength],
                    np.r_[zmax-corezlength, zmax]
                ))

            # For 3D mesh
            else:
                if dy is None:
                    raise Exception("You must input dy (m)")

                self.dy = dy
                self.npad_y = npad_y
                self.pad_rate_y = pad_rate_y

                ylocs = np.unique(self.electrode_locations[:, 1])
                ymin, ymax = ylocs.min(), ylocs.max()
                # 3 cells each for buffer in y-direction
                coreylength = ymax-ymin+dy*6
                ncy = np.round(coreylength/dy)
                hy = [
                    (dy, npad_y, -pad_rate_y),
                    (dy, ncy),
                    (dy, npad_y, pad_rate_y)
                ]
                y0 = ylocs.min()-dy/2.
                y0_mesh = -(
                    (dy * pad_rate_y ** (np.arange(npad_y)+1)).sum()
                    + dy*3 - y0
                )

                h = [hx, hy, hz]
                x0_for_mesh = [x0_mesh, y0_mesh, z0_mesh]
                self.xyzlim = np.vstack((
                    np.r_[x0, x0+lineLength],
                    np.r_[ymin, ymax],
                    np.r_[zmax-corezlength, zmax]
                ))
            mesh = Mesh.TensorMesh(h, x0=x0_for_mesh)
            actind = Utils.surface2ind_topo(mesh, locs, method=method)
        else:
            raise NotImplementedError()

        return mesh, actind

    def plotPseudoSection(
        self, data_type="apparent_resistivity",
        data=None,
        dataloc=True, aspect_ratio=2,
        scale="log",
        cmap="viridis", ncontour=10, ax=None,
        figname=None, clim=None, label=None,
        iline=0,
    ):
        """
            Plot 2D pseudo-section for DC-IP data
        """
        matplotlib.rcParams['font.size'] = 12

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
            label = "Apparent Res. ($\Omega$m)"
        elif data_type == "volt":
            if data is None:
                val = self.voltages[inds]
            else:
                val = data.copy()[inds]
            label = "Voltage (V)"
        elif data_type == "apparent_conductivity":
            if data is None:
                val = self.apparent_conductivity[inds]
            else:
                val = data.copy()[inds]
            label = "Apparent Cond. (S/m)"
        elif data_type == "apparent_chargeability":
            if data is not None:
                val = data.copy()[inds]
            else:
                val = self.apparent_chargeability.copy()[inds] * 1e3
            label = "Apparent Charg. (mV/V)"
        elif data_type == "volt_ip":
            if data is not None:
                val = data.copy()[inds]
            else:
                val = self.voltages_ip.copy()[inds] * 1e3
            label = "Secondary voltage. (mV)"
        else:
            print (data_type)
            raise NotImplementedError()
        if scale == "log":
            fmt = "10$^{%.1f}$"
        elif scale == "linear":
            fmt = "%.1e"
        else:
            raise NotImplementedError()

        out = Utils.plot2Ddata(
            grids, val,
            contourOpts={'cmap': cmap},
            ax=ax,
            dataloc=dataloc,
            scale=scale,
            ncontour=ncontour,
            clim=clim
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
