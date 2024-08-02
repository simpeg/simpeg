import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings

from discretize import TensorMesh, TreeMesh
from discretize.base import BaseMesh
from discretize.utils import refine_tree_xyz, unpack_widths, active_from_xyz

from ....utils import (
    sdiag,
    unique_rows,
    plot2Ddata,
    validate_type,
    validate_integer,
    validate_string,
    validate_ndarray_with_shape,
    validate_float,
)
from ..utils import geometric_factor
from . import sources as Src
from . import receivers as Rx
from .survey import Survey


class IO:
    def __init__(
        self,
        survey_layout="SURFACE",
        survey_type="dipole-dipole",
        dimension=2,
        a_locations=None,
        b_locations=None,
        m_locations=None,
        n_location=None,
        electrode_locations=None,
        data_dc_type="volt",
        data_dc=None,
        data_ip_type="volt",
        data_ip=None,
        data_sip_type="volt",
        data_sip=None,
        times_ip=None,
        G=None,
        grids=None,
        space_type="half-space",
        line_inds=None,
        sort_inds=None,
        mesh=None,
        dx=None,
        dy=None,
        dz=None,
        npad_x=5,
        npad_y=5,
        npad_z=5,
        pad_rate_x=1.3,
        pad_rate_y=1.3,
        pad_rate_z=1.3,
        ncell_per_dipole=4,
        corezlength=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.survey_layout = survey_layout
        self.survey_type = survey_type
        self.dimension = dimension
        self.a_locations = a_locations
        self.b_locations = b_locations
        self.m_locations = m_locations
        self.n_location = n_location
        self.electrode_locations = electrode_locations
        self.data_dc_type = data_dc_type
        self.data_dc = data_dc
        self.data_ip_type = data_ip_type
        self.data_ip = data_ip
        self.data_sip_type = data_sip_type
        self.data_sip = data_sip
        self.times_ip = times_ip
        self.G = G
        self.grids = grids
        self.space_type = space_type
        self.line_inds = line_inds
        self.sort_inds = sort_inds
        self.mesh = mesh
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.npad_x = npad_x
        self.npad_y = npad_y
        self.npad_z = npad_z
        self.pad_rate_x = pad_rate_x
        self.pad_rate_y = pad_rate_y
        self.pad_rate_z = pad_rate_z
        self.ncell_per_dipole = ncell_per_dipole
        self.corezlength = corezlength
        warnings.warn(
            "code under construction - API might change in the future",
            stacklevel=2,
        )

    @property
    def survey_layout(self):
        """Survey geometry of DC surveys.

        Returns
        -------
        {"SURFACE", "BOREHOLE", "GENERAL"}
        """
        return self._survey_layout

    @survey_layout.setter
    def survey_layout(self, value):
        self._survey_layout = validate_string(
            "survey_layout", value, ("SURFACE", "BOREHOLE", "GENERAL")
        )

    @property
    def survey_type(self):
        """Survey geometry of DC surveys.

        Returns
        -------
        {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}
        """
        return self._survey_type

    @survey_type.setter
    def survey_type(self, value):
        self._survey_type = validate_string(
            "survey_type",
            value,
            ("dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"),
        )

    @property
    def dimension(self):
        """Dimension of electrode locations.

        Returns
        -------
        int
        """
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = validate_integer("dimension", value, min_val=2, max_val=3)

    @property
    def a_locations(self):
        """Locations of the positive (+) current electrodes.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._a_locations

    @a_locations.setter
    def a_locations(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("a_locations", value, shape=("*", "*"))
        self._a_locations = value

    @property
    def b_locations(self):
        """Locations of the negative (-) current electrodes.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._b_locations

    @b_locations.setter
    def b_locations(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("b_locations", value, shape=("*", "*"))
        self._b_locations = value

    @property
    def m_locations(self):
        """Locations of the positive (+) potential electrodes.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._m_locations

    @m_locations.setter
    def m_locations(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("m_locations", value, shape=("*", "*"))
        self._m_locations = value

    @property
    def n_locations(self):
        """Locations of the negative (-) potential electrodes.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._n_locations

    @n_locations.setter
    def n_locations(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("n_locations", value, shape=("*", "*"))
        self._n_locations = value

    @property
    def electrode_locations(self):
        """Unique locations of a, b, m, n electrodes.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._electrode_locations

    @electrode_locations.setter
    def electrode_locations(self, value):
        if value is not None:
            value = validate_ndarray_with_shape(
                "electrode_locations", value, shape=("*", "*")
            )
        self._electrode_locations = value

    # Data
    @property
    def data_dc_type(self):
        """Type of DC-IP survey.

        Returns
        -------
        {"volt", "apparent_resistivity", "apparent_conductivity"}
        """
        return self._data_dc_type

    @data_dc_type.setter
    def data_dc_type(self, value):
        self._data_dc_type = validate_string(
            "data_dc_type",
            value,
            ("volt", "apparent_resistivity", "apparent_conductivity"),
        )

    @property
    def data_dc(self):
        """Measured DC data.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._data_dc

    @data_dc.setter
    def data_dc(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("data_dc", value, shape=("*",))
        self._data_dc = value

    @property
    def data_ip_type(self):
        """Type of DC-IP survey.

        Returns
        -------
        {"volt", "apparent_chargeability"}
        """
        return self._data_ip_type

    @data_ip_type.setter
    def data_ip_type(self, value):
        self._data_ip_type = validate_string(
            "data_ip_type", value, ("volt", "apparent_chargeability")
        )

    @property
    def data_ip(self):
        """Measured IP data.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._data_ip

    @data_ip.setter
    def data_ip(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("data_ip", value, shape=("*",))
        self._data_ip = value

    @property
    def data_sip_type(self):
        """Type of DC-IP survey.

        Returns
        -------
        {"volt", "apparent_chargeability"}
        """
        return self._data_sip_type

    @data_sip_type.setter
    def data_sip_type(self, value):
        self._data_sip_type = validate_string(
            "data_sip_type", value, ("volt", "apparent_chargeability")
        )

    @property
    def data_sip(self):
        """Measured Spectral IP data.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._data_sip

    @data_sip.setter
    def data_sip(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("data_sip", value, shape=("*",))
        self._data_sip = value

    @property
    def times_ip(self):
        """Time channels of measured Spectral IP voltages (s).

        Returns
        -------
        numpy.ndarray of float
        """
        return self._data_sip

    @times_ip.setter
    def times_ip(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("times_ip", value, shape=("*",))
        self._times_ip = value

    @property
    def G(self):
        """Geometric factor of DC-IP survey.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._G

    @G.setter
    def G(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("G", value, shape=("*",))
        self._G = value

    @property
    def grids(self):
        """Geometric factor of DC-IP survey.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._grids

    @grids.setter
    def grids(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("grids", value, shape=("*", "*"))
        self._grids = value

    @property
    def space_type(self):
        """Type of DC-IP survey.

        Returns
        -------
        {"half-space", "whole-space"}
        """
        return self._space_type

    @space_type.setter
    def space_type(self, value):
        self._space_type = validate_string(
            "space_type", value, ("half-space", "whole-space")
        )

    @property
    def line_inds(self):
        """Line indices.

        Returns
        -------
        numpy.ndarray of int
        """
        return self._line_inds

    @line_inds.setter
    def line_inds(self, value):
        if value is not None:
            value = validate_ndarray_with_shape(
                "line_inds", value, shape=("*",), dtype=int
            )
        self._line_inds = value

    @property
    def sort_inds(self):
        """Sorting indices from ABMN

        Returns
        -------
        numpy.ndarray of int
        """
        return self._sort_inds

    @sort_inds.setter
    def sort_inds(self, value):
        if value is not None:
            value = validate_ndarray_with_shape(
                "sort_inds", value, shape=("*",), dtype=int
            )
        self._sort_inds = value

    # Related to Physics and Discretization
    @property
    def mesh(self):
        """Mesh for discretization.

        Returns
        -------
        discretize.base.BaseMesh
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is not None:
            value = validate_type("mesh", value, BaseMesh, cast=False)
        self._mesh = value

    @property
    def dx(self):
        """Length of corecell in x-direction.

        Returns
        -------
        float
        """
        return self._dx

    @dx.setter
    def dx(self, value):
        if value is not None:
            value = validate_float("dx", value, min_val=0.0, inclusive_min=False)
        self._dx = value

    @property
    def dy(self):
        """Length of corecell in y-direction.

        Returns
        -------
        float
        """
        return self._dy

    @dy.setter
    def dy(self, value):
        if value is not None:
            value = validate_float("dy", value, min_val=0.0, inclusive_min=False)
        self._dy = value

    @property
    def dz(self):
        """Length of corecell in z-direction.

        Returns
        -------
        float
        """
        return self._dz

    @dz.setter
    def dz(self, value):
        if value is not None:
            value = validate_float("dz", value, min_val=0.0, inclusive_min=False)
        self._dz = value

    @property
    def npad_x(self):
        """The number of padding cells x-direction.

        Returns
        -------
        int
        """
        return self._npad_x

    @npad_x.setter
    def npad_x(self, value):
        self._npad_x = validate_integer("npad_x", value, min_val=0)

    @property
    def npad_y(self):
        """The number of padding cells y-direction.

        Returns
        -------
        int
        """
        return self._npad_y

    @npad_y.setter
    def npad_y(self, value):
        self._npad_y = validate_integer("npad_y", value, min_val=0)

    @property
    def npad_z(self):
        """The number of padding cells y-direction.

        Returns
        -------
        int
        """
        return self._npad_z

    @npad_z.setter
    def npad_z(self, value):
        self._npad_z = validate_integer("npad_z", value, min_val=0)

    @property
    def pad_rate_x(self):
        """Expansion rate of padding cells in  x-direction.

        Returns
        -------
        float
        """
        return self._pad_rate_x

    @pad_rate_x.setter
    def pad_rate_x(self, value):
        self._pad_rate_x = validate_float("pad_rate_x", value)

    @property
    def pad_rate_y(self):
        """Expansion rate of padding cells in  x-direction.

        Returns
        -------
        float
        """
        return self._pad_rate_y

    @pad_rate_y.setter
    def pad_rate_y(self, value):
        self._pad_rate_y = validate_float("pad_rate_y", value)

    @property
    def pad_rate_z(self):
        """Expansion rate of padding cells in  x-direction.

        Returns
        -------
        float
        """
        return self._pad_rate_z

    @pad_rate_z.setter
    def pad_rate_z(self, value):
        self._pad_rate_z = validate_float("pad_rate_z", value)

    @property
    def ncell_per_dipole(self):
        """The number of cells between dipole electrodes.

        Returns
        -------
        int
        """
        return self._ncell_per_dipole

    @ncell_per_dipole.setter
    def ncell_per_dipole(self, value):
        self._ncell_per_dipole = validate_integer("ncell_per_dipole", value, min_val=0)

    @property
    def corezlength(self):
        """Core depth (m).

        Returns
        -------
        float
        """
        return self._pad_rate_z

    @corezlength.setter
    def corezlength(self, value):
        if value is not None:
            value = validate_float("corezlength", value)
        self._corezlength = value

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
        G = geometric_factor(survey, space_type=self.space_type)
        return G

    def from_abmn_locations_to_survey(
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

        uniqSrc = unique_rows(np.c_[self.a_locations, self.b_locations])
        uniqElec = unique_rows(
            np.vstack(
                (self.a_locations, self.b_locations, self.m_locations, self.n_locations)
            )
        )
        self.electrode_locations = uniqElec[0]
        nSrc = uniqSrc[0].shape[0]
        ndata = self.a_locations.shape[0]

        if self.survey_layout == "SURFACE":
            # 2D locations
            source_lists = []
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

                source_lists.append(src)

            self.sort_inds = np.hstack(sort_inds)

            if dimension in (2, 3):
                survey = Survey(source_lists)
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
                        stacklevel=2,
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
                        stacklevel=2,
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
                pad_length_x = np.sum(unpack_widths([(dx, npad_x, pad_rate_x)]))
                pad_length_z = np.sum(unpack_widths([(dz, npad_z, pad_rate_z)]))

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

                mesh_width = np.sum(unpack_widths(hx))
                mesh_height = np.sum(unpack_widths(hz))

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

        actind = active_from_xyz(mesh, locs, method=method)

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
            label_tmp = r"Apparent Res. ($\Omega$m)"
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
        obsfile = np.genfromtxt(filename, delimiter=" \n", dtype=str, comments="!")
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
                topo = tmp_topo[1:, :]
                if topo.shape[0] != n_topo:
                    print(
                        ">> # of points for the topography is "
                        f"not {n_topo}, but {topo.shape[0]}"
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

        survey = self.from_abmn_locations_to_survey(
            a, b, m, n, survey_type=survey_type, data_dc=voltage
        )
        survey.dobs = voltage[self.sort_inds]
        survey.std = standard_deviation[self.sort_inds]
        survey.topo = topo
        return survey

    def write_to_csv(self, fname, dobs, standard_deviation=None, **kwargs):
        uncert = kwargs.pop("uncertainty", None)
        if uncert is not None:
            raise TypeError(
                "The uncertainty option has been removed, please use standard_deviation."
            )

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
            survey = self.from_abmn_locations_to_survey(
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
