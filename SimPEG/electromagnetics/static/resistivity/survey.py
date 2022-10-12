from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

# import properties
from ....utils.code_utils import deprecate_property, validate_string

from ....survey import BaseSurvey
from ..utils import drapeTopotoLoc
from . import receivers as Rx
from . import sources as Src
from ..utils import static_utils
from SimPEG import data


class Survey(BaseSurvey):
    """DC/IP survey class

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.static.resistivity.sources.BaseSrc
        List of SimPEG DC/IP sources
    survey_geometry : {"surface", "borehole", "general"}
        Survey geometry.
    survey_type : {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}
        Survey type.
    """

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A DC source", Src.BaseSrc),
    #     default=[],
    # )

    # # Survey
    # survey_geometry = properties.StringChoice(
    #     "Survey geometry of DC surveys",
    #     default="surface",
    #     choices=["surface", "borehole", "general"],
    # )

    # survey_type = properties.StringChoice(
    #     "DC-IP Survey type",
    #     default="dipole-dipole",
    #     choices=["dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"],
    # )

    def __init__(
        self,
        source_list,
        survey_geometry="surface",
        survey_type="dipole-dipole",
        **kwargs,
    ):
        super(Survey, self).__init__(source_list, **kwargs)
        self.survey_geometry = survey_geometry
        self.survey_type = survey_type

    @property
    def survey_geometry(self):
        """Survey geometry

        This property is deprecated.

        Returns
        -------
        str
            Survey geometry; one of {"surface", "borehole", "general"}
        """
        return self._survey_geometry

    @survey_geometry.setter
    def survey_geometry(self, var):
        var = validate_string(
            "survey_geometry", var, ("surface", "borehole", "general")
        )
        self._survey_geometry = var

    @property
    def survey_type(self):
        """Survey type; one of {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}

        Returns
        -------
        str
            Survey type; one of {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}
        """
        return self._survey_type

    @survey_type.setter
    def survey_type(self, var):
        var = validate_string(
            "survey_type",
            var,
            ("dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"),
        )
        self._survey_type = var

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.survey_type}; "
            f"#sources: {self.nSrc}; #data: {self.nD})"
        )

    @property
    def locations_a(self):
        """
        Locations of the positive (+) current electrodes in the survey

        Returns
        -------
        (nD, dim) numpy.ndarray
            Locations of the positive (+) current electrodes in the survey
        """
        if getattr(self, "_locations_a", None) is None:
            self._set_abmn_locations()
        return self._locations_a

    @property
    def locations_b(self):
        """
        Locations of the negative (-) current electrodes in the survey

        Returns
        -------
        (nD, dim) numpy.ndarray
            Locations of the negative (-) current electrodes in the survey
        """
        if getattr(self, "_locations_b", None) is None:
            self._set_abmn_locations()
        return self._locations_b

    @property
    def locations_m(self):
        """
        Locations of the positive (+) potential electrodes in the survey

        Returns
        -------
        (nD, dim) numpy.ndarray
            Locations of the positive (+) potential electrodes in the survey
        """
        if getattr(self, "_locations_m", None) is None:
            self._set_abmn_locations()
        return self._locations_m

    @property
    def locations_n(self):
        """
        Locations of the negative (-) potential electrodes in the survey

        Returns
        -------
        (nD, dim) numpy.ndarray
            Locations of the negative (-) potential electrodes in the survey
        """
        if getattr(self, "_locations_n", None) is None:
            self._set_abmn_locations()
        return self._locations_n

    @property
    def unique_electrode_locations(self):
        """
        Unique locations of the A, B, M, N electrodes

        Returns
        -------
        (n, dim) numpy.ndarray
            The set of unique electrode locations used in the survey
        """
        loc_a = self.locations_a
        loc_b = self.locations_b
        loc_m = self.locations_m
        loc_n = self.locations_n
        return np.unique(np.vstack((loc_a, loc_b, loc_m, loc_n)), axis=0)

    electrode_locations = deprecate_property(
        unique_electrode_locations,
        "electrode_locations",
        new_name="unique_electrode_locations",
        removal_version="0.17.0",
        future_warn=True,
    )

    @property
    def source_locations(self):
        """
        Returns, in order, the source locations for all sources in the survey.

        Returns
        -------
        list of numpy.ndarray
            List of length 2 containing the A and B electrode locations, in order.
        """
        src_a = []
        src_b = []

        for src in self.source_list:

            src_a.append(src.location_a)
            src_b.append(src.location_b)

        return [np.vstack(src_a), np.vstack(src_b)]

    def set_geometric_factor(
        self,
        space_type="halfspace",
        data_type=None,
        survey_type=None,
    ):
        """
        Set and return the geometric factor for all data

        Parameters
        ----------
        space_type : {'halfspace', 'wholespace'}
            Calculate geometric factors using a half-space or whole-space formula.
        data_type : str, default = ``None``
            This input argument is now deprecated
        survey_type : str, default = ``None``
            This input argument is now deprecated

        Returns
        -------
        (nD) numpy.ndarray
            The geometric factor for each datum
        """
        if data_type is not None:
            raise TypeError(
                "The data_type kwarg has been removed, please set the data_type on the "
                "receiver object itself."
            )
        if survey_type is not None:
            raise TypeError("The survey_type parameter is no longer needed")

        geometric_factor = static_utils.geometric_factor(self, space_type=space_type)

        geometric_factor = data.Data(self, geometric_factor)
        for source in self.source_list:
            for rx in source.receiver_list:
                if data_type is not None:
                    rx.data_type = data_type
                if rx.data_type == "apparent_resistivity":
                    rx._geometric_factor[source] = geometric_factor[source, rx]
        return geometric_factor

    def _set_abmn_locations(self):
        locations_a = []
        locations_b = []
        locations_m = []
        locations_n = []
        for source in self.source_list:
            for rx in source.receiver_list:
                nRx = rx.nD
                # Pole Source
                if isinstance(source, Src.Pole):
                    locations_a.append(
                        source.location.reshape([1, -1]).repeat(nRx, axis=0)
                    )
                    locations_b.append(
                        source.location.reshape([1, -1]).repeat(nRx, axis=0)
                    )
                # Dipole Source
                elif isinstance(source, Src.Dipole):
                    locations_a.append(
                        source.location[0].reshape([1, -1]).repeat(nRx, axis=0)
                    )
                    locations_b.append(
                        source.location[1].reshape([1, -1]).repeat(nRx, axis=0)
                    )
                elif isinstance(source, Src.Multipole):
                    location_tiled = np.tile(source.location, (nRx, 1))
                    locations_a.append(location_tiled)
                    locations_b.append(location_tiled)
                # Pole RX
                if isinstance(rx, Rx.Pole):
                    locations_m.append(rx.locations)
                    locations_n.append(rx.locations)

                # Dipole RX
                elif isinstance(rx, Rx.Dipole):
                    locations_m.append(rx.locations[0])
                    locations_n.append(rx.locations[1])

        self._locations_a = np.vstack(locations_a)
        self._locations_b = np.vstack(locations_b)
        self._locations_m = np.vstack(locations_m)
        self._locations_n = np.vstack(locations_n)

    def getABMN_locations(self):
        """The 'getABMN_locations' method has been removed."""
        raise TypeError(
            "The getABMN_locations method has been Removed. Please instead "
            "ask for the property of interest: survey.locations_a, "
            "survey.locations_b, survey.locations_m, or survey.locations_n."
        )

    def drape_electrodes_on_topography(
        self, mesh, ind_active, option="top", topography=None, force=False
    ):
        """Shift electrode locations to discrete surface topography.

        Parameters
        ----------
        mesh : discretize.TensorMesh or discretize.TreeMesh
            The mesh on which the discretized fields are computed
        ind_active : numpy.ndarray of int or bool
            Active topography cells
        option :{"top", "center"}
            Define topography at tops of cells or cell centers.
        topography : (n, dim) numpy.ndarray, default = ``None``
            Surface topography
        force : bool, default = ``False``
            If ``True`` force electrodes to surface even if borehole

        """
        if self.survey_geometry == "surface":
            loc_a = self.locations_a[:, :2]
            loc_b = self.locations_b[:, :2]
            loc_m = self.locations_m[:, :2]
            loc_n = self.locations_n[:, :2]
            unique_electrodes, inv = np.unique(
                np.vstack((loc_a, loc_b, loc_m, loc_n)), return_inverse=True, axis=0
            )
            inv_a, inv = inv[: len(loc_a)], inv[len(loc_a) :]
            inv_b, inv = inv[: len(loc_b)], inv[len(loc_b) :]
            inv_m, inv_n = inv[: len(loc_m)], inv[len(loc_m) :]

            electrodes_shifted = drapeTopotoLoc(
                mesh, unique_electrodes, ind_active=ind_active, option=option
            )
            a_shifted = electrodes_shifted[inv_a]
            b_shifted = electrodes_shifted[inv_b]
            m_shifted = electrodes_shifted[inv_m]
            n_shifted = electrodes_shifted[inv_n]
            # These should all be in the same order as the survey datas
            ind = 0
            for src in self.source_list:
                a_loc, b_loc = a_shifted[ind], b_shifted[ind]
                if isinstance(src, (Src.Pole, Src.Multipole)):
                    src.location = [a_loc]
                else:
                    src.location = [a_loc, b_loc]
                for rx in src.receiver_list:
                    end = ind + rx.nD
                    m_locs, n_locs = m_shifted[ind:end], n_shifted[ind:end]
                    if isinstance(rx, Rx.Pole):
                        rx.locations = m_locs
                    else:
                        rx.locations = [m_locs, n_locs]
                    ind = end
            self._locations_a = a_shifted
            self._locations_b = b_shifted
            self._locations_m = m_shifted
            self._locations_n = n_shifted

        elif self.survey_geometry == "borehole":
            raise Exception("Not implemented yet for borehole survey_geometry")
        else:
            raise Exception(
                f"Input valid survey survey_geometry: {self.survey_geometry}"
            )

    def drapeTopo(self, *args, **kwargs):
        """This method is deprecated. See :meth:`drape_electrodes_on_topography`"""
        raise TypeError(
            "The drapeTopo method has been removed. Please instead "
            "use the drape_electrodes_on_topography method."
        )
