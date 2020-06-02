from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator
import properties
from ....utils.code_utils import deprecate_class, deprecate_property

from ....utils import uniqueRows
from ....survey import BaseSurvey
from ..utils import drapeTopotoLoc
from . import receivers as Rx
from . import sources as Src
from ..utils import static_utils
from SimPEG import data

import warnings


class Survey(BaseSurvey):
    """
    Base DC survey
    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A DC source", Src.BaseSrc),
        default=[],
    )

    # Survey
    survey_geometry = properties.StringChoice(
        "Survey geometry of DC surveys",
        default="surface",
        choices=["surface", "borehole", "general"],
    )

    survey_type = properties.StringChoice(
        "DC-IP Survey type",
        default="dipole-dipole",
        choices=["dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"],
    )

    electrodes_info = None
    topo_function = None

    def __init__(self, source_list, **kwargs):
        super(Survey, self).__init__(source_list, **kwargs)

    @property
    def locations_a(self):
        """
        Location of the positive (+) current electrodes for each datum
        """
        if getattr(self, "_locations_a", None) is None:
            self._set_abmn_locations()
        return self._locations_a

    @property
    def locations_b(self):
        """
        Location of the negative (-) current electrodes for each datum
        """
        if getattr(self, "_locations_b", None) is None:
            self._set_abmn_locations()
        return self._locations_b

    @property
    def locations_m(self):
        """
        Location of the positive (+) potential electrodes for each datum
        """
        if getattr(self, "_locations_m", None) is None:
            self._set_abmn_locations()
        return self._locations_m

    @property
    def locations_n(self):
        """
        Location of the negative (-) potential electrodes for each datum
        """
        if getattr(self, "_locations_n", None) is None:
            self._set_abmn_locations()
        return self._locations_n

    a_locations = deprecate_property(
        locations_a, "a_locations", new_name="locations_a", removal_version="0.15.0"
    )
    b_locations = deprecate_property(
        locations_b, "b_locations", new_name="locations_b", removal_version="0.15.0"
    )
    m_locations = deprecate_property(
        locations_m, "m_locations", new_name="locations_m", removal_version="0.15.0"
    )
    n_locations = deprecate_property(
        locations_n, "n_locations", new_name="locations_n", removal_version="0.15.0"
    )

    @property
    def electrode_locations(self):
        """
        Unique locations of the A, B, M, N electrodes
        """
        loc_a = self.locations_a
        loc_b = self.locations_b
        loc_m = self.locations_m
        loc_n = self.locations_n
        return np.unique(np.vstack((loc_a, loc_b, loc_m, loc_n)), axis=0)

    def set_geometric_factor(
        self, data_type="volt", survey_type="dipole-dipole", space_type="half-space"
    ):

        geometric_factor = static_utils.geometric_factor(
            self, survey_type=survey_type, space_type=space_type
        )

        geometric_factor = data.Data(self, geometric_factor)
        for source in self.source_list:
            for rx in source.receiver_list:
                rx._geometric_factor = geometric_factor[source, rx]
                rx.data_type = data_type
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

                # Pole RX
                if isinstance(rx, Rx.Pole) or isinstance(rx, Rx.Pole):
                    locations_m.append(rx.locations)
                    locations_n.append(rx.locations)

                # Dipole RX
                elif isinstance(rx, Rx.Dipole) or isinstance(rx, Rx.Dipole):
                    locations_m.append(rx.locations[0])
                    locations_n.append(rx.locations[1])

        self._locations_a = np.vstack(locations_a)
        self._locations_b = np.vstack(locations_b)
        self._locations_m = np.vstack(locations_m)
        self._locations_n = np.vstack(locations_n)

    def getABMN_locations(self):
        warnings.warn(
            "The getABMN_locations method has been deprecated. Please instead "
            "ask for the property of interest: survey.locations_a, "
            "survey.locations_b, survey.locations_m, or survey.locations_n. "
            "This will be removed in version 0.15.0 of SimPEG",
            DeprecationWarning,
        )

    def drapeTopo(self, mesh, actind, option="top", topography=None, force=False):

        # 2D
        if mesh.dim == 2:
            if self.survey_geometry == "surface":
                if self.electrodes_info is None:
                    self.electrodes_info = uniqueRows(
                        np.hstack(
                            (
                                self.locations_a[:, 0],
                                self.locations_b[:, 0],
                                self.locations_m[:, 0],
                                self.locations_n[:, 0],
                            )
                        ).reshape([-1, 1])
                    )
                    self._electrode_locations = drapeTopotoLoc(
                        mesh,
                        self.electrodes_info[0].flatten(),
                        actind=actind,
                        option=option,
                    )
                temp = (self.electrode_locations[self.electrodes_info[2], 1]).reshape(
                    (self.locations_a.shape[0], 4), order="F"
                )
                self._locations_a = np.c_[self.locations_a[:, 0], temp[:, 0]]
                self._locations_b = np.c_[self.locations_b[:, 0], temp[:, 1]]
                self._locations_m = np.c_[self.locations_m[:, 0], temp[:, 2]]
                self._locations_n = np.c_[self.locations_n[:, 0], temp[:, 3]]

                # Make interpolation function
                self.topo_function = interp1d(
                    self.electrode_locations[:, 0], self.electrode_locations[:, 1]
                )

                # Loop over all Src and Rx locs and Drape topo
                for source in self.source_list:
                    # Pole Src
                    if isinstance(source, Src.Pole):
                        locA = source.location.flatten()
                        z_SrcA = self.topo_function(locA[0])
                        source.location = np.array([locA[0], z_SrcA])
                        for rx in source.receiver_list:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole):
                                locM = rx.locations.copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                rx.locations = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole):
                                locM = rx.locations[0].copy()
                                locN = rx.locations[1].copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                z_RxN = self.topo_function(locN[:, 0])
                                rx.locations[0] = np.c_[locM[:, 0], z_RxM]
                                rx.locations[1] = np.c_[locN[:, 0], z_RxN]
                            else:
                                raise Exception()

                    # Dipole Src
                    elif isinstance(source, Src.Dipole):
                        locA = source.location[0].flatten()
                        locB = source.location[1].flatten()
                        z_SrcA = self.topo_function(locA[0])
                        z_SrcB = self.topo_function(locB[0])

                        source.location[0] = np.array([locA[0], z_SrcA])
                        source.location[1] = np.array([locB[0], z_SrcB])

                        for rx in source.receiver_list:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole):
                                locM = rx.locations.copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                rx.locations = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole):
                                locM = rx.locations[0].copy()
                                locN = rx.locations[1].copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                z_RxN = self.topo_function(locN[:, 0])
                                rx.locations[0] = np.c_[locM[:, 0], z_RxM]
                                rx.locations[1] = np.c_[locN[:, 0], z_RxN]
                            else:
                                raise Exception()

            elif self.survey_geometry == "borehole":
                raise Exception("Not implemented yet for borehole survey_geometry")
            else:
                raise Exception(
                    "Input valid survey survey_geometry: surface or borehole"
                )

        if mesh.dim == 3:
            if self.survey_geometry == "surface":
                if self.electrodes_info is None:
                    self.electrodes_info = uniqueRows(
                        np.vstack(
                            (
                                self.locations_a[:, :2],
                                self.locations_b[:, :2],
                                self.locations_m[:, :2],
                                self.locations_n[:, :2],
                            )
                        )
                    )
                self.electrode_locations = drapeTopotoLoc(
                    mesh, self.electrodes_info[0], actind=actind, topo=topography
                )

                temp = (self.electrode_locations[self.electrodes_info[2], 1]).reshape(
                    (self.locations_a.shape[0], 4), order="F"
                )

                self.locations_a = np.c_[self.locations_a[:, :2], temp[:, 0]]
                self.locations_b = np.c_[self.locations_b[:, :2], temp[:, 1]]
                self.locations_m = np.c_[self.locations_m[:, :2], temp[:, 2]]
                self.locations_n = np.c_[self.locations_n[:, :2], temp[:, 3]]

                # Make interpolation function
                self.topo_function = NearestNDInterpolator(
                    self.electrode_locations[:, :2], self.electrode_locations[:, 2]
                )
                # Loop over all Src and Rx locs and Drape topo
                for source in self.source_list:
                    # Pole Src
                    if isinstance(source, Src.Pole):
                        locA = source.location.reshape([1, -1])
                        z_SrcA = self.topo_function(locA[0, :2])
                        source.location = np.r_[locA[0, :2].flatten(), z_SrcA]

                        for rx in source.receiver_list:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole):
                                locM = rx.locations.copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                rx.locations = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole):
                                locM = rx.locations[0].copy()
                                locN = rx.locations[1].copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                z_RxN = self.topo_function(locN[:, :2])
                                rx.locations[0] = np.c_[locM[:, :2], z_RxM]
                                rx.locations[1] = np.c_[locN[:, :2], z_RxN]
                            else:
                                raise Exception()

                    # Dipole Src
                    elif isinstance(source, Src.Dipole):
                        locA = source.location[0].reshape([1, -1])
                        locB = source.location[1].reshape([1, -1])
                        z_SrcA = self.topo_function(locA[0, :2])
                        z_SrcB = self.topo_function(locB[0, :2])
                        source.location[0] = np.r_[locA[0, :2].flatten(), z_SrcA]
                        source.location[1] = np.r_[locB[0, :2].flatten(), z_SrcB]

                        for rx in source.receiver_list:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole):
                                locM = rx.locations.copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                rx.locations = np.c_[locM[:, :2], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole):
                                locM = rx.locations[0].copy()
                                locN = rx.locations[1].copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                z_RxN = self.topo_function(locN[:, :2])
                                rx.locations[0] = np.c_[locM[:, :2], z_RxM]
                                rx.locations[1] = np.c_[locN[:, :2], z_RxN]
                            else:
                                raise Exception()

            elif self.survey_geometry == "borehole":
                raise Exception("Not implemented yet for borehole survey_geometry")
            else:
                raise Exception(
                    "Input valid survey survey_geometry: surface or borehole"
                )


############
# Deprecated
############


@deprecate_class(removal_version="0.15.0")
class Survey_ky(Survey):
    pass
