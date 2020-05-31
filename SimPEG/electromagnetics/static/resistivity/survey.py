from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator
import properties
from ....utils.code_utils import deprecate_class

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
    def a_locations(self):
        """
        Location of the positive (+) current electrodes for each datum
        """
        if getattr(self, "_a_locations", None) is None:
            self._set_abmn_locations()
        return self._a_locations

    @property
    def b_locations(self):
        """
        Location of the negative (-) current electrodes for each datum
        """
        if getattr(self, "_b_locations", None) is None:
            self._set_abmn_locations()
        return self._b_locations

    @property
    def m_locations(self):
        """
        Location of the positive (+) potential electrodes for each datum
        """
        if getattr(self, "_m_locations", None) is None:
            self._set_abmn_locations()
        return self._m_locations

    @property
    def n_locations(self):
        """
        Location of the negative (-) potential electrodes for each datum
        """
        if getattr(self, "_n_locations", None) is None:
            self._set_abmn_locations()
        return self._n_locations

    @property
    def electrode_locations(self):
        """
        Locations of the A, B, M, N electrodes stacketd vertically
        [A.T, B.T, M.T, N.T].T
        """
        return np.vstack(
            [self.a_locations, self.b_locations, self.m_locations, self.n_locations,]
        )

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
        a_locations = []
        b_locations = []
        m_locations = []
        n_locations = []
        for source in self.source_list:
            for rx in source.receiver_list:
                nRx = rx.nD
                # Pole Source
                if isinstance(source, Src.Pole):
                    a_locations.append(
                        source.location.reshape([1, -1]).repeat(nRx, axis=0)
                    )
                    b_locations.append(
                        source.location.reshape([1, -1]).repeat(nRx, axis=0)
                    )
                # Dipole Source
                elif isinstance(source, Src.Dipole):
                    a_locations.append(
                        source.location[0].reshape([1, -1]).repeat(nRx, axis=0)
                    )
                    b_locations.append(
                        source.location[1].reshape([1, -1]).repeat(nRx, axis=0)
                    )

                # Pole RX
                if isinstance(rx, Rx.Pole) or isinstance(rx, Rx.Pole):
                    m_locations.append(rx.locations)
                    n_locations.append(rx.locations)

                # Dipole RX
                elif isinstance(rx, Rx.Dipole) or isinstance(rx, Rx.Dipole):
                    m_locations.append(rx.locations[0])
                    n_locations.append(rx.locations[1])

        self._a_locations = np.vstack(a_locations)
        self._b_locations = np.vstack(b_locations)
        self._m_locations = np.vstack(m_locations)
        self._n_locations = np.vstack(n_locations)

    def getABMN_locations(self):
        warnings.warn(
            "The getABMN_locations method has been deprecated. Please instead "
            "ask for the property of interest: survey.a_locations, "
            "survey.b_locations, survey.m_locations, or survey.n_locations. "
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
                                self.a_locations[:, 0],
                                self.b_locations[:, 0],
                                self.m_locations[:, 0],
                                self.n_locations[:, 0],
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
                    (self.a_locations.shape[0], 4), order="F"
                )
                self._a_locations = np.c_[self.a_locations[:, 0], temp[:, 0]]
                self._b_locations = np.c_[self.b_locations[:, 0], temp[:, 1]]
                self._m_locations = np.c_[self.m_locations[:, 0], temp[:, 2]]
                self._n_locations = np.c_[self.n_locations[:, 0], temp[:, 3]]

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
                                self.a_locations[:, :2],
                                self.b_locations[:, :2],
                                self.m_locations[:, :2],
                                self.n_locations[:, :2],
                            )
                        )
                    )
                self.electrode_locations = drapeTopotoLoc(
                    mesh, self.electrodes_info[0], actind=actind, topo=topography
                )

                temp = (self.electrode_locations[self.electrodes_info[2], 1]).reshape(
                    (self.a_locations.shape[0], 4), order="F"
                )

                self.a_locations = np.c_[self.a_locations[:, :2], temp[:, 0]]
                self.b_locations = np.c_[self.b_locations[:, :2], temp[:, 1]]
                self.m_locations = np.c_[self.m_locations[:, :2], temp[:, 2]]
                self.n_locations = np.c_[self.n_locations[:, :2], temp[:, 3]]

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
