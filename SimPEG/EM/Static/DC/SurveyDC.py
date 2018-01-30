from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
from . import RxDC as Rx
from . import SrcDC as Src
from SimPEG.EM.Base import BaseEMSurvey
import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator
import properties


class Survey(BaseEMSurvey, properties.HasProperties):
    """
    Base DC survey
    """
    rxPair = Rx.BaseRx
    srcPair = Src.BaseSrc

    # Survey
    survey_geometry = properties.StringChoice(
        "Survey geometry of DC surveys",
        default="surface",
        choices=["surface", "borehole", "general"]
    )

    survey_type = properties.StringChoice(
        "DC-IP Survey type",
        default="dipole-dipole",
        choices=[
            "dipole-dipole", "pole-dipole",
            "dipole-pole", "pole-pole"
        ]
    )

    a_locations = properties.Array(
        "locations of the positive (+) current electrodes",
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    b_locations = properties.Array(
        "locations of the negative (-) current electrodes",
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    m_locations = properties.Array(
        "locations of the positive (+) potential electrodes",
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    n_locations = properties.Array(
        "locations of the negative (-) potential electrodes",
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    electrode_locations = properties.Array(
        "unique locations of a, b, m, n electrodes",
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    electrodes_info = None
    topo_function = None

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)

    def getABMN_locations(self):
        a_locations = []
        b_locations = []
        m_locations = []
        n_locations = []
        for src in self.srcList:
            # Pole
            if isinstance(src, Src.Pole):
                for rx in src.rxList:
                    nRx = rx.nD
                    a_locations.append(
                        src.loc.reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    b_locations.append(
                        src.loc.reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    # Pole
                    if isinstance(rx, Rx.Pole) or isinstance(rx, Rx.Pole_ky):
                        m_locations.append(rx.locs)
                        n_locations.append(rx.locs)
                    # Dipole
                    elif isinstance(rx, Rx.Dipole) or isinstance(rx, Rx.Dipole_ky):
                        m_locations.append(rx.locs[0])
                        n_locations.append(rx.locs[1])
            # Dipole
            elif isinstance(src, Src.Dipole):
                for rx in src.rxList:
                    nRx = rx.nD
                    a_locations.append(
                        src.loc[0].reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    b_locations.append(
                        src.loc[1].reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    # Pole
                    if isinstance(rx, Rx.Pole) or isinstance(rx, Rx.Pole_ky):
                        m_locations.append(rx.locs)
                        n_locations.append(rx.locs)
                    # Dipole
                    elif isinstance(rx, Rx.Dipole) or isinstance(rx, Rx.Dipole_ky):
                        m_locations.append(rx.locs[0])
                        n_locations.append(rx.locs[1])

        self.a_locations = np.vstack(a_locations)
        self.b_locations = np.vstack(b_locations)
        self.m_locations = np.vstack(m_locations)
        self.n_locations = np.vstack(n_locations)

    def drapeTopo(self, mesh, actind, option='top'):
        if self.a_locations is None:
            self.getABMN_locations()

        # 2D
        if mesh.dim == 2:
            if self.survey_geometry == "surface":
                if self.electrodes_info is None:
                    self.electrodes_info = SimPEG.Utils.uniqueRows(
                        np.hstack((
                            self.a_locations[:, 0],
                            self.b_locations[:, 0],
                            self.m_locations[:, 0],
                            self.n_locations[:, 0],
                            )).reshape([-1, 1])
                        )
                    self.electrode_locations = SimPEG.EM.Static.Utils.drapeTopotoLoc(
                        mesh,
                        self.electrodes_info[0].flatten(),
                        actind=actind,
                        option=option
                    )
                temp = (
                    self.electrode_locations[self.electrodes_info[2], 1]
                ).reshape((self.a_locations.shape[0], 4), order="F")
                self.a_locations = np.c_[self.a_locations[:, 0], temp[:, 0]]
                self.b_locations = np.c_[self.b_locations[:, 0], temp[:, 1]]
                self.m_locations = np.c_[self.m_locations[:, 0], temp[:, 2]]
                self.n_locations = np.c_[self.n_locations[:, 0], temp[:, 3]]

                # Make interpolation function
                self.topo_function = interp1d(
                    self.electrode_locations[:, 0], self.electrode_locations[:, 1]
                    )

                # Loop over all Src and Rx locs and Drape topo
                for src in self.srcList:
                    # Pole Src
                    if isinstance(src, Src.Pole):
                        locA = src.loc.flatten()
                        z_SrcA = self.topo_function(locA[0])
                        src.loc = np.array([locA[0], z_SrcA])
                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole) or isinstance(rx, Rx.Pole_ky):
                                locM = rx.locs.copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                rx.locs = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole) or isinstance(rx, Rx.Dipole_ky):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                z_RxN = self.topo_function(locN[:, 0])
                                rx.locs[0] = np.c_[locM[:, 0], z_RxM]
                                rx.locs[1] = np.c_[locN[:, 0], z_RxN]
                            else:
                                raise Exception()

                    # Dipole Src
                    elif isinstance(src, Src.Dipole):
                        locA = src.loc[0].flatten()
                        locB = src.loc[1].flatten()
                        z_SrcA = self.topo_function(locA[0])
                        z_SrcB = self.topo_function(locB[0])

                        src.loc[0] = np.array([locA[0], z_SrcA])
                        src.loc[1] = np.array([locB[0], z_SrcB])

                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole) or isinstance(rx, Rx.Pole_ky):
                                locM = rx.locs.copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                rx.locs = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole) or isinstance(rx, Rx.Dipole_ky):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topo_function(locM[:, 0])
                                z_RxN = self.topo_function(locN[:, 0])
                                rx.locs[0] = np.c_[locM[:, 0], z_RxM]
                                rx.locs[1] = np.c_[locN[:, 0], z_RxN]
                            else:
                                raise Exception()

            elif self.survey_geometry == "borehole":
                raise Exception(
                    "Not implemented yet for borehole survey_geometry"
                    )
            else:
                raise Exception(
                    "Input valid survey survey_geometry: surface or borehole"
                    )

        if mesh.dim == 3:
            if self.survey_geometry == "surface":
                if self.electrodes_info is None:
                    self.electrodes_info = SimPEG.Utils.uniqueRows(
                        np.vstack((
                            self.a_locations[:, :2],
                            self.b_locations[:, :2],
                            self.m_locations[:, :2],
                            self.n_locations[:, :2],
                            ))
                        )
                    self.electrode_locations = SimPEG.EM.Static.Utils.drapeTopotoLoc(
                        mesh, self.electrodes_info[0], actind=actind
                        )
                temp = (
                    self.electrode_locations[self.electrodes_info[2], 1]
                    ).reshape((self.a_locations.shape[0], 4), order="F")

                self.a_locations = np.c_[self.a_locations[:, :2], temp[:, 0]]
                self.b_locations = np.c_[self.b_locations[:, :2], temp[:, 1]]
                self.m_locations = np.c_[self.m_locations[:, :2], temp[:, 2]]
                self.n_locations = np.c_[self.n_locations[:, :2], temp[:, 3]]

                # Make interpolation function
                self.topo_function = NearestNDInterpolator(
                    self.electrode_locations[:, :2],
                    self.electrode_locations[:, 2]
                    )
                # Loop over all Src and Rx locs and Drape topo
                for src in self.srcList:
                    # Pole Src
                    if isinstance(src, Src.Pole):
                        locA = src.loc.reshape([1, -1])
                        z_SrcA = self.topo_function(locA[0, :2])
                        src.loc = np.r_[locA[0, :2].flatten(), z_SrcA]

                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole):
                                locM = rx.locs.copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                rx.locs = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                z_RxN = self.topo_function(locN[:, :2])
                                rx.locs[0] = np.c_[locM[:, :2], z_RxM]
                                rx.locs[1] = np.c_[locN[:, :2], z_RxN]
                            else:
                                raise Exception()

                    # Dipole Src
                    elif isinstance(src, Src.Dipole):
                        locA = src.loc[0].reshape([1, -1])
                        locB = src.loc[1].reshape([1, -1])
                        z_SrcA = self.topo_function(locA[0, :2])
                        z_SrcB = self.topo_function(locB[0, :2])
                        src.loc[0] = np.r_[locA[0, :2].flatten(), z_SrcA]
                        src.loc[1] = np.r_[locB[0, :2].flatten(), z_SrcB]

                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, Rx.Pole):
                                locM = rx.locs.copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                rx.locs = np.c_[locM[:, :2], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, Rx.Dipole):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topo_function(locM[:, :2])
                                z_RxN = self.topo_function(locN[:, :2])
                                rx.locs[0] = np.c_[locM[:, :2], z_RxM]
                                rx.locs[1] = np.c_[locN[:, :2], z_RxN]
                            else:
                                raise Exception()

            elif self.survey_geometry == "borehole":
                raise Exception(
                    "Not implemented yet for borehole survey_geometry"
                    )
            else:
                raise Exception(
                    "Input valid survey survey_geometry: surface or borehole"
                    )


class Survey_ky(Survey):
    """
    2.5D survey
    """
    rxPair = Rx.BaseRx
    srcPair = Src.BaseSrc

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)

    def eval(self, f):
        """
        Project fields to receiver locations
        :param Fields u: fields object
        :rtype: numpy.ndarray
        :return: data
        """
        data = SimPEG.Survey.Data(self)
        kys = self.prob.kys
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.eval(kys, src, self.mesh, f)
        return data
