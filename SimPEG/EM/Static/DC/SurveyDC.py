from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from .RxDC import BaseRx
from .SrcDC import BaseSrc
import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator

class Survey(BaseEMSurvey):
    """
    Base DC survey
    """
    rxPair = BaseRx
    srcPair = BaseSrc
    # TODO: change this using properties
    Alocs = None
    Blocs = None
    Mlocs = None
    Nlocs = None
    elecInfo = None
    uniqElecLocs = None
    geometry = "SURFACE"
    topoFunc = None
    surveyType = None

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)

    def getABMNLocs(self):
        Alocs = []
        Blocs = []
        Mlocs = []
        Nlocs = []
        for src in self.srcList:
            # Pole
            if isinstance(src, SimPEG.EM.Static.DC.Src.Pole):
                for rx in src.rxList:
                    nRx = rx.nD
                    Alocs.append(
                        src.loc.reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    Blocs.append(
                        src.loc.reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    # Pole
                    if isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole_ky):
                        Mlocs.append(rx.locs)
                        Nlocs.append(rx.locs)
                    # Dipole
                    elif isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole_ky):
                        Mlocs.append(rx.locs[0])
                        Nlocs.append(rx.locs[1])
            # Dipole
            elif isinstance(src, SimPEG.EM.Static.DC.Src.Dipole):
                for rx in src.rxList:
                    nRx = rx.nD
                    Alocs.append(
                        src.loc[0].reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    Blocs.append(
                        src.loc[1].reshape([1, -1]).repeat(nRx, axis=0)
                        )
                    # Pole
                    if isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole_ky):
                        Mlocs.append(rx.locs)
                        Nlocs.append(rx.locs)
                    # Dipole
                    elif isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole_ky):
                        Mlocs.append(rx.locs[0])
                        Nlocs.append(rx.locs[1])

        self.Alocs = np.vstack(Alocs)
        self.Blocs = np.vstack(Blocs)
        self.Mlocs = np.vstack(Mlocs)
        self.Nlocs = np.vstack(Nlocs)

    def drapeTopo(self, mesh, actind, option='top'):
        if self.Alocs is None:
            self.getABMNLocs()

        # 2D
        if mesh.dim == 2:
            if self.geometry == "SURFACE":
                if self.elecInfo is None:
                    self.elecInfo = SimPEG.Utils.uniqueRows(
                        np.hstack((
                            self.Alocs[:, 0],
                            self.Blocs[:, 0],
                            self.Mlocs[:, 0],
                            self.Nlocs[:, 0],
                            )).reshape([-1, 1])
                        )
                    self.uniqElecLocs = SimPEG.EM.Static.Utils.drapeTopotoLoc(
                        mesh,
                        self.elecInfo[0].flatten(),
                        actind=actind,
                        option=option
                        )
                temp = (
                    self.uniqElecLocs[self.elecInfo[2], 1]
                    ).reshape((self.Alocs.shape[0], 4), order="F")
                self.Alocs = np.c_[self.Alocs[:, 0], temp[:, 0]]
                self.Blocs = np.c_[self.Blocs[:, 0], temp[:, 1]]
                self.Mlocs = np.c_[self.Mlocs[:, 0], temp[:, 2]]
                self.Nlocs = np.c_[self.Nlocs[:, 0], temp[:, 3]]

                # Make interpolation function
                self.topoFunc = interp1d(
                    self.uniqElecLocs[:, 0], self.uniqElecLocs[:, 1]
                    )

                # Loop over all Src and Rx locs and Drape topo
                for src in self.srcList:
                    # Pole Src
                    if isinstance(src, SimPEG.EM.Static.DC.Src.Pole):
                        locA = src.loc.flatten()
                        z_SrcA = self.topoFunc(locA[0])
                        src.loc = np.array([locA[0], z_SrcA])
                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole_ky):
                                locM = rx.locs.copy()
                                z_RxM = self.topoFunc(locM[:, 0])
                                rx.locs = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole_ky):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topoFunc(locM[:, 0])
                                z_RxN = self.topoFunc(locN[:, 0])
                                rx.locs[0] = np.c_[locM[:, 0], z_RxM]
                                rx.locs[1] = np.c_[locN[:, 0], z_RxN]
                            else:
                                raise Exception()

                    # Dipole Src
                    elif isinstance(src, SimPEG.EM.Static.DC.Src.Dipole):
                        locA = src.loc[0].flatten()
                        locB = src.loc[1].flatten()
                        z_SrcA = self.topoFunc(locA[0])
                        z_SrcB = self.topoFunc(locB[0])

                        src.loc[0] = np.array([locA[0], z_SrcA])
                        src.loc[1] = np.array([locB[0], z_SrcB])

                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole_ky):
                                locM = rx.locs.copy()
                                z_RxM = self.topoFunc(locM[:, 0])
                                rx.locs = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole) or isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole_ky):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topoFunc(locM[:, 0])
                                z_RxN = self.topoFunc(locN[:, 0])
                                rx.locs[0] = np.c_[locM[:, 0], z_RxM]
                                rx.locs[1] = np.c_[locN[:, 0], z_RxN]
                            else:
                                raise Exception()

            elif self.geometry == "BOREHOLE":
                raise Exception(
                    "Not implemented yet for BOREHOLE geometry"
                    )
            else:
                raise Exception(
                    "Input valid survey geometry: SURFACE or BOREHOLE"
                    )

        if mesh.dim == 3:
            if self.geometry == "SURFACE":
                if self.elecInfo is None:
                    self.elecInfo = SimPEG.Utils.uniqueRows(
                        np.vstack((
                            self.Alocs[:, :2],
                            self.Blocs[:, :2],
                            self.Mlocs[:, :2],
                            self.Nlocs[:, :2],
                            ))
                        )
                    self.uniqElecLocs = SimPEG.EM.Static.Utils.drapeTopotoLoc(
                        mesh, self.elecInfo[0], actind=actind
                        )
                temp = (
                    self.uniqElecLocs[self.elecInfo[2], 1]
                    ).reshape((self.Alocs.shape[0], 4), order="F")

                self.Alocs = np.c_[self.Alocs[:, :2], temp[:, 0]]
                self.Blocs = np.c_[self.Blocs[:, :2], temp[:, 1]]
                self.Mlocs = np.c_[self.Mlocs[:, :2], temp[:, 2]]
                self.Nlocs = np.c_[self.Nlocs[:, :2], temp[:, 3]]

                # Make interpolation function
                self.topoFunc = NearestNDInterpolator(
                    self.uniqElecLocs[:, :2],
                    self.uniqElecLocs[:, 2]
                    )
                # Loop over all Src and Rx locs and Drape topo
                for src in self.srcList:
                    # Pole Src
                    if isinstance(src, SimPEG.EM.Static.DC.Src.Pole):
                        locA = src.loc.reshape([1, -1])
                        z_SrcA = self.topoFunc(locA[0, :2])
                        src.loc = np.r_[locA[0, :2].flatten(), z_SrcA]

                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole):
                                locM = rx.locs.copy()
                                z_RxM = self.topoFunc(locM[:, :2])
                                rx.locs = np.c_[locM[:, 0], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topoFunc(locM[:, :2])
                                z_RxN = self.topoFunc(locN[:, :2])
                                rx.locs[0] = np.c_[locM[:, :2], z_RxM]
                                rx.locs[1] = np.c_[locN[:, :2], z_RxN]
                            else:
                                raise Exception()

                    # Dipole Src
                    elif isinstance(src, SimPEG.EM.Static.DC.Src.Dipole):
                        locA = src.loc[0].reshape([1, -1])
                        locB = src.loc[1].reshape([1, -1])
                        z_SrcA = self.topoFunc(locA[0, :2])
                        z_SrcB = self.topoFunc(locB[0, :2])
                        src.loc[0] = np.r_[locA[0, :2].flatten(), z_SrcA]
                        src.loc[1] = np.r_[locB[0, :2].flatten(), z_SrcB]

                        for rx in src.rxList:
                            # Pole Rx
                            if isinstance(rx, SimPEG.EM.Static.DC.Rx.Pole):
                                locM = rx.locs.copy()
                                z_RxM = self.topoFunc(locM[:, :2])
                                rx.locs = np.c_[locM[:, :2], z_RxM]
                            # Dipole Rx
                            elif isinstance(rx, SimPEG.EM.Static.DC.Rx.Dipole):
                                locM = rx.locs[0].copy()
                                locN = rx.locs[1].copy()
                                z_RxM = self.topoFunc(locM[:, :2])
                                z_RxN = self.topoFunc(locN[:, :2])
                                rx.locs[0] = np.c_[locM[:, :2], z_RxM]
                                rx.locs[1] = np.c_[locN[:, :2], z_RxN]
                            else:
                                raise Exception()

            elif self.geometry == "BOREHOLE":
                raise Exception(
                    "Not implemented yet for BOREHOLE geometry"
                    )
            else:
                raise Exception(
                    "Input valid survey geometry: SURFACE or BOREHOLE"
                    )


class Survey_ky(Survey):
    """
    2.5D survey
    """
    rxPair = BaseRx
    srcPair = BaseSrc

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
