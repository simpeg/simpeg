import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import properties

import SimPEG
from SimPEG import Utils, Mesh
from . import SrcDC as Src   # Pole
from . import RxDC as Rx
from .SurveyDC import Survey_ky


class IO(properties.HasProperties):
    """Input and Output for DC, IP, SP, ..."""

    # Survey
    survey_geometry = properties.StringChoice(
        "Survey geometry of DC surveys",
        default="SURFACE",
        choices=["SURFACE", "BOREHOLE", "GENERAL"]
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

    # Data
    data_type = properties.StringChoice(
        "Type of DC survey",
        default="volt",
        choices=["volt", "appResistivity", "appConductivity"]
    )

    electrode_locations = properties.Array(
        "unique locations of a, b, m, n electrodes",
        shape=('*', '*'),  # ('*', 3) for 3D or ('*', 2) for 2D
        dtype=float  # data are floats
    )

    V = None
    appResistivity = None
    dobs = None
    G = None
    grids = None

    # Related to Physics and Discretization
    mesh = properties.Instance(
        "Mesh for discretization", Mesh.BaseMesh, required=True
    )

    dx = None
    dy = None
    dz = None

    npadx = 5
    npady = 5
    npadz = 5

    padratex = 1.3
    padratey = 1.3
    padratez = 1.3
    ncellperdipole = 4

    def from_ambnlocations_to_survey(
        self, a_locations, b_locations, m_locations, n_locations,
        surveyType, dobs=None,
        data_type="volt", fname=None, dim=2
    ):
        """
        read ABMN location and data (V or appResistivity)
        """
        self.a_locations = a_locations.copy()
        self.b_locations = b_locations.copy()
        self.m_locations = m_locations.copy()
        self.n_locations = n_locations.copy()
        self.surveyType = surveyType
        self.data_type = data_type
        self.dim = dim

        uniqSrc = Utils.uniqueRows(np.c_[self.a_locations, self.b_locations])
        uniqElec = SimPEG.Utils.uniqueRows(
            np.vstack((self.a_locations, self.b_locations, self.m_locations, self.n_locations))
            )
        self.electrode_locations = uniqElec[0]
        nSrc = uniqSrc[0].shape[0]
        ndata = self.a_locations.shape[0]

        if dim == 2:

            srcLists = []
            sortinds = []
            for iSrc in range (nSrc):
                inds = uniqSrc[2] == iSrc
                sortinds.append(np.arange(ndata)[inds])

                locsM = self.m_locations[inds, :]
                locsN = self.n_locations[inds, :]

                if (surveyType == 'dipole-dipole') or (surveyType == 'pole-dipole'):
                    rx = Rx.Dipole_ky(locsM, locsN)
                elif (surveyType == 'dipole-pole') or (surveyType == 'pole-pole'):
                    rx = Rx.Pole_ky(locsM)

                locA = uniqSrc[0][iSrc, :2]
                locB = uniqSrc[0][iSrc, 2:]

                if (surveyType == 'dipole-dipole') or (surveyType == 'dipole-pole'):
                    src = Src.Dipole([rx], locA, locB)
                elif (surveyType == 'pole-dipole') or (surveyType == 'pole-pole'):
                    src = Src.Pole([rx], locA)

                srcLists.append(src)

            self.sortinds = np.hstack(sortinds)
            survey = Survey_ky(srcLists)
            self.a_locations = self.a_locations[self.sortinds, :]
            self.b_locations = self.b_locations[self.sortinds, :]
            self.m_locations = self.m_locations[self.sortinds, :]
            self.n_locations = self.n_locations[self.sortinds, :]
            G = self.getGeometricFactor()
            if self.dobs is None:
                self.dobs = 100.*self.G
            else:
                self.dobs = dobs[self.sortinds]

            if self.data_type == "volt":
                self.V = self.dobs.copy()
                self.appResistivity = self.V / G
            elif self.data_type == "appResistivity":
                self.appResistivity = self.dobs.copy()
                self.V = self.appResistivity * self.G
                self.dobs = self.V.copy()

            midAB = (self.a_locations[:, 0] + self.b_locations[:, 0])*0.5
            midMN = (self.m_locations[:, 0] + self.n_locations[:, 0])*0.5
            z = abs(midAB-midMN)*1./3.
            x = (midAB+midMN)*0.5
            self.grids = np.c_[x, z]
        else:
            raise NotImplementedError()
        return survey

    def getGeometricFactor(self):

        if self.survey_geometry == 'SURFACE':

            if self.dim == 2:
                MA = abs(self.a_locations[:, 0] - self.m_locations[:, 0])
                MB = abs(self.b_locations[:, 0] - self.m_locations[:, 0])
                NA = abs(self.a_locations[:, 0] - self.n_locations[:, 0])
                NB = abs(self.b_locations[:, 0] - self.n_locations[:, 0])

            elif self.dim == 3:
                MA = np.sqrt(
                    abs(self.a_locations[:, 0] - self.m_locations[:, 0])**2. +
                    abs(self.a_locations[:, 1] - self.m_locations[:, 1])**2.
                    )
                MB = np.sqrt(
                    abs(self.b_locations[:, 0] - self.m_locations[:, 0])**2. +
                    abs(self.b_locations[:, 1] - self.m_locations[:, 1])**2.
                    )
                NA = np.sqrt(
                    abs(self.a_locations[:, 0] - self.n_locations[:, 0])**2. +
                    abs(self.a_locations[:, 1] - self.n_locations[:, 1])**2.
                    )
                NB = np.sqrt(
                    abs(self.b_locations[:, 0] - self.n_locations[:, 0])**2. +
                    abs(self.b_locations[:, 1] - self.n_locations[:, 1])**2.
                    )

            if self.surveyType == 'dipole-dipole':
                self.G = 1./(2*np.pi) * (1./MA - 1./MB + 1./NB - 1./NA)
            elif surveyType == 'pole-dipole':
                self.G = 1./(2*np.pi) * (1./MA - 1./NA)
            elif surveyType == 'dipole-pole':
                self.G = 1./(2*np.pi) * (1./MA - 1./MB)
            elif surveyType == 'pole-pole':
                self.G = 1./(2*np.pi) * (1./MA)

        elif self.survey_geometry == 'BOREHOLE':
            raise NotImplementedError()

        return self.G

    def setMesh(self, topo=None, dx=None, dz=None, nSpacing=None, corezlength=None, npadx=7, npadz=7, padratex=1.3, padratez=1.3, ncellperdipole=4, meshType='TensorMesh', dim=2):
        if meshType == 'TreeMesh':
            raise NotImplementedError()

        if dim == 2:
            a = abs(np.diff(np.sort(self.electrode_locations[:, 0]))).min()
            lineLength = abs(self.electrode_locations[:, 0].max()-self.electrode_locations[:, 0].min())
            dx_ideal = a/ncellperdipole
            if dx is None:
                dx = dx_ideal
            if dz is None:
                dz = dx*0.5
            x0 = self.electrode_locations[:, 0].min()
            if topo is None:
                locs = self.electrode_locations
            else:
                locs = np.vstack((topo, self.electrode_locations))
            zmax = locs[:, 1].max()
            zmin = locs[:, 1].min()
            if dx > dx_ideal:
                print (">>Input dx is greater than expected")
                print (
                    (": You may need %.1e m cell, that is %i cells per %.1e m dipole legnth") %
                    (dx_ideal, ncellperdipole, a)
                    )
            # TODO: conditional statement for dz?
            # Inject variables into the class
            self.dx = dx
            self.dz = dz
            self.npadx = npadx
            self.npadz = npadz
            self.padratex = padratex
            self.padratez = padratez
            self.ncellperdipole = ncellperdipole
            # 3 cells each for buffer
            corexlength = lineLength + dx * 6
            if corezlength is None:
                corezlength = self.grids[:, 1].max() + zmax - zmin

            ncx = np.floor(corexlength/dx)
            ncz = np.floor(corezlength/dz)
            hx = [(dx, npadx, -padratex), (dx, ncx), (dx, npadx, padratex)]
            hz = [(dz, npadz, -padratez), (dz, ncz)]
            x0_mesh = -(
                (dx * 1.3 ** (np.arange(npadx)+1)).sum() + dx * 3 - x0
                )
            z0_mesh = -((dz * 1.3 ** (np.arange(npadz)+1)).sum() + dz * ncz) + zmax
            mesh = Mesh.TensorMesh([hx, hz], x0=[x0_mesh, z0_mesh])
            actind = Utils.surface2ind_topo(mesh, locs)
            print (mesh)
        elif dim == 3:
            raise NotImplementedError()

        else:
            raise NotImplementedError()

        return mesh, actind

    def plotPseudoSection(self, data_type="appResistivity", scale="log", dataloc=True,aspect_ratio=2, cmap="jet", ncontour=10, ax=None, dobs=None, figname=None):
        matplotlib.rcParams['font.size'] = 12

        if dobs is None:
            dobs = self.dobs.copy()
            appResistivity = self.appResistivity.copy()
        else:
            G = self.getGeometricFactor()
            appResistivity = dobs / G

        if self.dim == 2:
            fig = plt.figure(figsize = (10, 5))
            if ax is None:
                ax = plt.subplot(111)
            if data_type == "appResistivity":
                val = appResistivity.copy()
                label = "Apparent Res. ($\Omega$m)"
            elif data_type == "volt":
                val = dobs.copy()
                label = "Voltage (V)"
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
                contourOpts={'cmap':cmap},
                ax = ax,
                dataloc=dataloc,
                scale=scale,
                ncontour=ncontour
                )
            ax.invert_yaxis()
            ax.set_xlabel("x (m)")
            ax.set_yticklabels([])
            ax.set_ylabel("n-spacing")
            cb = plt.colorbar(out[0], fraction=0.01, format=fmt)
            cb.set_label(label)
            ax.set_aspect(aspect_ratio)
            plt.tight_layout()
            if figname is not None:
                fig.savefig(figname, dpi=200)
            plt.show()

    def gen_locations_2D(self, surveyType, x0, lineLength, a, nSpacing):
        """
        genDClocs: Compute
        """
        self.surveyType = surveyType
        self.x0 = x0
        self.lineLength = lineLength
        self.a = a
        self.nSpacing = nSpacing

        nElec = int(np.floor(lineLength / a)) + 1
        xElec = x0 + np.arange(nElec)*a
        if surveyType == "dipole-dipole":
            SrcLoc = np.c_[xElec[:-1], xElec[1:]]
            RxLoc = np.c_[xElec[:-1], xElec[1:]]
            nSrc = SrcLoc.shape[0]
            nRx = RxLoc.shape[0]
            SrcID = []
            RxID = []
            nLeg = []
            for iSrc in range(nSrc-2):
                if nSrc-iSrc-1 > nSpacing:
                    nSounding = nSpacing
                else:
                    nSounding = nSrc - iSrc - 2
                SrcID.append(np.ones(nSounding, dtype=int) * iSrc)
                RxID.append(np.arange(nSounding) + iSrc + 2)
                nLeg.append(np.arange(nSounding)+1)
            # Set actual number of source dipole
            self.nSrc = nSrc-2

        elif surveyType == "pole-dipole":
            SrcLoc = np.c_[xElec, xElec]
            RxLoc = np.c_[xElec[:-1], xElec[1:]]
            nSrc = SrcLoc.shape[0]
            nRx = RxLoc.shape[0]
            SrcID = []
            RxID = []
            nLeg = []
            for iSrc in range(nSrc-2):
                if nSrc - iSrc - 2 > nSpacing:
                    nSounding = nSpacing
                else:
                    nSounding = nSrc - iSrc - 2
                SrcID.append(np.ones(nSounding, dtype=int) * iSrc)
                RxID.append(np.arange(nSounding) + iSrc + 1)
                nLeg.append(np.arange(nSounding)+1)
            # Set actual number of source pole
            self.nSrc = nSrc-2

        elif surveyType == "dipole-pole":
            SrcLoc = np.c_[xElec, xElec]
            RxLoc = np.c_[xElec[:-1], xElec[1:]]
            nSrc = SrcLoc.shape[0]
            nRx = RxLoc.shape[0]
            SrcID = []
            RxID = []
            nLeg = []
            for iSrc in range(nSrc-2):
                if nSrc - iSrc - 2 > nSpacing:
                    nSounding = nSpacing
                else:
                    nSounding = nSrc - iSrc - 2
                SrcID.append(np.ones(nSounding, dtype=int) * abs(iSrc-nSrc+1))
                RxID.append(abs(np.arange(nSounding) + iSrc + 1-nRx+1))
                nLeg.append(np.arange(nSounding)+1)
            self.nSrc = nSrc-2

        elif surveyType == "pole-pole":
            SrcLoc = np.c_[xElec, xElec]
            RxLoc = np.c_[xElec, xElec]
            nSrc = SrcLoc.shape[0]
            nRx = RxLoc.shape[0]
            SrcID = []
            RxID = []
            nLeg = []
            for iSrc in range(nSrc-1):
                if nSrc - iSrc - 1 > nSpacing:
                    nSounding = nSpacing
                else:
                    nSounding = nSrc - iSrc - 1
                SrcID.append(np.ones(nSounding, dtype=int) * iSrc)
                RxID.append(np.arange(nSounding) + iSrc +1)
                nLeg.append(np.arange(nSounding)+1)
            self.nSrc = nSrc-1

        else:
            raise NotImplementedError()

        SrcID = np.hstack(SrcID)
        RxID = np.hstack(RxID)
        nLeg = np.hstack(nLeg)

        if surveyType == "dipole-dipole":
            A = np.c_[SrcLoc[SrcID, 0], np.ones(SrcID.size)]
            B = np.c_[SrcLoc[SrcID, 1], np.ones(SrcID.size)]
            M = np.c_[RxLoc[RxID, 0], np.ones(SrcID.size)]
            N = np.c_[RxLoc[RxID, 1], np.ones(SrcID.size)]
        elif surveyType == "pole-dipole":
            A = np.c_[SrcLoc[SrcID], np.ones(SrcID.size)]
            B = np.c_[SrcLoc[SrcID], np.ones(SrcID.size)]
            M = np.c_[RxLoc[RxID, 0], np.ones(SrcID.size)]
            N = np.c_[RxLoc[RxID, 1], np.ones(SrcID.size)]
        elif surveyType == "dipole-pole":
            A = np.c_[SrcLoc[SrcID, 0], np.ones(SrcID.size)]
            B = np.c_[SrcLoc[SrcID, 1], np.ones(SrcID.size)]
            M = np.c_[RxLoc[RxID], np.ones(SrcID.size)]
            N = np.c_[RxLoc[RxID], np.ones(SrcID.size)]
        elif surveyType == "pole-pole":
            A = np.c_[SrcLoc[SrcID], np.ones(SrcID.size)]
            B = np.c_[SrcLoc[SrcID], np.ones(SrcID.size)]
            M = np.c_[RxLoc[RxID], np.ones(SrcID.size)]
            N = np.c_[RxLoc[RxID], np.ones(SrcID.size)]
        else:
            raise NotImplementedError()
        return A, B, M, N

