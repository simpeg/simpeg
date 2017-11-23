from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from SimPEG import Utils, Mesh
from SimPEG.EM.Static import DC
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pymatsolver import PardisoSolver
from matplotlib.colors import LogNorm, SymLogNorm


class SurveyDesign(object):
    """docstring for SurveyDesign"""

    # TODO: Use properties package
    surveyType = None
    x0 = None
    lineLength = None
    a = None
    nSpacing = None
    SrcID = None
    SrcLoc = None
    RxID = None
    RxLoc = None
    nLeg = None
    G = None
    survey = None
    problem = None
    dpred = None
    F = None
    # Mesh
    dx = None
    dy = None
    npadx = None
    npady = None
    padratex = None
    padratey = None
    ncellperdipole = None

    def getGeometricFactor_2D(self):
        rAM = abs(self.SrcLoc[self.SrcID, 0]-self.RxLoc[self.RxID, 0])
        rAN = abs(self.SrcLoc[self.SrcID, 0]-self.RxLoc[self.RxID, 1])
        rBM = abs(self.SrcLoc[self.SrcID, 1]-self.RxLoc[self.RxID, 0])
        rBN = abs(self.SrcLoc[self.SrcID, 1]-self.RxLoc[self.RxID, 1])

        if self.surveyType == 'dipole-dipole':
            G = (1./rAM - 1./rAN) - (1./rBM - 1./rBN)
        elif (self.surveyType == 'pole-dipole') or (self.surveyType == 'dipole-pole'):
            G = (1./rAM - 1./rAN)
        elif self.surveyType == 'pole-pole':
            G = 1./rAM
        # Not sure why ... there is a coherent error
        factor = 1.11
        self.G = G

    def genLocs_2D(self, surveyType, x0, lineLength, a, nSpacing):
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

        if surveyType == "pole-dipole":
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

        if surveyType == "dipole-pole":
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

        if surveyType == "pole-pole":
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

        self.SrcID = np.hstack(SrcID)
        self.RxID = np.hstack(RxID)
        self.nLeg = np.hstack(nLeg)
        self.SrcLoc = SrcLoc
        self.RxLoc = RxLoc
        self.xElec = xElec

    def plot2Dgeometry(self, iSrc=0, showIt=True):
        if showIt:
            fig = plt.figure(figsize=(8, 3))
            xloc = (
                self.SrcLoc[self.SrcID, 0] + self.SrcLoc[self.SrcID, 1] +
                self.RxLoc[self.RxID, 0] + self.RxLoc[self.RxID, 1]
                ) * 0.25
            zloc = self.nLeg
            x_temp = (
                self.SrcLoc[self.SrcID[self.SrcID == iSrc], 0] +
                self.SrcLoc[self.SrcID[self.SrcID == iSrc], 1] +
                self.RxLoc[self.RxID[self.SrcID == iSrc], 0] +
                self.RxLoc[self.RxID[self.SrcID == iSrc], 1]
                ) * 0.25
            z_temp = self.nLeg[self.SrcID == iSrc]
            plt.plot(
                self.SrcLoc[self.SrcID[self.SrcID == iSrc], 0],
                np.zeros_like(self.SrcLoc[self.SrcID[self.SrcID == iSrc], 0]),
                'yv'
                )
            plt.plot(
                self.SrcLoc[self.SrcID[self.SrcID == iSrc], 1],
                np.zeros_like(self.SrcLoc[self.SrcID[self.SrcID == iSrc], 1]),
                'gv'
                )
            plt.plot(
                self.RxLoc[self.RxID[self.SrcID == iSrc], 0][0],
                np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 0][0]),
                'rv'
                )
            plt.plot(
                self.RxLoc[self.RxID[self.SrcID == iSrc], 1][0],
                np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 1][0]),
                'bv'
                )
            plt.legend(("A", "B", "M", "N"))
            plt.plot(
                self.RxLoc[self.RxID[self.SrcID == iSrc], 0],
                np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 0]),
                'rv'
                )
            plt.plot(
                self.RxLoc[self.RxID[self.SrcID == iSrc], 1],
                np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 1]),
                'bv'
                )
            plt.plot(x_temp, z_temp, 'ro')
            plt.plot(xloc, zloc, 'k.')
            for x in self.xElec:
                plt.plot(np.ones(2)*x, np.r_[0, -0.6], 'k-', alpha=0.3)
            plt.plot(
                np.r_[self.xElec.min(), self.xElec.max()],
                np.r_[-0.6, -0.6], 'k-', alpha=0.3
                )
            plt.ylabel("n-Spacing")
            plt.gca().invert_yaxis()
            plt.show()

    def setMesh_2D(self, dx, dz, npadx=5, npadz=5, padratex=1.3, padratez=1.3, ncellperdipole=4):
        dx_ideal = self.a/ncellperdipole
        if dx > dx_ideal:
            print (">>Input dx is greater than expected")
            print (
                (": You may need %.1e m cell, that is %i cells per %.1e m dipole legnth") %
                (dx_ideal, ncellperdipole, self.a)
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
        corexlength = self.lineLength + dx * 6
        # Use nPacing x a /2 to compute coredepth
        corezlegnth = self.nSpacing * self.a / 2.
        x0core = self.x0 - dx * 3
        self.ncx = np.floor(corexlength/dx)
        self.ncz = np.floor(corezlegnth/dz)
        hx = [(dx, npadx, -padratex), (dx, self.ncx), (dx, npadz, padratex)]
        hz = [(dz, npadz, -padratez), (dz, self.ncz)]
        x0_mesh = -(
            (dx * 1.3 ** (np.arange(npadx)+1)).sum() + dx * 3 - self.x0
            )
        z0_mesh = -((dz * 1.3 ** (np.arange(npadz)+1)).sum() + dz * self.ncz)
        self.mesh = Mesh.TensorMesh([hx, hz], x0=[x0_mesh, z0_mesh])

    def genDCSurvey_2D(self, problemType="2.5D"):
        srcList = []
        nSrc = self.SrcLoc.shape[0]
        for iSrc in range(self.nSrc):
            if self.surveyType == "dipole-dipole":
                Mx = self.RxLoc[self.RxID[self.SrcID == iSrc], 0]
                Nx = self.RxLoc[self.RxID[self.SrcID == iSrc], 1]
                Ax = self.SrcLoc[iSrc, 0]
                Bx = self.SrcLoc[iSrc, 1]
                # TODO: Drape topography?
                M = np.c_[Mx, np.zeros_like(Mx)]
                N = np.c_[Nx, np.zeros_like(Nx)]
                A = np.r_[Ax, 0.].reshape([1, -1])
                B = np.r_[Bx, 0.].reshape([1, -1])
                if problemType == "2.5D":
                    rx = DC.Rx.Dipole_ky(M, N)
                else:
                    rx = DC.Rx.Dipole(M, N)
                src = DC.Src.Dipole([rx], A, B)
            elif self.surveyType == "pole-dipole":
                Mx = self.RxLoc[self.RxID[self.SrcID == iSrc], 0]
                Nx = self.RxLoc[self.RxID[self.SrcID == iSrc], 1]
                Ax = self.SrcLoc[iSrc, 0]
                # TODO: Drape topography?
                M = np.c_[Mx, np.zeros_like(Mx)]
                N = np.c_[Nx, np.zeros_like(Nx)]
                A = np.r_[Ax, 0.].reshape([1, -1])
                if problemType == "2.5D":
                    rx = DC.Rx.Dipole_ky(M, N)
                else:
                    rx = DC.Rx.Dipole(M, N)
                src = DC.Src.Pole([rx], A)
            elif self.surveyType == "dipole-pole":
                Mx = self.RxLoc[self.RxID[self.SrcID == abs(iSrc-nSrc+1)], 0]
                Nx = self.RxLoc[self.RxID[self.SrcID == abs(iSrc-nSrc+1)], 1]
                Ax = self.SrcLoc[abs(iSrc-nSrc+1), 0]
                # TODO: Drape topography?
                M = np.c_[Mx, np.zeros_like(Mx)]
                N = np.c_[Nx, np.zeros_like(Nx)]
                A = np.r_[Ax, 0.].reshape([1, -1])
                if problemType == "2.5D":
                    rx = DC.Rx.Dipole_ky(M, N)
                else:
                    rx = DC.Rx.Dipole(M, N)
                src = DC.Src.Pole([rx], A)

            elif self.surveyType == "pole-pole":
                Mx = self.RxLoc[self.RxID[self.SrcID == iSrc], 0]
                Ax = self.SrcLoc[iSrc, 0]
                # TODO: Drape topography?
                M = np.c_[Mx, np.zeros_like(Mx)]
                A = np.r_[Ax, 0.].reshape([1, -1])
                if problemType == "2.5D":
                    rx = DC.Rx.Pole_ky(M)
                else:
                    rx = DC.Rx.Pole(M, N)
                src = DC.Src.Pole([rx], A)
            else:
                raise Exception("Input valid surveyType!")
            srcList.append(src)
        self.survey = DC.Survey_ky(srcList)

    def runForwardSimulation(self, sigma, problemType="2.5D_N"):
        if problemType == "2.5D_N":
            self.problem = DC.Problem2D_N(self.mesh, sigma=sigma)
        elif problemType == "2.5D_CC":
            self.problem = DC.Problem2D_CC(self.mesh, sigma=sigma)
        elif problemType == "3D_N":
            self.problem = DC.Problem3D_N(self.mesh, sigma=sigma)
        elif problemType == "3D_CC":
            self.problem = DC.Problem3D_CC(self.mesh, sigma=sigma)
        # self.problem.Solver = PardisoSolver
        self.survey.pair(self.problem)
        f = self.problem.fields(sigma)
        self.F = self.problem.fields_to_space(f)
        self.voltage = self.survey.dpred(sigma, f=f)
        self.getGeometricFactor_2D()
        self.appResistivity = self.voltage / self.G * np.pi * 2.

    def plotData_2D(self, dataType="appResistivity", ms=100, showIt=True, scale='linear', clim=None, pcolorOpts={}):
        if showIt:
            fig = plt.figure(figsize=(8, 3))
            xloc = (
                self.SrcLoc[self.SrcID, 0] + self.SrcLoc[self.SrcID, 1] +
                self.RxLoc[self.RxID, 0] + self.RxLoc[self.RxID, 1]
                ) * 0.25
            zloc = self.nLeg
            if dataType == "appResistivity":
                val = self.appResistivity.copy()
            elif dataType == "voltage":
                val = self.appResistivity.copy()
            else:
                raise Exception("Input valid dataType")

            if scale == "log":
                val = np.log10(abs(val))

            if clim is None:
                vmin, vmax = val.min(), val.max()
                clim = vmin, vmax
            # Grid points
            print (clim)
            out = plt.scatter(xloc, zloc, c=val, s=ms, clim=clim, vmin=clim[0], vmax=clim[1])
            cb = plt.colorbar(out)

            for x in self.xElec:
                plt.plot(np.ones(2)*x, np.r_[0, -0.6], 'k-', alpha=0.3)
            plt.plot(
                np.r_[self.xElec.min(), self.xElec.max()],
                np.r_[-0.6, -0.6], 'k-', alpha=0.3
                )
            plt.title("appResistivity")
            plt.ylabel("n-Spacing")
            plt.gca().invert_yaxis()
            plt.show()

    def plotfields_2D(self, iSrc=0, fieldType="E", showIt=True):
        if showIt:
            fig = plt.figure(figsize=(8, 2))
            if self.problem._formulation == "HJ":
                if fieldType == "E":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'e'], vType='F',
                        view='vec',
                        streamOpts={'color': 'w'}, pcolorOpts={"norm": LogNorm()}
                        )
                elif fieldType == "J":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'j'], vType='F',
                        view='vec',
                        streamOpts={'color': 'w'}, pcolorOpts={"norm": LogNorm()}
                        )
                elif fieldType == "Phi":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'phi'], vType='CC',
                        pcolorOpts={"norm": SymLogNorm(1e-10)}
                        )
                elif fieldType == "Charge":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'charge'], vType='CC',
                        pcolorOpts={"norm": SymLogNorm(1e-10)}
                        )
            elif self.problem._formulation == "EB":
                if fieldType == "E":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'e'], vType='E',
                        view='vec',
                        streamOpts={'color': 'w'}, pcolorOpts={"norm": LogNorm()}
                        )
                elif fieldType == "J":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'j'], vType='E',
                        view='vec',
                        streamOpts={'color': 'w'}, pcolorOpts={"norm": LogNorm()}
                        )
                elif fieldType == "Phi":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'phi'], vType='N',
                        pcolorOpts={"norm": SymLogNorm(1e-10)}
                        )
                elif fieldType == "Charge":
                    self.mesh.plotImage(
                        self.F[self.survey.srcList[iSrc], 'charge'], vType='N',
                        pcolorOpts={"norm": SymLogNorm(1e-10)}
                        )

            xlim = self.x0, self.lineLength
            zlim = -self.dz * self.ncz, 0.
            plt.xlim(xlim)
            plt.ylim(zlim)
            plt.gca().set_aspect("equal")
            plt.show()


def genTopography(mesh, zmin, zmax, seed=None, its=100, anisotropy=None):
    if mesh.dim == 3:
        hx = mesh.hx
        hy = mesh.hy
        mesh2D = Mesh.TensorMesh(
            [mesh.hx, mesh.hy], x0 = [mesh.x0[0], mesh.x0[1]]
            )
        out = Utils.ModelBuilder.randomModel(
            mesh.vnC[:2], bounds=[zmin, zmax], its=its,
            seed=seed, anisotropy=anisotropy
            )
        return out, mesh2D
    elif mesh.dim == 2:
        hx = mesh.hx
        mesh1D = Mesh.TensorMesh([mesh.hx], x0 = [mesh.x0[0]])
        out = Utils.ModelBuilder.randomModel(
            mesh.vnC[:1], bounds=[zmin, zmax], its=its,
            seed=seed, anisotropy=anisotropy
            )
        return out, mesh1D
    else:
        raise Exception("Only works for 2D and 3D models")
