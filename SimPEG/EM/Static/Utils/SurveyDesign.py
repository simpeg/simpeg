from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from SimPEG import Utils, Mesh
from SimPEG.EM.Static import DC
import matplotlib.pyplot as plt


class SurveyDesign(object):
    """docstring for SurveyDesign"""
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
    survey = None
    problem = None
    dpred = None
    F = None

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

        self.SrcID = np.hstack(SrcID)
        self.RxID = np.hstack(RxID)
        self.nLeg = np.hstack(nLeg)
        self.SrcLoc = SrcLoc
        self.RxLoc = RxLoc
        self.xElec = xElec
        self.nSrc = nSrc

    def plot2Dgeometry(self, iSrc=0):
        fig = plt.figure(figsize=(8, 3))
        xloc = (
            self.SrcLoc[self.SrcID, 0] + self.SrcLoc[self.SrcID, 1] +
            self.RxLoc[self.RxID, 0] + self.RxLoc[self.RxID, 1]
            ) * 0.25
        zloc = self.nLeg
        x_temp = (
            self.SrcLoc[self.SrcID[self.SrcID==iSrc], 0] +
            self.SrcLoc[self.SrcID[self.SrcID==iSrc], 1] +
            self.RxLoc[self.RxID[self.SrcID==iSrc], 0] +
            self.RxLoc[self.RxID[self.SrcID==iSrc], 1]
            ) * 0.25
        z_temp = self.nLeg[self.SrcID == iSrc]
        plt.plot(
            self.SrcLoc[self.SrcID[self.SrcID==iSrc], 0],
            np.zeros_like(self.SrcLoc[self.SrcID[self.SrcID == iSrc], 0]), 'yv'
            )
        plt.plot(
            self.SrcLoc[self.SrcID[self.SrcID == iSrc], 1],
            np.zeros_like(self.SrcLoc[self.SrcID[self.SrcID == iSrc], 1]), 'gv'
            )
        plt.plot(
            self.RxLoc[self.RxID[self.SrcID == iSrc], 0][0],
            np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 0][0]), 'rv'
            )
        plt.plot(
            self.RxLoc[self.RxID[self.SrcID == iSrc], 1][0],
            np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 1][0]), 'bv'
            )
        plt.legend(("A", "B", "M", "N"))
        plt.plot(
            self.RxLoc[self.RxID[self.SrcID == iSrc], 0],
            np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 0]), 'rv'
            )
        plt.plot(
            self.RxLoc[self.RxID[self.SrcID == iSrc], 1],
            np.zeros_like(self.RxLoc[self.RxID[self.SrcID == iSrc], 1]), 'bv'
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

    def setMesh_2D(self):
        self.mesh = Mesh.TensorMesh([hx, hy], x0="CN")

    def genDCSurvey_2D(self, problemType="2.5D"):
        srcList = []
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
            elif (self.surveyType == "pole-dipole") or (self.surveyType == "dipole-pole"):
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
            elif self.surveyType == "pole-pole":
                Mx = self.RxLoc[self.RxID[self.SrcID == iSrc], 0]
                Ax = self.SrcLoc[iSrc, 0]
                # TODO: Drape topography?
                M = np.c_[Mx, np.zeros_like(Mx)]
                if problemType == "2.5D":
                    rx = DC.Rx.Pole_ky(M)
                else:
                    rx = DC.Rx.Pole(M, N)
                src = DC.Src.Pole([rx], A)
            else:
                raise Exception("Input valid surveyType!")
            srcList.append(src)
        self.survey = DC.Survey_ky(srcList)

    def runForward(self, sigma, problemType = "2.5D_N"):
        if problemType == "2.5D_N":
            self.problem = DC.Problem2D_N(mesh, sigma=sigma)
        elif problemType == "3D_N":
            self.problem = DC.Problem3D_N(mesh, sigma=sigma)
        self.survey.pair(self.problem)
        self.F = self.problem.fields(sigma)
        self.dpred = self.survey.dpred(sigma, f=F)

