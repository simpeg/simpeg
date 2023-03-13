import discretize
import numpy as np
import scipy.sparse as sp

from ...simulation import BaseSimulation
from ... import props
from ... import maps
from ...utils import (
    mkvc,
    validate_type,
    validate_ndarray_with_shape,
    validate_active_indices,
)

from .survey import SurveyVRM
from .receivers import Point, SquareLoop


############################################
# BASE VRM PROBLEM CLASS
############################################


class BaseVRMSimulation(BaseSimulation):
    """"""

    _AisSet = False

    def __init__(
        self,
        mesh,
        survey=None,
        refinement_factor=None,
        refinement_distance=None,
        indActive=None,
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)

        if refinement_distance is None:
            if refinement_factor is None:
                refinement_factor = 3
            refinement_distance = (
                1.25
                * np.mean(
                    np.r_[np.min(mesh.h[0]), np.min(mesh.h[1]), np.min(mesh.h[2])]
                )
                * np.arange(1, refinement_factor + 1)
            )
        self.refinement_distance = refinement_distance
        if indActive is None:
            indActive = np.ones(self.mesh.n_cells, dtype=bool)
        self.indActive = indActive

    @BaseSimulation.mesh.setter
    def mesh(self, value):
        value = validate_type(
            "mesh", value, (discretize.TensorMesh, discretize.TreeMesh), cast=False
        )
        if value.dim != 3:
            raise ValueError(
                f"Mesh must be 3D tensor or 3D tree. Current mesh is {value.dim}"
            )
        self._mesh = value

    @property
    def survey(self):
        """The VRM survey.

        Returns
        -------
        SimPEG.electromagnetics.viscous_temanent_magnetization.survey.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey.")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, SurveyVRM, cast=False)
        self._survey = value

    @property
    def refinement_factor(self):
        """The number of refinement distances.

        Returns
        -------
        int
        """
        return len(self.refinement_distance)

    @property
    def refinement_distance(self):
        """Sensitivity refinement radii from sources.

        Returns
        -------
        numpy.ndarray
        """
        return self._refinement_distance

    @refinement_distance.setter
    def refinement_distance(self, value):
        self._refinement_distance = validate_ndarray_with_shape(
            "refinement_distance", value, shape=("*",)
        )

    @property
    def indActive(self):
        """Topography active cells.

        Returns
        -------
        (mesh.n_cells) numpy.ndarray of bool
        """
        return self._indActive

    @indActive.setter
    def indActive(self, value):
        self._indActive = validate_active_indices("indActive", value, self.mesh.n_cells)

    def _getH0matrix(self, xyz, pp):
        """
                Creates sparse matrix containing inducing field components
                for source pp

        ..        REQUIRED ARGUMENTS:
        ..
        ..        xyz: N X 3 array of locations to predict field
        ..
        ..        pp: Source index
        ..
        ..        OUTPUTS:
        ..
        ..        H0: A 3N X N sparse array containing Hx, Hy and Hz at all locations
        ..
        """

        srcObj = self.survey.source_list[pp]

        h0 = srcObj.getH0(xyz)

        hx0 = sp.diags(h0[:, 0], format="csr")
        hy0 = sp.diags(h0[:, 1], format="csr")
        hz0 = sp.diags(h0[:, 2], format="csr")

        h0 = sp.vstack([hx0, hy0, hz0])

        return h0

    def _getGeometryMatrix(self, xyzc, xyzh, pp):
        """
                Creates the dense geometry matrix which maps from the magnetized voxel
                cells to the receiver locations for source pp
        ..
        ..        REQUIRED ARGUMENTS:
        ..
        ..        xyzc: N by 3 numpy array containing cell center locations [xc,yc,zc]
        ..
        ..        xyzh: N by 3 numpy array containing cell dimensions [hx,hy,hz]
        ..
        ..        pp: Source index
        ..
        ..        OUTPUTS:
        ..
        ..        G: Linear geometry operator

        """

        srcObj = self.survey.source_list[pp]

        nC = np.shape(xyzc)[0]  # Number of cells
        nRx = srcObj.nRx  # Number of receiver in all receiver_list

        ax = np.reshape(xyzc[:, 0] - xyzh[:, 0] / 2, (1, nC))
        bx = np.reshape(xyzc[:, 0] + xyzh[:, 0] / 2, (1, nC))
        ay = np.reshape(xyzc[:, 1] - xyzh[:, 1] / 2, (1, nC))
        by = np.reshape(xyzc[:, 1] + xyzh[:, 1] / 2, (1, nC))
        az = np.reshape(xyzc[:, 2] - xyzh[:, 2] / 2, (1, nC))
        bz = np.reshape(xyzc[:, 2] + xyzh[:, 2] / 2, (1, nC))

        G = np.zeros((nRx, 3 * nC))
        c = -(1 / (4 * np.pi))
        tol = 1e-10  # Tolerance constant for numerical stability
        tol2 = 1000.0  # Tolerance constant for numerical stability

        COUNT = 0

        for qq in range(0, len(srcObj.receiver_list)):
            rxObj = srcObj.receiver_list[qq]
            dComp = rxObj.orientation
            locs = rxObj.locations
            nLoc = np.shape(locs)[0]

            if isinstance(rxObj, Point) and not isinstance(rxObj, SquareLoop):
                if dComp.lower() == "x":
                    for rr in range(0, nLoc):
                        u1 = locs[rr, 0] - ax
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0]) / tol2
                        u2 = locs[rr, 0] - bx
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0]) / tol2
                        v1 = locs[rr, 1] - ay
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1]) / tol2
                        v2 = locs[rr, 1] - by
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1]) / tol2
                        w1 = locs[rr, 2] - az
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2]) / tol2
                        w2 = locs[rr, 2] - bz
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2]) / tol2

                        d111 = np.sqrt(u1**2 + v1**2 + w1**2)
                        d211 = np.sqrt(u2**2 + v1**2 + w1**2)
                        d221 = np.sqrt(u2**2 + v2**2 + w1**2)
                        d121 = np.sqrt(u1**2 + v2**2 + w1**2)
                        d122 = np.sqrt(u1**2 + v2**2 + w2**2)
                        d112 = np.sqrt(u1**2 + v1**2 + w2**2)
                        d212 = np.sqrt(u2**2 + v1**2 + w2**2)
                        d222 = np.sqrt(u2**2 + v2**2 + w2**2)

                        Gxx = (
                            np.arctan((v1 * w1) / (u1 * d111 + tol))
                            - np.arctan((v1 * w1) / (u2 * d211 + tol))
                            + np.arctan((v2 * w1) / (u2 * d221 + tol))
                            - np.arctan((v2 * w1) / (u1 * d121 + tol))
                            + np.arctan((v2 * w2) / (u1 * d122 + tol))
                            - np.arctan((v1 * w2) / (u1 * d112 + tol))
                            + np.arctan((v1 * w2) / (u2 * d212 + tol))
                            - np.arctan((v2 * w2) / (u2 * d222 + tol))
                        )

                        Gyx = (
                            np.log(d111 - w1)
                            - np.log(d211 - w1)
                            + np.log(d221 - w1)
                            - np.log(d121 - w1)
                            + np.log(d122 - w2)
                            - np.log(d112 - w2)
                            + np.log(d212 - w2)
                            - np.log(d222 - w2)
                        )

                        Gzx = (
                            np.log(d111 - v1)
                            - np.log(d211 - v1)
                            + np.log(d221 - v2)
                            - np.log(d121 - v2)
                            + np.log(d122 - v2)
                            - np.log(d112 - v1)
                            + np.log(d212 - v1)
                            - np.log(d222 - v2)
                        )

                        G[COUNT, :] = c * np.c_[Gxx, Gyx, Gzx]
                        COUNT = COUNT + 1

                elif dComp.lower() == "y":
                    for rr in range(0, nLoc):
                        u1 = locs[rr, 0] - ax
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0]) / tol2
                        u2 = locs[rr, 0] - bx
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0]) / tol2
                        v1 = locs[rr, 1] - ay
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1]) / tol2
                        v2 = locs[rr, 1] - by
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1]) / tol2
                        w1 = locs[rr, 2] - az
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2]) / tol2
                        w2 = locs[rr, 2] - bz
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2]) / tol2

                        d111 = np.sqrt(u1**2 + v1**2 + w1**2)
                        d211 = np.sqrt(u2**2 + v1**2 + w1**2)
                        d221 = np.sqrt(u2**2 + v2**2 + w1**2)
                        d121 = np.sqrt(u1**2 + v2**2 + w1**2)
                        d122 = np.sqrt(u1**2 + v2**2 + w2**2)
                        d112 = np.sqrt(u1**2 + v1**2 + w2**2)
                        d212 = np.sqrt(u2**2 + v1**2 + w2**2)
                        d222 = np.sqrt(u2**2 + v2**2 + w2**2)

                        Gxy = (
                            np.log(d111 - w1)
                            - np.log(d211 - w1)
                            + np.log(d221 - w1)
                            - np.log(d121 - w1)
                            + np.log(d122 - w2)
                            - np.log(d112 - w2)
                            + np.log(d212 - w2)
                            - np.log(d222 - w2)
                        )

                        Gyy = (
                            np.arctan((u1 * w1) / (v1 * d111 + tol))
                            - np.arctan((u2 * w1) / (v1 * d211 + tol))
                            + np.arctan((u2 * w1) / (v2 * d221 + tol))
                            - np.arctan((u1 * w1) / (v2 * d121 + tol))
                            + np.arctan((u1 * w2) / (v2 * d122 + tol))
                            - np.arctan((u1 * w2) / (v1 * d112 + tol))
                            + np.arctan((u2 * w2) / (v1 * d212 + tol))
                            - np.arctan((u2 * w2) / (v2 * d222 + tol))
                        )

                        Gzy = (
                            np.log(d111 - u1)
                            - np.log(d211 - u2)
                            + np.log(d221 - u2)
                            - np.log(d121 - u1)
                            + np.log(d122 - u1)
                            - np.log(d112 - u1)
                            + np.log(d212 - u2)
                            - np.log(d222 - u2)
                        )

                        G[COUNT, :] = c * np.c_[Gxy, Gyy, Gzy]
                        COUNT = COUNT + 1

                elif dComp.lower() == "z":
                    for rr in range(0, nLoc):
                        u1 = locs[rr, 0] - ax
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0]) / tol2
                        u2 = locs[rr, 0] - bx
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0]) / tol2
                        v1 = locs[rr, 1] - ay
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1]) / tol2
                        v2 = locs[rr, 1] - by
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1]) / tol2
                        w1 = locs[rr, 2] - az
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2]) / tol2
                        w2 = locs[rr, 2] - bz
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2]) / tol2

                        d111 = np.sqrt(u1**2 + v1**2 + w1**2)
                        d211 = np.sqrt(u2**2 + v1**2 + w1**2)
                        d221 = np.sqrt(u2**2 + v2**2 + w1**2)
                        d121 = np.sqrt(u1**2 + v2**2 + w1**2)
                        d122 = np.sqrt(u1**2 + v2**2 + w2**2)
                        d112 = np.sqrt(u1**2 + v1**2 + w2**2)
                        d212 = np.sqrt(u2**2 + v1**2 + w2**2)
                        d222 = np.sqrt(u2**2 + v2**2 + w2**2)

                        Gxz = (
                            np.log(d111 - v1)
                            - np.log(d211 - v1)
                            + np.log(d221 - v2)
                            - np.log(d121 - v2)
                            + np.log(d122 - v2)
                            - np.log(d112 - v1)
                            + np.log(d212 - v1)
                            - np.log(d222 - v2)
                        )

                        Gyz = (
                            np.log(d111 - u1)
                            - np.log(d211 - u2)
                            + np.log(d221 - u2)
                            - np.log(d121 - u1)
                            + np.log(d122 - u1)
                            - np.log(d112 - u1)
                            + np.log(d212 - u2)
                            - np.log(d222 - u2)
                        )

                        Gzz = (
                            -np.arctan((v1 * w1) / (u1 * d111 + tol))
                            + np.arctan((v1 * w1) / (u2 * d211 + tol))
                            - np.arctan((v2 * w1) / (u2 * d221 + tol))
                            + np.arctan((v2 * w1) / (u1 * d121 + tol))
                            - np.arctan((v2 * w2) / (u1 * d122 + tol))
                            + np.arctan((v1 * w2) / (u1 * d112 + tol))
                            - np.arctan((v1 * w2) / (u2 * d212 + tol))
                            + np.arctan((v2 * w2) / (u2 * d222 + tol))
                        )

                        Gzz = (
                            Gzz
                            - np.arctan((u1 * w1) / (v1 * d111 + tol))
                            + np.arctan((u2 * w1) / (v1 * d211 + tol))
                            - np.arctan((u2 * w1) / (v2 * d221 + tol))
                            + np.arctan((u1 * w1) / (v2 * d121 + tol))
                            - np.arctan((u1 * w2) / (v2 * d122 + tol))
                            + np.arctan((u1 * w2) / (v1 * d112 + tol))
                            - np.arctan((u2 * w2) / (v1 * d212 + tol))
                            + np.arctan((u2 * w2) / (v2 * d222 + tol))
                        )

                        G[COUNT, :] = c * np.c_[Gxz, Gyz, Gzz]
                        COUNT = COUNT + 1

            elif isinstance(rxObj, SquareLoop):
                # Gaussian quadrature weights
                wt = [
                    np.r_[2.0],
                    np.r_[1.0, 1.0],
                    np.r_[0.555556, 0.888889, 0.555556],
                    np.r_[0.347855, 0.652145, 0.652145, 0.347855],
                    np.r_[0.236927, 0.478629, 0.568889, 0.478629, 0.236927],
                    np.r_[0.171324, 0.467914, 0.360762, 0.360762, 0.467914, 0.171324],
                    np.r_[
                        0.129485,
                        0.279705,
                        0.381830,
                        0.417959,
                        0.381830,
                        0.279705,
                        0.129485,
                    ],
                ]
                wt = wt[rxObj.quadrature_order - 1]
                nw = len(wt)
                wt = (
                    rxObj.n_turns
                    * (rxObj.width / 2) ** 2
                    * np.reshape(np.outer(wt, wt), (1, nw**2))
                )

                # Gaussian quadrature locations on [-1,1]
                ds = [
                    np.r_[0.0],
                    np.r_[-0.57735, 0.57735],
                    np.r_[-0.774597, 0.0, 0.774597],
                    np.r_[-0.861136, -0.339981, 0.339981, 0.861136],
                    np.r_[-0.906180, -0.538469, 0, 0.538469, 0.906180],
                    np.r_[
                        -0.932470, -0.238619, -0.661209, 0.661209, 0.238619, 0.932470
                    ],
                    np.r_[
                        -0.949108,
                        -0.741531,
                        -0.405845,
                        0.0,
                        0.405845,
                        0.741531,
                        0.949108,
                    ],
                ]

                s1 = (
                    0.5
                    * rxObj.width
                    * np.reshape(
                        np.kron(ds[rxObj.quadrature_order - 1], np.ones(nw)),
                        (nw**2, 1),
                    )
                )
                s2 = (
                    0.5
                    * rxObj.width
                    * np.reshape(
                        np.kron(np.ones(nw), ds[rxObj.quadrature_order - 1]),
                        (nw**2, 1),
                    )
                )

                if dComp.lower() == "x":
                    for rr in range(0, nLoc):
                        u1 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax)
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0]) / tol2
                        u2 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx)
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0]) / tol2

                        v1 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay) + np.kron(
                            s1, np.ones((1, nC))
                        )
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1]) / tol2
                        v2 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by) + np.kron(
                            s1, np.ones((1, nC))
                        )
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1]) / tol2

                        w1 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az) + np.kron(
                            s2, np.ones((1, nC))
                        )
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2]) / tol2
                        w2 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz) + np.kron(
                            s2, np.ones((1, nC))
                        )
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2]) / tol2

                        d111 = np.sqrt(u1**2 + v1**2 + w1**2)
                        d211 = np.sqrt(u2**2 + v1**2 + w1**2)
                        d221 = np.sqrt(u2**2 + v2**2 + w1**2)
                        d121 = np.sqrt(u1**2 + v2**2 + w1**2)
                        d122 = np.sqrt(u1**2 + v2**2 + w2**2)
                        d112 = np.sqrt(u1**2 + v1**2 + w2**2)
                        d212 = np.sqrt(u2**2 + v1**2 + w2**2)
                        d222 = np.sqrt(u2**2 + v2**2 + w2**2)

                        Gxx = (
                            np.arctan((v1 * w1) / (u1 * d111 + tol))
                            - np.arctan((v1 * w1) / (u2 * d211 + tol))
                            + np.arctan((v2 * w1) / (u2 * d221 + tol))
                            - np.arctan((v2 * w1) / (u1 * d121 + tol))
                            + np.arctan((v2 * w2) / (u1 * d122 + tol))
                            - np.arctan((v1 * w2) / (u1 * d112 + tol))
                            + np.arctan((v1 * w2) / (u2 * d212 + tol))
                            - np.arctan((v2 * w2) / (u2 * d222 + tol))
                        )

                        Gxx = np.dot(wt, Gxx)

                        Gyx = (
                            np.log(d111 - w1)
                            - np.log(d211 - w1)
                            + np.log(d221 - w1)
                            - np.log(d121 - w1)
                            + np.log(d122 - w2)
                            - np.log(d112 - w2)
                            + np.log(d212 - w2)
                            - np.log(d222 - w2)
                        )

                        Gyx = np.dot(wt, Gyx)

                        Gzx = (
                            np.log(d111 - v1)
                            - np.log(d211 - v1)
                            + np.log(d221 - v2)
                            - np.log(d121 - v2)
                            + np.log(d122 - v2)
                            - np.log(d112 - v1)
                            + np.log(d212 - v1)
                            - np.log(d222 - v2)
                        )

                        Gzx = np.dot(wt, Gzx)

                        G[COUNT, :] = c * np.c_[Gxx, Gyx, Gzx]
                        COUNT = COUNT + 1

                elif dComp.lower() == "y":
                    for rr in range(0, nLoc):
                        u1 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax) + np.kron(
                            s1, np.ones((1, nC))
                        )
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0]) / tol2
                        u2 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx) + np.kron(
                            s1, np.ones((1, nC))
                        )
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0]) / tol2

                        v1 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay)
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1]) / tol2
                        v2 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by)
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1]) / tol2

                        w1 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az) + np.kron(
                            s2, np.ones((1, nC))
                        )
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2]) / tol2
                        w2 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz) + np.kron(
                            s2, np.ones((1, nC))
                        )
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2]) / tol2

                        d111 = np.sqrt(u1**2 + v1**2 + w1**2)
                        d211 = np.sqrt(u2**2 + v1**2 + w1**2)
                        d221 = np.sqrt(u2**2 + v2**2 + w1**2)
                        d121 = np.sqrt(u1**2 + v2**2 + w1**2)
                        d122 = np.sqrt(u1**2 + v2**2 + w2**2)
                        d112 = np.sqrt(u1**2 + v1**2 + w2**2)
                        d212 = np.sqrt(u2**2 + v1**2 + w2**2)
                        d222 = np.sqrt(u2**2 + v2**2 + w2**2)

                        Gxy = (
                            np.log(d111 - w1)
                            - np.log(d211 - w1)
                            + np.log(d221 - w1)
                            - np.log(d121 - w1)
                            + np.log(d122 - w2)
                            - np.log(d112 - w2)
                            + np.log(d212 - w2)
                            - np.log(d222 - w2)
                        )

                        Gxy = np.dot(wt, Gxy)

                        Gyy = (
                            np.arctan((u1 * w1) / (v1 * d111 + tol))
                            - np.arctan((u2 * w1) / (v1 * d211 + tol))
                            + np.arctan((u2 * w1) / (v2 * d221 + tol))
                            - np.arctan((u1 * w1) / (v2 * d121 + tol))
                            + np.arctan((u1 * w2) / (v2 * d122 + tol))
                            - np.arctan((u1 * w2) / (v1 * d112 + tol))
                            + np.arctan((u2 * w2) / (v1 * d212 + tol))
                            - np.arctan((u2 * w2) / (v2 * d222 + tol))
                        )

                        Gyy = np.dot(wt, Gyy)

                        Gzy = (
                            np.log(d111 - u1)
                            - np.log(d211 - u2)
                            + np.log(d221 - u2)
                            - np.log(d121 - u1)
                            + np.log(d122 - u1)
                            - np.log(d112 - u1)
                            + np.log(d212 - u2)
                            - np.log(d222 - u2)
                        )

                        Gzy = np.dot(wt, Gzy)

                        G[COUNT, :] = c * np.c_[Gxy, Gyy, Gzy]
                        COUNT = COUNT + 1

                elif dComp.lower() == "z":
                    for rr in range(0, nLoc):
                        u1 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax) + np.kron(
                            s1, np.ones((1, nC))
                        )
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0]) / tol2
                        u2 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx) + np.kron(
                            s1, np.ones((1, nC))
                        )

                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0]) / tol2
                        v1 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay) + np.kron(
                            s2, np.ones((1, nC))
                        )
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1]) / tol2
                        v2 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by) + np.kron(
                            s2, np.ones((1, nC))
                        )
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1]) / tol2

                        w1 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az)
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2]) / tol2
                        w2 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz)
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2]) / tol2

                        d111 = np.sqrt(u1**2 + v1**2 + w1**2)
                        d211 = np.sqrt(u2**2 + v1**2 + w1**2)
                        d221 = np.sqrt(u2**2 + v2**2 + w1**2)
                        d121 = np.sqrt(u1**2 + v2**2 + w1**2)
                        d122 = np.sqrt(u1**2 + v2**2 + w2**2)
                        d112 = np.sqrt(u1**2 + v1**2 + w2**2)
                        d212 = np.sqrt(u2**2 + v1**2 + w2**2)
                        d222 = np.sqrt(u2**2 + v2**2 + w2**2)

                        Gxz = (
                            np.log(d111 - v1)
                            - np.log(d211 - v1)
                            + np.log(d221 - v2)
                            - np.log(d121 - v2)
                            + np.log(d122 - v2)
                            - np.log(d112 - v1)
                            + np.log(d212 - v1)
                            - np.log(d222 - v2)
                        )

                        Gxz = np.dot(wt, Gxz)

                        Gyz = (
                            np.log(d111 - u1)
                            - np.log(d211 - u2)
                            + np.log(d221 - u2)
                            - np.log(d121 - u1)
                            + np.log(d122 - u1)
                            - np.log(d112 - u1)
                            + np.log(d212 - u2)
                            - np.log(d222 - u2)
                        )

                        Gyz = np.dot(wt, Gyz)

                        Gzz = (
                            -np.arctan((v1 * w1) / (u1 * d111 + tol))
                            + np.arctan((v1 * w1) / (u2 * d211 + tol))
                            - np.arctan((v2 * w1) / (u2 * d221 + tol))
                            + np.arctan((v2 * w1) / (u1 * d121 + tol))
                            - np.arctan((v2 * w2) / (u1 * d122 + tol))
                            + np.arctan((v1 * w2) / (u1 * d112 + tol))
                            - np.arctan((v1 * w2) / (u2 * d212 + tol))
                            + np.arctan((v2 * w2) / (u2 * d222 + tol))
                        )

                        Gzz = (
                            Gzz
                            - np.arctan((u1 * w1) / (v1 * d111 + tol))
                            + np.arctan((u2 * w1) / (v1 * d211 + tol))
                            - np.arctan((u2 * w1) / (v2 * d221 + tol))
                            + np.arctan((u1 * w1) / (v2 * d121 + tol))
                            - np.arctan((u1 * w2) / (v2 * d122 + tol))
                            + np.arctan((u1 * w2) / (v1 * d112 + tol))
                            - np.arctan((u2 * w2) / (v1 * d212 + tol))
                            + np.arctan((u2 * w2) / (v2 * d222 + tol))
                        )

                        Gzz = np.dot(wt, Gzz)

                        G[COUNT, :] = c * np.c_[Gxz, Gyz, Gzz]
                        COUNT = COUNT + 1

        return G

    def _getAMatricies(self):
        """Returns the full geometric operator"""

        indActive = self.indActive

        # GET CELL INFORMATION FOR FORWARD MODELING
        meshObj = self.mesh
        xyzc = meshObj.gridCC[indActive, :]
        xyzh = meshObj.h_gridded[indActive, :]

        # GET LIST OF A MATRICIES
        A = []
        for pp in range(0, self.survey.nSrc):
            # Create initial A matrix
            G = self._getGeometryMatrix(xyzc, xyzh, pp)
            H0 = self._getH0matrix(xyzc, pp)
            A.append(G * H0)

            # Refine A matrix
            refinement_factor = self.refinement_factor
            refinement_distance = self.refinement_distance

            if refinement_factor > 0:
                srcObj = self.survey.source_list[pp]
                refFlag = srcObj._getRefineFlags(
                    xyzc, refinement_factor, refinement_distance
                )

                for qq in range(1, refinement_factor + 1):
                    if len(refFlag[refFlag == qq]) != 0:
                        A[pp][:, refFlag == qq] = self._getSubsetAcolumns(
                            xyzc, xyzh, pp, qq, refFlag
                        )

        return A

    def _getSubsetAcolumns(self, xyzc, xyzh, pp, qq, refFlag):
        """
                This method returns the refined sensitivities for columns that will be
                replaced in the A matrix for source pp and refinement factor qq.
        ..
        ..        INPUTS:
        ..
        ..        xyzc -- Cell centers of topo mesh cells N X 3 array
        ..
        ..        xyzh -- Cell widths of topo mesh cells N X 3 array
        ..
        ..        pp -- Source ID
        ..
        ..        qq -- Mesh refinement factor
        ..
        ..        refFlag -- refinement factors for all topo mesh cells
        ..
        ..        OUTPUTS:
        ..
        ..        Acols -- Columns containing replacement sensitivities

        """

        # GET SUBMESH GRID
        n = 2**qq
        [nx, ny, nz] = np.meshgrid(
            np.linspace(1, n, n) - 0.5,
            np.linspace(1, n, n) - 0.5,
            np.linspace(1, n, n) - 0.5,
        )
        nxyz_sub = np.c_[mkvc(nx), mkvc(ny), mkvc(nz)]

        xyzh_sub = xyzh[refFlag == qq, :]  # Get widths of cells to be refined
        xyzc_sub = (
            xyzc[refFlag == qq, :] - xyzh[refFlag == qq, :] / 2
        )  # Get bottom southwest corners of cells to be refined
        m = np.shape(xyzc_sub)[0]
        xyzc_sub = np.kron(
            xyzc_sub, np.ones((n**3, 1))
        )  # Kron for n**3 refined cells
        xyzh_sub = np.kron(
            xyzh_sub / n, np.ones((n**3, 1))
        )  # Kron for n**3 refined cells with widths h/n
        nxyz_sub = np.kron(np.ones((m, 1)), nxyz_sub)  # Kron for n**3 refined cells
        xyzc_sub = xyzc_sub + xyzh_sub * nxyz_sub

        # GET SUBMESH A MATRIX AND COLLAPSE TO COLUMNS
        G = self._getGeometryMatrix(xyzc_sub, xyzh_sub, pp)
        H0 = self._getH0matrix(xyzc_sub, pp)
        Acols = (G * H0) * sp.kron(sp.diags(np.ones(m)), np.ones((n**3, 1)))

        return Acols

    def dpred(self, m=None, f=None):
        """"""
        if f is None:
            f = self.fields(m)

        return f[self.survey.t_active]


#############################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND INVERSION)
#############################################################################


class Simulation3DLinear(BaseVRMSimulation):
    """"""

    _A = None
    _T = None
    _TisSet = False
    _xiMap = None

    xi, xiMap, xiDeriv = props.Invertible(
        "Amalgamated Viscous Remanent Magnetization Parameter xi = dchi/ln(tau2/tau1)"
    )

    def __init__(self, mesh, xi=None, xiMap=None, **kwargs):
        super().__init__(mesh, **kwargs)
        self.xi = xi
        self.xiMap = xiMap

        nAct = list(self.indActive).count(True)
        if self.xiMap is None:
            self.xiMap = maps.IdentityMap(nP=nAct)

    @property
    def A(self):
        """
        The geometric sensitivity matrix for the linear VRM problem. Accessing
        this property requires that the problem be paired with a survey object.

        """

        if self._AisSet is False:
            if self.survey is None:
                AssertionError("A survey must be set to generate A matrix")

            # Remove any previously stored A matrix
            if self._A is not None:
                self._A = None

            print("CREATING A MATRIX")

            # COLLAPSE ALL A MATRICIES INTO SINGLE OPERATOR
            self._A = np.vstack(self._getAMatricies())
            self._AisSet = True

            return self._A

        elif self._AisSet is True:
            return self._A

    @property
    def T(self):
        """
        The characteristic decay matrix for the VRM problem. Accessing this
        property requires that the problem be paired with a survey object.

        """

        if self._TisSet is False:
            if self.survey is None:
                AssertionError("A survey must be set to generate A matrix")

            # Remove any previously stored T matrix
            if self._T is not None:
                self._T = None

            print("CREATING T MATRIX")

            source_list = self.survey.source_list
            nSrc = len(source_list)
            T = []

            for pp in range(0, nSrc):
                receiver_list = source_list[pp].receiver_list
                nRx = len(receiver_list)
                waveObj = source_list[pp].waveform

                for qq in range(0, nRx):
                    times = receiver_list[qq].times
                    nLoc = np.shape(receiver_list[qq].locations)[0]

                    I = sp.diags(np.ones(nLoc))
                    eta = waveObj.getCharDecay(receiver_list[qq].field_type, times)
                    eta = np.atleast_2d(eta).T

                    T.append(sp.kron(I, eta))

            self._T = sp.block_diag(T)
            self._TisSet = True

            return self._T

        elif self._TisSet is True:
            return self._T

    def fields(self, m):
        """Computes the fields d = T*A*m"""

        if self.survey is None:
            AssertionError("A survey must be set to generate A matrix")

        self.model = m  # Initiates/updates model and initiates mapping

        # Project to active mesh cells
        m = self.xiMap * m

        # Must return as a numpy array
        return mkvc(sp.coo_matrix.dot(self.T, np.dot(self.A, m)))

    def Jvec(self, m, v, f=None):
        """Compute Pd*T*A*dxidm*v"""

        if self.survey is None:
            AssertionError("A survey must be set to generate A matrix")

        # Jacobian of xi wrt model
        dxidm = self.xiMap.deriv(m)

        # dxidm*v
        v = dxidm * v

        # Dot product with A
        v = self.A @ v

        # Get active time rows of T
        T = self.T.tocsr()[self.survey.t_active, :]

        # Must return an array
        return mkvc(sp.csr_matrix.dot(T, v))

    def Jtvec(self, m, v, f=None):
        """Compute (Pd*T*A*dxidm)^T * v"""

        if self.survey is None:
            AssertionError("A survey must be set to generate A matrix")

        # Define v as a column vector
        v = np.atleast_2d(v).T

        # Get T'*Pd'*v
        T = self.T.tocsr()[self.survey.t_active, :]
        v = sp.csc_matrix.dot(T.transpose(), v)

        # Multiply by A'
        v = (np.dot(v.T, self.A)).T

        # Jacobian of xi wrt model
        dxidm = self.xiMap.deriv(m)

        # Must return an array
        return mkvc(dxidm.T * v)


class Simulation3DLogUniform(BaseVRMSimulation):
    """"""

    _A = None
    _T = None
    _TisSet = False
    # _xiMap = None

    chi0 = props.PhysicalProperty("DC susceptibility")
    dchi = props.PhysicalProperty("Frequency dependence")
    tau1 = props.PhysicalProperty("Low bound time-relaxation constant")
    tau2 = props.PhysicalProperty("Upper bound time-relaxation constant")

    def __init__(
        self, mesh, survey=None, chi0=None, dchi=None, tau1=None, tau2=None, **kwargs
    ):
        super(Simulation3DLogUniform, self).__init__(mesh=mesh, survey=survey, **kwargs)
        self.chi0 = chi0
        self.dchi = dchi
        self.tau1 = tau1
        self.tau2 = tau2

    @property
    def A(self):
        """
        The geometric sensitivity matrix for the linear VRM problem. Accessing
        this property requires that the problem be paired with a survey object.

        """

        if self._AisSet is False:
            if self.survey is None:
                AssertionError("A survey must be set to generate A matrix")

            # Remove any previously stored A matrix
            if self._A is not None:
                self._A = None

            print("CREATING A MATRIX")

            # COLLAPSE ALL A MATRICIES INTO SINGLE OPERATOR
            self._A = self._getAMatricies()
            self._AisSet = True

            return self._A

        elif self._AisSet is True:
            return self._A

    def fields(self, m=None):
        """Computes the fields at every time d(t) = G*M(t)"""

        if self.survey is None:
            AssertionError("A survey must be set to generate A matrix")

        # Fields from each source
        source_list = self.survey.source_list
        nSrc = len(source_list)
        f = []

        for pp in range(0, nSrc):
            receiver_list = source_list[pp].receiver_list
            nRx = len(receiver_list)
            waveObj = source_list[pp].waveform

            for qq in range(0, nRx):
                times = receiver_list[qq].times
                eta = waveObj.getLogUniformDecay(
                    receiver_list[qq].field_type,
                    times,
                    self.chi0,
                    self.dchi,
                    self.tau1,
                    self.tau2,
                )

                f.append(mkvc(self.A[qq] @ eta))

        return np.array(np.hstack(f))
