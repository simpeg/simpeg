import numpy as np
import scipy.sparse as sp
import properties
from ...utils.code_utils import deprecate_class, deprecate_property

from ...simulation import BaseSimulation
from ... import props
from ... import maps
from ...utils import mkvc

from .survey import SurveyVRM
from .receivers import Point, SquareLoop


############################################
# BASE VRM PROBLEM CLASS
############################################


class BaseVRMSimulation(BaseSimulation):
    """

    """

    _AisSet = False
    refinement_factor = properties.Integer('Sensitivity refinement factor', min=0)
    refinement_distance = properties.Array('Sensitivity refinement radii from sources', dtype=float)
    indActive = properties.Array('Topography active cells', dtype=bool)

    ref_factor = deprecate_property(refinement_factor, 'ref_factor', removal_version='0.15.0')
    ref_radius = deprecate_property(refinement_distance, 'ref_radius', removal_version='0.15.0')

    def __init__(self, mesh=None, **kwargs):

        refinement_factor = kwargs.pop('refinement_factor', None)
        refinement_distance = kwargs.pop('refinement_distance', None)
        indActive = kwargs.pop('indActive', None)

        if len(mesh.h) != 3:
            raise ValueError('Mesh must be 3D tensor or 3D tree. Current mesh is {}'.format(len(mesh.h)))

        super(BaseVRMSimulation, self).__init__(mesh, **kwargs)

        if refinement_factor is None and refinement_distance is None:
            self.refinement_factor = 3
            self.refinement_distance = list(1.25*np.mean(np.r_[np.min(mesh.h[0]), np.min(mesh.h[1]), np.min(mesh.h[2])])*np.arange(1, 4))
        elif refinement_factor is None and refinement_distance is not None:
            self.refinement_factor = len(refinement_distance)
            self.refinement_distance = refinement_distance
        elif refinement_factor is not None and refinement_distance is None:
            self.refinement_factor = refinement_factor
            self.refinement_distance = list(1.25*np.mean(np.r_[np.min(mesh.h[0]), np.min(mesh.h[1]), np.min(mesh.h[2])])*np.arange(1, refinement_factor+1))
        else:
            self.refinement_factor = refinement_factor
            self.refinement_distance = refinement_distance

        if indActive is None:
            self.indActive = np.ones(mesh.nC, dtype=bool)
        else:
            self.indActive = indActive

    @properties.observer('refinement_factor')
    def _refinement_factor_observer(self, change):
        if change['value'] > 4:
            print("Refinement factor larger than 4 may result in computations which exceed memory limits")
        if self.refinement_distance is not None and change['value'] != len(self.refinement_distance):
            print("Number of refinement radii currently DOES NOT match refinement_factor")

    @properties.observer('refinement_distance')
    def _refinement_distance_validator(self, change):
        if self.refinement_factor is not None and len(change['value']) != self.refinement_factor:
            print("Number of refinement radii current DOES NOT match refinement_factor")

    @properties.validator('indActive')
    def _indActive_validator(self, change):

        if len(change['value']) != self.mesh.nC:
            raise ValueError("Length of active topo cells array must equal number of mesh cells (nC = {})".format(self.mesh.nC))

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

        nC = np.shape(xyzc)[0]   # Number of cells
        nRx = srcObj.nRx          # Number of receiver in all rxList

        ax = np.reshape(xyzc[:, 0] - xyzh[:, 0]/2, (1, nC))
        bx = np.reshape(xyzc[:, 0] + xyzh[:, 0]/2, (1, nC))
        ay = np.reshape(xyzc[:, 1] - xyzh[:, 1]/2, (1, nC))
        by = np.reshape(xyzc[:, 1] + xyzh[:, 1]/2, (1, nC))
        az = np.reshape(xyzc[:, 2] - xyzh[:, 2]/2, (1, nC))
        bz = np.reshape(xyzc[:, 2] + xyzh[:, 2]/2, (1, nC))

        G = np.zeros((nRx, 3*nC))
        c = -(1/(4*np.pi))
        tol = 1e-10   # Tolerance constant for numerical stability
        tol2 = 1000.  # Tolerance constant for numerical stability

        COUNT = 0

        for qq in range(0, len(srcObj.receiver_list)):

            rxObj = srcObj.receiver_list[qq]
            dComp = rxObj.orientation
            locs = rxObj.locations
            nLoc = np.shape(locs)[0]

            if isinstance(rxObj, Point) and not isinstance(rxObj, SquareLoop):

                if dComp.lower() == 'x':
                    for rr in range(0, nLoc):
                        u1 = locs[rr, 0] - ax
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = locs[rr, 0] - bx
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2
                        v1 = locs[rr, 1] - ay
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = locs[rr, 1] - by
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2
                        w1 = locs[rr, 2] - az
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = locs[rr, 2] - bz
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        d111 = np.sqrt(u1**2+v1**2+w1**2)
                        d211 = np.sqrt(u2**2+v1**2+w1**2)
                        d221 = np.sqrt(u2**2+v2**2+w1**2)
                        d121 = np.sqrt(u1**2+v2**2+w1**2)
                        d122 = np.sqrt(u1**2+v2**2+w2**2)
                        d112 = np.sqrt(u1**2+v1**2+w2**2)
                        d212 = np.sqrt(u2**2+v1**2+w2**2)
                        d222 = np.sqrt(u2**2+v2**2+w2**2)

                        Gxx = (
                            np.arctan((v1*w1)/(u1*d111+tol)) -
                            np.arctan((v1*w1)/(u2*d211+tol)) +
                            np.arctan((v2*w1)/(u2*d221+tol)) -
                            np.arctan((v2*w1)/(u1*d121+tol)) +
                            np.arctan((v2*w2)/(u1*d122+tol)) -
                            np.arctan((v1*w2)/(u1*d112+tol)) +
                            np.arctan((v1*w2)/(u2*d212+tol)) -
                            np.arctan((v2*w2)/(u2*d222+tol))
                        )

                        Gyx = (
                            np.log(d111-w1) -
                            np.log(d211-w1) +
                            np.log(d221-w1) -
                            np.log(d121-w1) +
                            np.log(d122-w2) -
                            np.log(d112-w2) +
                            np.log(d212-w2) -
                            np.log(d222-w2)
                        )

                        Gzx = (
                            np.log(d111-v1) -
                            np.log(d211-v1) +
                            np.log(d221-v2) -
                            np.log(d121-v2) +
                            np.log(d122-v2) -
                            np.log(d112-v1) +
                            np.log(d212-v1) -
                            np.log(d222-v2)
                        )

                        G[COUNT, :] = c*np.c_[Gxx, Gyx, Gzx]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'y':
                    for rr in range(0, nLoc):
                        u1 = locs[rr, 0] - ax
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = locs[rr, 0] - bx
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2
                        v1 = locs[rr, 1] - ay
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = locs[rr, 1] - by
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2
                        w1 = locs[rr, 2] - az
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = locs[rr, 2] - bz
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        d111 = np.sqrt(u1**2+v1**2+w1**2)
                        d211 = np.sqrt(u2**2+v1**2+w1**2)
                        d221 = np.sqrt(u2**2+v2**2+w1**2)
                        d121 = np.sqrt(u1**2+v2**2+w1**2)
                        d122 = np.sqrt(u1**2+v2**2+w2**2)
                        d112 = np.sqrt(u1**2+v1**2+w2**2)
                        d212 = np.sqrt(u2**2+v1**2+w2**2)
                        d222 = np.sqrt(u2**2+v2**2+w2**2)

                        Gxy = (
                            np.log(d111-w1) -
                            np.log(d211-w1) +
                            np.log(d221-w1) -
                            np.log(d121-w1) +
                            np.log(d122-w2) -
                            np.log(d112-w2) +
                            np.log(d212-w2) -
                            np.log(d222-w2)
                        )

                        Gyy = (
                            np.arctan((u1*w1)/(v1*d111+tol)) -
                            np.arctan((u2*w1)/(v1*d211+tol)) +
                            np.arctan((u2*w1)/(v2*d221+tol)) -
                            np.arctan((u1*w1)/(v2*d121+tol)) +
                            np.arctan((u1*w2)/(v2*d122+tol)) -
                            np.arctan((u1*w2)/(v1*d112+tol)) +
                            np.arctan((u2*w2)/(v1*d212+tol)) -
                            np.arctan((u2*w2)/(v2*d222+tol))
                        )

                        Gzy = (
                            np.log(d111-u1) -
                            np.log(d211-u2) +
                            np.log(d221-u2) -
                            np.log(d121-u1) +
                            np.log(d122-u1) -
                            np.log(d112-u1) +
                            np.log(d212-u2) -
                            np.log(d222-u2)
                        )

                        G[COUNT, :] = c*np.c_[Gxy, Gyy, Gzy]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'z':
                    for rr in range(0, nLoc):
                        u1 = locs[rr, 0] - ax
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = locs[rr, 0] - bx
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2
                        v1 = locs[rr, 1] - ay
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = locs[rr, 1] - by
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2
                        w1 = locs[rr, 2] - az
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = locs[rr, 2] - bz
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        d111 = np.sqrt(u1**2+v1**2+w1**2)
                        d211 = np.sqrt(u2**2+v1**2+w1**2)
                        d221 = np.sqrt(u2**2+v2**2+w1**2)
                        d121 = np.sqrt(u1**2+v2**2+w1**2)
                        d122 = np.sqrt(u1**2+v2**2+w2**2)
                        d112 = np.sqrt(u1**2+v1**2+w2**2)
                        d212 = np.sqrt(u2**2+v1**2+w2**2)
                        d222 = np.sqrt(u2**2+v2**2+w2**2)

                        Gxz = (
                            np.log(d111-v1) -
                            np.log(d211-v1) +
                            np.log(d221-v2) -
                            np.log(d121-v2) +
                            np.log(d122-v2) -
                            np.log(d112-v1) +
                            np.log(d212-v1) -
                            np.log(d222-v2)
                        )

                        Gyz = (
                            np.log(d111-u1) -
                            np.log(d211-u2) +
                            np.log(d221-u2) -
                            np.log(d121-u1) +
                            np.log(d122-u1) -
                            np.log(d112-u1) +
                            np.log(d212-u2) -
                            np.log(d222-u2)
                        )

                        Gzz = (
                            - np.arctan((v1*w1)/(u1*d111+tol)) +
                            np.arctan((v1*w1)/(u2*d211+tol)) -
                            np.arctan((v2*w1)/(u2*d221+tol)) +
                            np.arctan((v2*w1)/(u1*d121+tol)) -
                            np.arctan((v2*w2)/(u1*d122+tol)) +
                            np.arctan((v1*w2)/(u1*d112+tol)) -
                            np.arctan((v1*w2)/(u2*d212+tol)) +
                            np.arctan((v2*w2)/(u2*d222+tol))
                        )

                        Gzz = (
                            Gzz -
                            np.arctan((u1*w1)/(v1*d111+tol)) +
                            np.arctan((u2*w1)/(v1*d211+tol)) -
                            np.arctan((u2*w1)/(v2*d221+tol)) +
                            np.arctan((u1*w1)/(v2*d121+tol)) -
                            np.arctan((u1*w2)/(v2*d122+tol)) +
                            np.arctan((u1*w2)/(v1*d112+tol)) -
                            np.arctan((u2*w2)/(v1*d212+tol)) +
                            np.arctan((u2*w2)/(v2*d222+tol))
                        )

                        G[COUNT, :] = c*np.c_[Gxz, Gyz, Gzz]
                        COUNT = COUNT + 1

            elif isinstance(rxObj, SquareLoop):

                # Gaussian quadrature weights
                wt = [
                    np.r_[2.],
                    np.r_[1., 1.],
                    np.r_[0.555556, 0.888889, 0.555556],
                    np.r_[0.347855, 0.652145, 0.652145, 0.347855],
                    np.r_[0.236927, 0.478629, 0.568889, 0.478629, 0.236927],
                    np.r_[0.171324, 0.467914, 0.360762, 0.360762, 0.467914, 0.171324],
                    np.r_[0.129485, 0.279705, 0.381830, 0.417959, 0.381830, 0.279705, 0.129485]
                ]
                wt = wt[rxObj.quadOrder-1]
                nw = len(wt)
                wt = rxObj.nTurns*(rxObj.width/2)**2*np.reshape(np.outer(wt, wt), (1, nw**2))

                # Gaussian quadrature locations on [-1,1]
                ds = [
                    np.r_[0.],
                    np.r_[-0.57735, 0.57735],
                    np.r_[-0.774597, 0., 0.774597],
                    np.r_[-0.861136, -0.339981, 0.339981, 0.861136],
                    np.r_[-0.906180, -0.538469, 0, 0.538469, 0.906180],
                    np.r_[-0.932470, -0.238619, -0.661209, 0.661209, 0.238619, 0.932470],
                    np.r_[-0.949108, -0.741531, -0.405845, 0., 0.405845, 0.741531, 0.949108]
                ]

                s1 = (
                    0.5*rxObj.width *
                    np.reshape(np.kron(ds[rxObj.quadOrder-1], np.ones(nw)), (nw**2, 1))
                )
                s2 = (
                    0.5*rxObj.width *
                    np.reshape(np.kron(np.ones(nw), ds[rxObj.quadOrder-1]), (nw**2, 1))
                )

                if dComp.lower() == 'x':
                    for rr in range(0, nLoc):

                        u1 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax)
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx)
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2

                        v1 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay) +
                            np.kron(s1, np.ones((1, nC)))
                        )
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by) +
                            np.kron(s1, np.ones((1, nC)))
                        )
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2

                        w1 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az) +
                            np.kron(s2, np.ones((1, nC)))
                        )
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz) +
                            np.kron(s2, np.ones((1, nC)))
                        )
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        d111 = np.sqrt(u1**2+v1**2+w1**2)
                        d211 = np.sqrt(u2**2+v1**2+w1**2)
                        d221 = np.sqrt(u2**2+v2**2+w1**2)
                        d121 = np.sqrt(u1**2+v2**2+w1**2)
                        d122 = np.sqrt(u1**2+v2**2+w2**2)
                        d112 = np.sqrt(u1**2+v1**2+w2**2)
                        d212 = np.sqrt(u2**2+v1**2+w2**2)
                        d222 = np.sqrt(u2**2+v2**2+w2**2)

                        Gxx = (
                            np.arctan((v1*w1)/(u1*d111+tol)) -
                            np.arctan((v1*w1)/(u2*d211+tol)) +
                            np.arctan((v2*w1)/(u2*d221+tol)) -
                            np.arctan((v2*w1)/(u1*d121+tol)) +
                            np.arctan((v2*w2)/(u1*d122+tol)) -
                            np.arctan((v1*w2)/(u1*d112+tol)) +
                            np.arctan((v1*w2)/(u2*d212+tol)) -
                            np.arctan((v2*w2)/(u2*d222+tol))
                        )

                        Gxx = np.dot(wt, Gxx)

                        Gyx = (
                            np.log(d111-w1) -
                            np.log(d211-w1) +
                            np.log(d221-w1) -
                            np.log(d121-w1) +
                            np.log(d122-w2) -
                            np.log(d112-w2) +
                            np.log(d212-w2) -
                            np.log(d222-w2)
                        )

                        Gyx = np.dot(wt, Gyx)

                        Gzx = (
                            np.log(d111-v1) -
                            np.log(d211-v1) +
                            np.log(d221-v2) -
                            np.log(d121-v2) +
                            np.log(d122-v2) -
                            np.log(d112-v1) +
                            np.log(d212-v1) -
                            np.log(d222-v2)
                        )

                        Gzx = np.dot(wt, Gzx)

                        G[COUNT, :] = c*np.c_[Gxx, Gyx, Gzx]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'y':
                    for rr in range(0, nLoc):

                        u1 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax) +
                            np.kron(s1, np.ones((1, nC)))
                        )
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx) +
                            np.kron(s1, np.ones((1, nC)))
                        )
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2

                        v1 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay)
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by)
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2

                        w1 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az) +
                            np.kron(s2, np.ones((1, nC)))
                        )
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz) +
                            np.kron(s2, np.ones((1, nC)))
                        )
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        d111 = np.sqrt(u1**2+v1**2+w1**2)
                        d211 = np.sqrt(u2**2+v1**2+w1**2)
                        d221 = np.sqrt(u2**2+v2**2+w1**2)
                        d121 = np.sqrt(u1**2+v2**2+w1**2)
                        d122 = np.sqrt(u1**2+v2**2+w2**2)
                        d112 = np.sqrt(u1**2+v1**2+w2**2)
                        d212 = np.sqrt(u2**2+v1**2+w2**2)
                        d222 = np.sqrt(u2**2+v2**2+w2**2)

                        Gxy = (
                            np.log(d111-w1) -
                            np.log(d211-w1) +
                            np.log(d221-w1) -
                            np.log(d121-w1) +
                            np.log(d122-w2) -
                            np.log(d112-w2) +
                            np.log(d212-w2) -
                            np.log(d222-w2)
                        )

                        Gxy = np.dot(wt, Gxy)

                        Gyy = (
                            np.arctan((u1*w1)/(v1*d111+tol)) -
                            np.arctan((u2*w1)/(v1*d211+tol)) +
                            np.arctan((u2*w1)/(v2*d221+tol)) -
                            np.arctan((u1*w1)/(v2*d121+tol)) +
                            np.arctan((u1*w2)/(v2*d122+tol)) -
                            np.arctan((u1*w2)/(v1*d112+tol)) +
                            np.arctan((u2*w2)/(v1*d212+tol)) -
                            np.arctan((u2*w2)/(v2*d222+tol))
                        )

                        Gyy = np.dot(wt, Gyy)

                        Gzy = (
                            np.log(d111-u1) -
                            np.log(d211-u2) +
                            np.log(d221-u2) -
                            np.log(d121-u1) +
                            np.log(d122-u1) -
                            np.log(d112-u1) +
                            np.log(d212-u2) -
                            np.log(d222-u2)
                        )

                        Gzy = np.dot(wt, Gzy)

                        G[COUNT, :] = c*np.c_[Gxy, Gyy, Gzy]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'z':
                    for rr in range(0, nLoc):

                        u1 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax) +
                            np.kron(s1, np.ones((1, nC)))
                        )
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx) +
                            np.kron(s1, np.ones((1, nC)))
                        )

                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2
                        v1 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay) +
                            np.kron(s2, np.ones((1, nC)))
                        )
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = (
                            np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by) +
                            np.kron(s2, np.ones((1, nC)))
                        )
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2

                        w1 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az)
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz)
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        d111 = np.sqrt(u1**2+v1**2+w1**2)
                        d211 = np.sqrt(u2**2+v1**2+w1**2)
                        d221 = np.sqrt(u2**2+v2**2+w1**2)
                        d121 = np.sqrt(u1**2+v2**2+w1**2)
                        d122 = np.sqrt(u1**2+v2**2+w2**2)
                        d112 = np.sqrt(u1**2+v1**2+w2**2)
                        d212 = np.sqrt(u2**2+v1**2+w2**2)
                        d222 = np.sqrt(u2**2+v2**2+w2**2)

                        Gxz = (
                            np.log(d111-v1) -
                            np.log(d211-v1) +
                            np.log(d221-v2) -
                            np.log(d121-v2) +
                            np.log(d122-v2) -
                            np.log(d112-v1) +
                            np.log(d212-v1) -
                            np.log(d222-v2)
                        )

                        Gxz = np.dot(wt, Gxz)

                        Gyz = (
                            np.log(d111-u1) -
                            np.log(d211-u2) +
                            np.log(d221-u2) -
                            np.log(d121-u1) +
                            np.log(d122-u1) -
                            np.log(d112-u1) +
                            np.log(d212-u2) -
                            np.log(d222-u2)
                        )

                        Gyz = np.dot(wt, Gyz)

                        Gzz = (
                            - np.arctan((v1*w1)/(u1*d111+tol)) +
                            np.arctan((v1*w1)/(u2*d211+tol)) -
                            np.arctan((v2*w1)/(u2*d221+tol)) +
                            np.arctan((v2*w1)/(u1*d121+tol)) -
                            np.arctan((v2*w2)/(u1*d122+tol)) +
                            np.arctan((v1*w2)/(u1*d112+tol)) -
                            np.arctan((v1*w2)/(u2*d212+tol)) +
                            np.arctan((v2*w2)/(u2*d222+tol))
                        )

                        Gzz = (
                            Gzz -
                            np.arctan((u1*w1)/(v1*d111+tol)) +
                            np.arctan((u2*w1)/(v1*d211+tol)) -
                            np.arctan((u2*w1)/(v2*d221+tol)) +
                            np.arctan((u1*w1)/(v2*d121+tol)) -
                            np.arctan((u1*w2)/(v2*d122+tol)) +
                            np.arctan((u1*w2)/(v1*d112+tol)) -
                            np.arctan((u2*w2)/(v1*d212+tol)) +
                            np.arctan((u2*w2)/(v2*d222+tol))
                        )

                        Gzz = np.dot(wt, Gzz)

                        G[COUNT, :] = c*np.c_[Gxz, Gyz, Gzz]
                        COUNT = COUNT + 1

        return np.matrix(G)

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
            A.append(G*H0)

            # Refine A matrix
            refinement_factor = self.refinement_factor
            refinement_distance = self.refinement_distance

            if refinement_factor > 0:

                srcObj = self.survey.source_list[pp]
                refFlag = srcObj._getRefineFlags(xyzc, refinement_factor, refinement_distance)

                for qq in range(1, refinement_factor+1):
                    if len(refFlag[refFlag == qq]) != 0:
                        A[pp][:, refFlag == qq] =self._getSubsetAcolumns(xyzc, xyzh, pp, qq, refFlag)

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
            np.linspace(1, n, n)-0.5, np.linspace(1, n, n)-0.5, np.linspace(1, n, n)-0.5)
        nxyz_sub = np.c_[mkvc(nx), mkvc(ny), mkvc(nz)]

        xyzh_sub = xyzh[refFlag == qq, :]     # Get widths of cells to be refined
        xyzc_sub = xyzc[refFlag == qq, :] - xyzh[refFlag == qq, :]/2   # Get bottom southwest corners of cells to be refined
        m = np.shape(xyzc_sub)[0]
        xyzc_sub = np.kron(xyzc_sub, np.ones((n**3, 1)))     # Kron for n**3 refined cells
        xyzh_sub = np.kron(xyzh_sub/n, np.ones((n**3, 1)))   # Kron for n**3 refined cells with widths h/n
        nxyz_sub = np.kron(np.ones((m, 1)), nxyz_sub)        # Kron for n**3 refined cells
        xyzc_sub = xyzc_sub + xyzh_sub*nxyz_sub

        # GET SUBMESH A MATRIX AND COLLAPSE TO COLUMNS
        G = self._getGeometryMatrix(xyzc_sub, xyzh_sub, pp)
        H0 = self._getH0matrix(xyzc_sub, pp)
        Acols = (G*H0)*sp.kron(sp.diags(np.ones(m)), np.ones((n**3, 1)))

        return Acols

    def dpred(self, m=None, f=None):

        """

        """
        if f is None:
            f = self.fields(m)

        return f[self.survey.t_active]


#############################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND INVERSION)
#############################################################################


class Simulation3DLinear(BaseVRMSimulation):

    """

    """

    _A = None
    _T = None
    _TisSet = False
    _xiMap = None

    survey = properties.Instance(
        "VRM Survey", SurveyVRM
    )

    xi, xiMap, xiDeriv = props.Invertible(
        "Amalgamated Viscous Remanent Magnetization Parameter xi = dchi/ln(tau2/tau1)")

    def __init__(self, mesh, **kwargs):

        super(Simulation3DLinear, self).__init__(mesh, **kwargs)

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

            print('CREATING A MATRIX')

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

            print('CREATING T MATRIX')

            srcList = self.survey.source_list
            nSrc = len(srcList)
            T = []

            for pp in range(0, nSrc):

                rxList = srcList[pp].receiver_list
                nRx = len(rxList)
                waveObj = srcList[pp].waveform

                for qq in range(0, nRx):

                    times = rxList[qq].times
                    nLoc = np.shape(rxList[qq].locations)[0]

                    I = sp.diags(np.ones(nLoc))
                    eta = waveObj.getCharDecay(rxList[qq].fieldType, times)
                    eta = np.matrix(eta).T

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

        self.model = m   # Initiates/updates model and initiates mapping

        # Project to active mesh cells
        # m = np.matrix(self.xiMap * m).T
        m = np.matrix(self.xiMap * m).T

        # Must return as a numpy array
        return mkvc(sp.coo_matrix.dot(self.T, np.dot(self.A, m)))

    def Jvec(self, m, v, f=None):

        """Compute Pd*T*A*dxidm*v"""

        if self.survey is None:
            AssertionError("A survey must be set to generate A matrix")

        # Jacobian of xi wrt model
        dxidm = self.xiMap.deriv(m)

        # dxidm*v
        v = np.matrix(dxidm*v).T

        # Dot product with A
        v = self.A*v

        # Get active time rows of T
        T = self.T.tocsr()[self.survey.t_active, :]

        # Must return an array
        return mkvc(sp.csr_matrix.dot(T, v))

    def Jtvec(self, m, v, f=None):

        """Compute (Pd*T*A*dxidm)^T * v"""

        if self.survey is None:
            AssertionError("A survey must be set to generate A matrix")

        # Define v as a column vector
        v = np.matrix(v).T

        # Get T'*Pd'*v
        T = self.T.tocsr()[self.survey.t_active, :]
        v = sp.csc_matrix.dot(T.transpose(), v)

        # Multiply by A'
        v = (np.dot(v.T, self.A)).T

        # Jacobian of xi wrt model
        dxidm = self.xiMap.deriv(m)

        # Must return an array
        return mkvc(dxidm.T*v)


class Simulation3DLogUniform(BaseVRMSimulation):

    """

    """

    _A = None
    _T = None
    _TisSet = False
    # _xiMap = None

    chi0 = props.PhysicalProperty("DC susceptibility")
    dchi = props.PhysicalProperty("Frequency dependence")
    tau1 = props.PhysicalProperty("Low bound time-relaxation constant")
    tau2 = props.PhysicalProperty("Upper bound time-relaxation constant")

    def __init__(self, mesh, **kwargs):

        super(Simulation3DLogUniform, self).__init__(mesh, **kwargs)

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

            print('CREATING A MATRIX')

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
        srcList = self.survey.source_list
        nSrc = len(srcList)
        f = []

        for pp in range(0, nSrc):

            rxList = srcList[pp].receiver_list
            nRx = len(rxList)
            waveObj = srcList[pp].waveform

            for qq in range(0, nRx):

                times = rxList[qq].times
                eta = waveObj.getLogUniformDecay(
                    rxList[qq].fieldType, times, self.chi0, self.dchi, self.tau1, self.tau2
                )

                f.append(mkvc((self.A[qq] * np.matrix(eta)).T))

        return np.array(np.hstack(f))


############
# Deprecated
############

@deprecate_class(removal_version='0.15.0')
class Problem_Linear(Simulation3DLinear):
    pass


@deprecate_class(removal_version='0.15.0')
class Problem_LogUnifrom(Simulation3DLogUniform):
    pass
