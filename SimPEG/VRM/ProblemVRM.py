from SimPEG import Problem, mkvc, Maps, Props, Survey
from SimPEG.VRM.SurveyVRM import SurveyVRM
from SimPEG.VRM.RxVRM import Point, SquareLoop
import numpy as np
import scipy.sparse as sp
import properties

############################################
# BASE VRM PROBLEM CLASS
############################################


class Problem_BaseVRM(Problem.BaseProblem):
    """

    """

    # SET CLASS ATTRIBUTES
    # _ref_factor = None
    # _ref_radius = None
    # _indActive = None
    _AisSet = False


    ref_factor = properties.Integer('Sensitivity refinement factor', min=0)
    ref_radius = properties.Array('Sensitivity refinement radii from sources', dtype=float)
    indActive = properties.Array('Topography active cells', dtype=bool)




    def __init__(self, mesh, **kwargs):

        ref_factor = kwargs.pop('ref_factor', None)
        ref_radius = kwargs.pop('ref_radius', None)
        indActive = kwargs.pop('indActive', None)
        assert len(mesh.h) == 3, 'Problem requires 3D tensor or OcTree mesh'

        super(Problem_BaseVRM, self).__init__(mesh, **kwargs)

        if ref_factor is None and ref_radius is None:
            self.ref_factor = 3
            self.ref_radius = list(1.25*np.mean(np.r_[np.min(mesh.h[0]), np.min(mesh.h[1]), np.min(mesh.h[2])])*np.arange(1, 4))
        elif ref_factor is None and ref_radius is not None:
            self.ref_factor = len(ref_radius)
            self.ref_radius = ref_radius
        elif ref_factor is not None and ref_radius is None:
            self.ref_factor = ref_factor
            self.ref_radius = list(1.25*np.mean(np.r_[np.min(mesh.h[0]), np.min(mesh.h[1]), np.min(mesh.h[2])])*np.arange(1, ref_factor+1))
        else:
            self.ref_factor = ref_factor
            self.ref_radius = ref_radius

        if indActive is None:
            self.indActive = np.ones(mesh.nC, dtype=bool)
        else:
            self.indActive = indActive

        # **kwargs
        # self._ref_factor = kwargs.get('ref_factor', 3)
        # self._ref_radius = kwargs.get('ref_radius', list(1.25*np.mean(np.r_[np.min(mesh.h[0]), np.min(mesh.h[1]), np.min(mesh.h[2])])*np.arange(1, self.ref_factor+1)))
        # self._indActive = kwargs.get('indActive', np.ones(mesh.nC, dtype=bool))

        # Assertions
        # assert isinstance(self._ref_factor, int), "Refinement factor must be set as an integer"
        # assert isinstance(self._ref_radius, list), "Refinement radii must be a list with at least 1 entry"
        # assert len(self._ref_radius) >= self._ref_factor, 'Number of refinement radii must equal or greater than refinement factor'
        # assert list(self._indActive).count(True) + list(self._indActive).count(False) == len(self._indActive), "indActive must be a boolean array"

        # if self.ref_factor > 4:
        #     print("Refinement factor larger than 4 may result in computations which exceed memory limits")

    @properties.observer('ref_factor')
    def _ref_factor_observer(self, change):
        if change['value'] > 4:
            print("Refinement factor larger than 4 may result in computations which exceed memory limits")
        if self.ref_radius is not None and change['value'] != len(self.ref_radius):
            print("Number of refinement radii currently DOES NOT match ref_factor")

    @properties.observer('ref_radius')
    def _ref_radius_validator(self, change):
        if self.ref_factor is not None and len(change['value']) != self.ref_factor:
            print("Number of refinement radii current DOES NOT match ref_factor")

    @properties.validator('indActive')
    def _ind_active_validator(self, change):
        assert len(change['value']) == self.mesh.nC, "Length of active topo cells array must equal number of mesh cells"


    # @property
    # def ref_factor(self):
    #     return self._ref_factor

    # @ref_factor.setter
    # def ref_factor(self, Val):

    #     assert isinstance(Val, int) and Val > -1, "Refinement factor must be an integer value equal or larger than 0"

    #     if Val != len(self._ref_radius):
    #         print("Refinement factor no longer matches length of refinement radii array. Please ensure refinement factor is equal or less to number of elements in refinement radii")

    #     if Val > 4:
    #         print("Refinement factor larger than 4 may result in computations which exceed memory limits")

    #     self._ref_factor = Val

    # @property
    # def ref_radius(self):
    #     return self._ref_radius

    # @ref_radius.setter
    # def ref_radius(self, radList):
    #     assert isinstance(radList, (list, tuple)), "Array must be a numpy array"

    #     if self._ref_factor != len(radList):
    #         print("Refinement factor no longer matches length of refinement radii array. Please ensure that the number of elements in refinement radii is equal or greater than the refinement factor")

    #     self._ref_radius = radList

    # @property
    # def indActive(self):
    #     return self._indActive

    # @indActive.setter
    # def indActive(self, Vec):

    #     assert list(self._indActive).count(True) + list(self._indActive).count(False) == len(self._indActive), "indActive must be a boolean array"
    #     self._AisSet = False
    #     self._indActive = Vec

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

        SrcObj = self.survey.srcList[pp]

        H0 = SrcObj.getH0(xyz)

        Hx0 = sp.diags(H0[:, 0], format="csr")
        Hy0 = sp.diags(H0[:, 1], format="csr")
        Hz0 = sp.diags(H0[:, 2], format="csr")

        H0 = sp.vstack([Hx0, Hy0, Hz0])

        return H0

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

        srcObj = self.survey.srcList[pp]

        N = np.shape(xyzc)[0]   # Number of cells
        K = srcObj.nRx          # Number of receiver in all rxList

        ax = np.reshape(xyzc[:, 0] - xyzh[:, 0]/2, (1, N))
        bx = np.reshape(xyzc[:, 0] + xyzh[:, 0]/2, (1, N))
        ay = np.reshape(xyzc[:, 1] - xyzh[:, 1]/2, (1, N))
        by = np.reshape(xyzc[:, 1] + xyzh[:, 1]/2, (1, N))
        az = np.reshape(xyzc[:, 2] - xyzh[:, 2]/2, (1, N))
        bz = np.reshape(xyzc[:, 2] + xyzh[:, 2]/2, (1, N))

        G = np.zeros((K, 3*N))
        C = -(1/(4*np.pi))
        tol = 1e-10   # Tolerance constant for numerical stability
        tol2 = 1000.  # Tolerance constant for numerical stability

        COUNT = 0

        for qq in range(0, len(srcObj.rxList)):

            rxObj = srcObj.rxList[qq]
            dComp = rxObj.fieldComp
            locs = rxObj.locs
            M = np.shape(locs)[0]

            if isinstance(rxObj, Point):

                if dComp.lower() == 'x':
                    for rr in range(0, M):
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

                        Gxx = (
                            np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+tol)) -
                            np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+tol)) +
                            np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+tol)) -
                            np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+tol)) +
                            np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+tol)) -
                            np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+tol)) +
                            np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+tol)) -
                            np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        Gyx = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-w1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)
                        )

                        Gzx = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-v1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-v1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-v2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-v2) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-v2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-v1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-v1) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-v2)
                        )

                        G[COUNT, :] = C*np.c_[Gxx, Gyx, Gzx]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'y':
                    for rr in range(0, M):
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

                        Gxy = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-w1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)
                        )

                        Gyy = (
                            np.arctan((u1*w1)/(v1*np.sqrt(u1**2+v1**2+w1**2)+tol)) -
                            np.arctan((u2*w1)/(v1*np.sqrt(u2**2+v1**2+w1**2)+tol)) +
                            np.arctan((u2*w1)/(v2*np.sqrt(u2**2+v2**2+w1**2)+tol)) -
                            np.arctan((u1*w1)/(v2*np.sqrt(u1**2+v2**2+w1**2)+tol)) +
                            np.arctan((u1*w2)/(v2*np.sqrt(u1**2+v2**2+w2**2)+tol)) -
                            np.arctan((u1*w2)/(v1*np.sqrt(u1**2+v1**2+w2**2)+tol)) +
                            np.arctan((u2*w2)/(v1*np.sqrt(u2**2+v1**2+w2**2)+tol)) -
                            np.arctan((u2*w2)/(v2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        Gzy = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-u1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-u2) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-u2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-u1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-u1) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-u1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-u2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-u2)
                        )

                        G[COUNT, :] = C*np.c_[Gxy, Gyy, Gzy]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'z':
                    for rr in range(0, M):
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

                        Gxz = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-v1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-v1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-v2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-v2) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-v2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-v1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-v1) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-v2)
                        )

                        Gyz = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-u1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-u2) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-u2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-u1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-u1) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-u1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-u2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-u2)
                        )

                        Gzz = (
                            - np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+tol)) +
                            np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+tol)) -
                            np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+tol)) +
                            np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+tol)) -
                            np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+tol)) +
                            np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+tol)) -
                            np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+tol)) +
                            np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        Gzz = (
                            Gzz -
                            np.arctan((u1*w1)/(v1*np.sqrt(u1**2+v1**2+w1**2)+tol)) +
                            np.arctan((u2*w1)/(v1*np.sqrt(u2**2+v1**2+w1**2)+tol)) -
                            np.arctan((u2*w1)/(v2*np.sqrt(u2**2+v2**2+w1**2)+tol)) +
                            np.arctan((u1*w1)/(v2*np.sqrt(u1**2+v2**2+w1**2)+tol)) -
                            np.arctan((u1*w2)/(v2*np.sqrt(u1**2+v2**2+w2**2)+tol)) +
                            np.arctan((u1*w2)/(v1*np.sqrt(u1**2+v1**2+w2**2)+tol)) -
                            np.arctan((u2*w2)/(v1*np.sqrt(u2**2+v1**2+w2**2)+tol)) +
                            np.arctan((u2*w2)/(v2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        G[COUNT, :] = C*np.c_[Gxz, Gyz, Gzz]
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
                s1 = (rxObj.width/2)*np.reshape(np.kron(ds[rxObj.quadOrder-1], np.ones(nw)), (nw**2, 1))
                s2 = (rxObj.width/2)*np.reshape(np.kron(np.ones(nw), ds[rxObj.quadOrder-1]), (nw**2, 1))

                if dComp.lower() == 'x':
                    for rr in range(0, M):

                        u1 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax)
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx)
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2
                        v1 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay) + np.kron(s1, np.ones((1, N)))
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by) + np.kron(s1, np.ones((1, N)))
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2
                        w1 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az) + np.kron(s2, np.ones((1, N)))
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz) + np.kron(s2, np.ones((1, N)))
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        Gxx = (
                            np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+tol)) -
                            np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+tol)) +
                            np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+tol)) -
                            np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+tol)) +
                            np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+tol)) -
                            np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+tol)) +
                            np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+tol)) -
                            np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        Gxx = np.dot(wt, Gxx)

                        Gyx = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-w1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)
                        )

                        Gyx = np.dot(wt, Gyx)

                        Gzx = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-v1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-v1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-v2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-v2) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-v2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-v1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-v1) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-v2)
                        )

                        Gzx = np.dot(wt, Gzx)

                        G[COUNT, :] = C*np.c_[Gxx, Gyx, Gzx]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'y':
                    for rr in range(0, M):

                        u1 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax) + np.kron(s1, np.ones((1, N)))
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx) + np.kron(s1, np.ones((1, N)))
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2
                        v1 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay)
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by)
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2
                        w1 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az) + np.kron(s2, np.ones((1, N)))
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz) + np.kron(s2, np.ones((1, N)))
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        Gxy = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-w1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)
                        )

                        Gxy = np.dot(wt, Gxy)

                        Gyy = (
                            np.arctan((u1*w1)/(v1*np.sqrt(u1**2+v1**2+w1**2)+tol)) -
                            np.arctan((u2*w1)/(v1*np.sqrt(u2**2+v1**2+w1**2)+tol)) +
                            np.arctan((u2*w1)/(v2*np.sqrt(u2**2+v2**2+w1**2)+tol)) -
                            np.arctan((u1*w1)/(v2*np.sqrt(u1**2+v2**2+w1**2)+tol)) +
                            np.arctan((u1*w2)/(v2*np.sqrt(u1**2+v2**2+w2**2)+tol)) -
                            np.arctan((u1*w2)/(v1*np.sqrt(u1**2+v1**2+w2**2)+tol)) +
                            np.arctan((u2*w2)/(v1*np.sqrt(u2**2+v1**2+w2**2)+tol)) -
                            np.arctan((u2*w2)/(v2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        Gyy = np.dot(wt, Gyy)

                        Gzy = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-u1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-u2) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-u2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-u1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-u1) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-u1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-u2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-u2)
                        )

                        Gzy = np.dot(wt, Gzy)

                        G[COUNT, :] = C*np.c_[Gxy, Gyy, Gzy]
                        COUNT = COUNT + 1

                elif dComp.lower() == 'z':
                    for rr in range(0, M):

                        u1 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - ax) + np.kron(s1, np.ones((1, N)))
                        u1[np.abs(u1) < tol] = np.min(xyzh[:, 0])/tol2
                        u2 = np.kron(np.ones((nw**2, 1)), locs[rr, 0] - bx) + np.kron(s1, np.ones((1, N)))
                        u2[np.abs(u2) < tol] = -np.min(xyzh[:, 0])/tol2
                        v1 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - ay) + np.kron(s2, np.ones((1, N)))
                        v1[np.abs(v1) < tol] = np.min(xyzh[:, 1])/tol2
                        v2 = np.kron(np.ones((nw**2, 1)), locs[rr, 1] - by) + np.kron(s2, np.ones((1, N)))
                        v2[np.abs(v2) < tol] = -np.min(xyzh[:, 1])/tol2
                        w1 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - az)
                        w1[np.abs(w1) < tol] = np.min(xyzh[:, 2])/tol2
                        w2 = np.kron(np.ones((nw**2, 1)), locs[rr, 2] - bz)
                        w2[np.abs(w2) < tol] = -np.min(xyzh[:, 2])/tol2

                        Gxz = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-v1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-v1) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-v2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-v2) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-v2) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-v1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-v1) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-v2)
                        )

                        Gxz = np.dot(wt, Gxz)

                        Gyz = (
                            np.log(np.sqrt(u1**2+v1**2+w1**2)-u1) -
                            np.log(np.sqrt(u2**2+v1**2+w1**2)-u2) +
                            np.log(np.sqrt(u2**2+v2**2+w1**2)-u2) -
                            np.log(np.sqrt(u1**2+v2**2+w1**2)-u1) +
                            np.log(np.sqrt(u1**2+v2**2+w2**2)-u1) -
                            np.log(np.sqrt(u1**2+v1**2+w2**2)-u1) +
                            np.log(np.sqrt(u2**2+v1**2+w2**2)-u2) -
                            np.log(np.sqrt(u2**2+v2**2+w2**2)-u2)
                        )

                        Gyz = np.dot(wt, Gyz)

                        Gzz = (
                            - np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+tol)) +
                            np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+tol)) -
                            np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+tol)) +
                            np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+tol)) -
                            np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+tol)) +
                            np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+tol)) -
                            np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+tol)) +
                            np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        Gzz = (
                            Gzz -
                            np.arctan((u1*w1)/(v1*np.sqrt(u1**2+v1**2+w1**2)+tol)) +
                            np.arctan((u2*w1)/(v1*np.sqrt(u2**2+v1**2+w1**2)+tol)) -
                            np.arctan((u2*w1)/(v2*np.sqrt(u2**2+v2**2+w1**2)+tol)) +
                            np.arctan((u1*w1)/(v2*np.sqrt(u1**2+v2**2+w1**2)+tol)) -
                            np.arctan((u1*w2)/(v2*np.sqrt(u1**2+v2**2+w2**2)+tol)) +
                            np.arctan((u1*w2)/(v1*np.sqrt(u1**2+v1**2+w2**2)+tol)) -
                            np.arctan((u2*w2)/(v1*np.sqrt(u2**2+v1**2+w2**2)+tol)) +
                            np.arctan((u2*w2)/(v2*np.sqrt(u2**2+v2**2+w2**2)+tol))
                        )

                        Gzz = np.dot(wt, Gzz)

                        G[COUNT, :] = C*np.c_[Gxz, Gyz, Gzz]
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
            ref_factor = self.ref_factor
            ref_radius = self.ref_radius

            if ref_factor > 0:

                srcObj = self.survey.srcList[pp]
                refFlag = srcObj._getRefineFlags(xyzc, ref_factor, ref_radius)

                for qq in range(1, ref_factor+1):
                    if len(refFlag[refFlag == qq]) != 0:
                        A[pp][:, refFlag == qq] = self._getSubsetAcolumns(xyzc, xyzh, pp, qq, refFlag)

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
        [nx, ny, nz] = np.meshgrid(np.linspace(1, n, n)-0.5, np.linspace(1, n, n)-0.5, np.linspace(1, n, n)-0.5)
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


#############################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND INVERSION)
#############################################################################


class Problem_Linear(Problem_BaseVRM):

    """

    """

    _A = None
    _T = None
    _TisSet = False
    _xiMap = None

    surveyPair = SurveyVRM     # Only linear problem can have survey and be inverted

    # xi = Props.PhysicalProperty("Amalgamated Viscous Remanent Magnetization Parameter xi = dchi/ln(tau2/tau1)")
    xi, xiMap, xiDeriv = Props.Invertible("Amalgamated Viscous Remanent Magnetization Parameter xi = dchi/ln(tau2/tau1)")

    def __init__(self, mesh, **kwargs):

        super(Problem_Linear, self).__init__(mesh, **kwargs)

        nAct = list(self.indActive).count(True)
        if self.xiMap is None:
            self.xiMap = Maps.IdentityMap(nP=nAct)

    @property
    def A(self):

        """
        The geometric sensitivity matrix for the linear VRM problem. Accessing
        this property requires that the problem be paired with a survey object.

        """

        if self._AisSet is False:

            assert self.ispaired, "Problem must be paired with survey to generate A matrix"

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

            assert self.ispaired, "Problem must be paired with survey to generate T matrix"

            # Remove any previously stored T matrix
            if self._T is not None:
                self._T = None

            print('CREATING T MATRIX')

            srcList = self.survey.srcList
            nSrc = len(srcList)
            T = []

            for pp in range(0, nSrc):

                rxList = srcList[pp].rxList
                nRx = len(rxList)
                waveObj = srcList[pp].waveform

                for qq in range(0, nRx):

                    times = rxList[qq].times
                    nLoc = np.shape(rxList[qq].locs)[0]

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

        assert self.ispaired, "Problem must be paired with survey to predict data"

        self.model = m   # Initiates/updates model and initiates mapping

        # Project to active mesh cells
        # m = np.matrix(self.xiMap * m).T
        m = np.matrix(self.xiMap * m).T

        # Must return as a numpy array
        return mkvc(sp.coo_matrix.dot(self.T, np.dot(self.A, m)))

    def Jvec(self, m, v, f=None):

        """Compute Pd*T*A*dxidm*v"""

        assert self.ispaired, "Problem must be paired with survey to predict data"

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

        assert self.ispaired, "Problem must be paired with survey to predict data"

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

    def unpair(self):
        """Unbind a survey from this problem instance."""
        if not self.ispaired:
            return
        self.survey._prob = None
        self._survey = None
        self._A = None
        self._T = None
        self._AisSet = False
        self._TisSet = False


class Problem_LogUniform(Problem_BaseVRM):

    """

    """

    _A = None
    _T = None
    _TisSet = False
    # _xiMap = None

    surveyPair = Survey.BaseSurvey

    chi0 = Props.PhysicalProperty("DC susceptibility")
    dchi = Props.PhysicalProperty("Frequency dependence")
    tau1 = Props.PhysicalProperty("Low bound time-relaxation constant")
    tau2 = Props.PhysicalProperty("Upper bound time-relaxation constant")

    def __init__(self, mesh, **kwargs):

        super(Problem_LogUniform, self).__init__(mesh, **kwargs)

    @property
    def A(self):

        """
        The geometric sensitivity matrix for the linear VRM problem. Accessing
        this property requires that the problem be paired with a survey object.

        """

        if self._AisSet is False:

            assert self.ispaired, "Problem must be paired with survey to generate A matrix"

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

        assert self.ispaired, "Problem must be paired with survey to predict data"

        # Fields from each source
        srcList = self.survey.srcList
        nSrc = len(srcList)
        f = []

        for pp in range(0, nSrc):

            rxList = srcList[pp].rxList
            nRx = len(rxList)
            waveObj = srcList[pp].waveform

            for qq in range(0, nRx):

                times = rxList[qq].times
                eta = waveObj.getLogUniformDecay(rxList[qq].fieldType, times, self.chi0, self.dchi, self.tau1, self.tau2)

                f.append(mkvc((self.A[qq] * np.matrix(eta)).T))

        return np.array(np.hstack(f))
