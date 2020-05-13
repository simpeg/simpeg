from __future__ import print_function
from ...utils.code_utils import deprecate_class
from SimPEG import utils
from SimPEG.utils import mkvc, sdiag
from SimPEG import props
from ...simulation import BaseSimulation
from ..base import BasePFSimulation
import scipy.constants as constants
import numpy as np


class Simulation3DIntegral(BasePFSimulation):
    """
    Gravity simulation in integral form.

    """

    rho, rhoMap, rhoDeriv = props.Invertible(
        "Physical property",
        default=1.
    )

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self._G = None
        self._gtg_diagonal = None
        self.modelMap = self.rhoMap

    def fields(self, m):
        self.model = m

        if self.store_sensitivities == 'forward_only':
            self.model = m
            # Compute the linear operation without forming the full dense G
            fields = mkvc(self.linear_operator())
        else:
            fields = self.G@(self.rhoMap@m).astype(np.float32)

        return np.asarray(fields)

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        self.model = m

        if W is None:
            W = np.ones(self.nD)
        else:
            W = W.diagonal()**2
        if getattr(self, "_gtg_diagonal", None) is None:

            diag = np.zeros(self.G.shape[1])
            for i in range(len(W)):
                diag += W[i]*(self.G[i]*self.G[i])
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal
        return mkvc((sdiag(np.sqrt(diag))@self.rhoDeriv).power(2).sum(axis=0))

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """
        return self.G.dot(self.rhoDeriv)

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector
        """
        dmu_dm_v = (self.rhoDeriv @ v)
        return self.G@dmu_dm_v.astype(np.float32)

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transposed times a vector
        """
        Jtvec = self.G.T@v.astype(np.float32)
        return np.asarray(self.rhoDeriv.T@Jtvec)

    @property
    def G(self):
        """
        Gravity forward operator
        """
        if getattr(self, '_G', None) is None:

            self._G = self.linear_operator()

        return self._G

    @property
    def gtg_diagonal(self):
        """
        Diagonal of GtG
        """
        if getattr(self, '_gtg_diagonal', None) is None:

            return None

        return self._gtg_diagonal

    def evaluate_integral(self, receiver_location, components):
        """
            Compute the forward linear relationship between the model and the physics at a point
            and for every components of the survey.

            :param numpy.ndarray receiver_location:  array with shape (n_receivers, 3)
                Array of receiver locations as x, y, z columns.
            :param list[str] components: List of gravity components chosen from:
                'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'guv'

            :rtype numpy.ndarray: rows
            :returns: ndarray with shape (n_components, n_cells)
                Dense array mapping of the contribution of all active cells to data components::

                    rows =
                        g_1 = [g_1x g_1y g_1z]
                        g_2 = [g_2x g_2y g_2z]
                               ...
                        g_c = [g_cx g_cy g_cz]

        """
        eps = 1e-8

        dx = self.Xn - receiver_location[0]
        dy = self.Yn - receiver_location[1]
        dz = self.Zn - receiver_location[2]

        rows = {component: np.zeros(self.Xn.shape[0]) for component in components}

        gxx = np.zeros(self.Xn.shape[0])
        gyy = np.zeros(self.Xn.shape[0])

        for aa in range(2):
            for bb in range(2):
                for cc in range(2):

                    r = (
                            mkvc(dx[:, aa]) ** 2 +
                            mkvc(dy[:, bb]) ** 2 +
                            mkvc(dz[:, cc]) ** 2
                        ) ** (0.50) + eps

                    dz_r = dz[:, cc] + r + eps
                    dy_r = dy[:, bb] + r + eps
                    dx_r = dx[:, aa] + r + eps

                    dxr = dx[:, aa] * r + eps
                    dyr = dy[:, bb] * r + eps
                    dzr = dz[:, cc] * r + eps

                    dydz = dy[:, bb] * dz[:, cc]
                    dxdy = dx[:, aa] * dy[:, bb]
                    dxdz = dx[:, aa] * dz[:, cc]

                    if 'gx' in components:
                        rows['gx'] += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz_r) +
                            dz[:, cc] * np.log(dy_r) -
                            dx[:, aa] * np.arctan(dydz /
                                                  dxr)
                        )

                    if 'gy' in components:
                        rows['gy'] += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz_r) +
                            dz[:, cc] * np.log(dx_r) -
                            dy[:, bb] * np.arctan(dxdz /
                                                  dyr)
                        )

                    if 'gz' in components:
                        rows['gz'] += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy_r) +
                            dy[:, bb] * np.log(dx_r) -
                            dz[:, cc] * np.arctan(dxdy /
                                                  dzr)
                        )

                    arg = dy[:, bb] * dz[:, cc] / dxr

                    if ('gxx' in components) or ("gzz" in components) or ("guv" in components):
                        gxx -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r * dz_r + eps) +
                            dxdz / (r * dy_r + eps) -
                            np.arctan(arg+eps) +
                            dx[:, aa] * (1./ (1+arg**2.)) *
                            dydz/dxr**2. *
                            (r + dx[:, aa]**2./r)
                        )

                    if 'gxy' in components:
                        rows['gxy'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dz_r) + dy[:, bb]**2./ (r*dz_r) +
                            dz[:, cc] / r  -
                            1. / (1+arg**2.+ eps) * (dz[:, cc]/r**2) * (r - dy[:, bb]**2./r)

                        )

                    if 'gxz' in components:
                        rows['gxz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dy_r) + dz[:, cc]**2./ (r*dy_r) +
                            dy[:, bb] / r  -
                            1. / (1+arg**2.) * (dy[:, bb]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

                    arg = dx[:, aa]*dz[:, cc]/dyr

                    if ('gyy' in components) or ("gzz" in components) or ("guv" in components):
                        gyy -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r*dz_r+ eps) +
                            dydz / (r*dx_r+ eps) -
                            np.arctan(arg+eps) +
                            dy[:, bb] * (1./ (1+arg**2.+ eps)) *
                            dxdz/dyr**2. *
                            (r + dy[:, bb]**2./r)
                        )

                    if 'gyz' in components:
                        rows['gyz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dx_r) + dz[:, cc]**2./ (r*(dx_r)) +
                            dx[:, aa] / r  -
                            1. / (1+arg**2.) * (dx[:, aa]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

        if 'gyy' in components:
            rows['gyy'] = gyy

        if 'gxx' in components:
            rows['gxx'] = gxx

        if 'gzz' in components:
            rows['gzz'] = -gxx - gyy

        if 'guv' in components:
            rows['guv'] = -0.5*(gxx - gyy)

        for component in components:
            if len(component) == 3:
                rows[component] *= constants.G*1E12  # conversion for Eotvos
            else:
                rows[component] *= constants.G*1E8  # conversion for mGal

        return np.vstack([rows[component] for component in components])


class Simulation3DDifferential(BaseSimulation):
    """
        Gravity in differential equations!
    """

    _deprecate_main_map = 'rhoMap'

    rho, rhoMap, rhoDeriv = props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    solver = None

    def __init__(self, mesh, **kwargs):
        BaseSimulation.__init__(self, mesh, **kwargs)

        self.mesh.setCellGradBC('dirichlet')

        self._Div = self.mesh.cellGrad

    @property
    def MfI(self): return self._MfI

    @property
    def Mfi(self): return self._Mfi

    def makeMassMatrices(self, m):
        self.model = m
        self._Mfi = self.mesh.getFaceInnerProduct()
        self._MfI = utils.sdiag(1. / self._Mfi.diagonal())

    def getRHS(self, m):
        """


        """

        Mc = utils.sdiag(self.mesh.vol)

        self.model = m
        rho = self.rho

        return Mc * rho

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return -self._Div.T * self.Mfi * self._Div

    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell nodes [nC x 1]
            gField: defined on the cell faces [nF x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        from scipy.constants import G as NewtG

        self.makeMassMatrices(m)
        A = self.getA(m)
        RHS = self.getRHS(m)

        Ainv = self.solver(A)
        u = Ainv * RHS

        gField = 4. * np.pi * NewtG * 1e+8 * self._Div * u

        return {'G': gField, 'u': u}


############
# Deprecated
############

@deprecate_class(removal_version='0.15.0')
class GravityIntegral(Simulation3DIntegral):
    pass


@deprecate_class(removal_version='0.15.0')
class Problem3D_Diff(Simulation3DDifferential):
    pass
