from __future__ import print_function
from SimPEG import utils
from SimPEG.utils import mkvc, sdiag
from SimPEG import props
from ...simulation import BaseSimulation
from ..base import BasePFSimulation
import scipy as sp
import scipy.constants as constants
import os
import numpy as np
import dask
import dask.array as da
from scipy.sparse import csr_matrix as csr
from dask.diagnostics import ProgressBar
import multiprocessing


class GravityIntegralSimulation(BasePFSimulation):
    """
    Gravity simulation in integral form.

    """

    rho, rhoMap, rhoMapDeriv = props.Invertible(
        "Physical property",
        default=1.
    )

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self._G = None
        self._gtg_diagonal = None
        self.modelMap = self.rhoMap

    def fields(self, m):
        # self.model = self.rhoMap*m

        if not self.store_sensitivity:

            # Compute the linear operation without forming the full dense G
            return np.array(self.linear_operator(m=m), dtype='float')

        else:

            return da.dot(self.G, (self.rhoMap*m).astype(np.float32)).compute()

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        self.model = m

        if self.gtg_diagonal is None:

            if W is None:
                w = np.ones(self.G.shape[1])
            else:
                w = W.diagonal()

            self._gtg_diagonal = da.sum(self.G**2., 0).compute()

        return mkvc(
            np.sum((
                sdiag(mkvc(self.gtg_diagonal)**0.5) * self.rhoMap.deriv(m)
            ).power(2.), axis=0)
        )

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """
        return da.dot(self.G, self.rhoMap.deriv(m))

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector
        """
        dmu_dm_v = da.from_array(
            self.rhoMap.deriv(m)*v, chunks=self.G.chunks[1]
        )

        return da.dot(self.G, dmu_dm_v.astype(np.float32))

    def Jtvec(self, m, v, f=None):
        """
        Sensitivity transposed times a vector
        """
        Jtvec = da.dot(v.astype(np.float32), self.G)
        dmudm_v = dask.delayed(csr.dot)(Jtvec, self.rhoMap.deriv(m))

        return da.from_delayed(
            dmudm_v, dtype=float, shape=[self.rhoMap.deriv(m).shape[1]]
        ).compute()

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

    def evaluate_integral(self, receiver_location):
        """
            Compute the forward linear relationship between the model and the physics at a point
            and for every components of the survey.

            INPUT:
            receiver_location:  numpy.ndarray (n_receivers, 3)
                Array of receiver locations as x, y, z columns.

            OUTPUT:
            G_1 = [g_1x g_1y g_1z]
            G_2 = [g_2x g_2y g_2z]
            ...
            G_c = [g_cx g_cy g_cz]

        """
        eps = 1e-8

        dx = self.Xn - receiver_location[0]
        dy = self.Yn - receiver_location[1]
        dz = self.Zn - receiver_location[2]

        components = {key: np.zeros(self.Xn.shape[0]) for key in self.survey.components}

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

                    if 'gx' in self.survey.components.keys():
                        components['gx'] += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz_r) +
                            dz[:, cc] * np.log(dy_r) -
                            dx[:, aa] * np.arctan(dydz /
                                                  dxr)
                        )

                    if 'gy' in self.survey.components.keys():
                        components['gy']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz_r) +
                            dz[:, cc] * np.log(dx_r) -
                            dy[:, bb] * np.arctan(dxdz /
                                                  dyr)
                        )

                    if 'gz' in self.survey.components.keys():
                        components['gz']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy_r) +
                            dy[:, bb] * np.log(dx_r) -
                            dz[:, cc] * np.arctan(dxdy /
                                                  dzr)
                        )

                    arg = dy[:, bb] * dz[:, cc] / dxr

                    if ('gxx' in self.survey.components.keys()) or ("gzz" in self.survey.components.keys()) or ("guv" in self.survey.components.keys()):
                        gxx -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r * dz_r + eps) +
                            dxdz / (r * dy_r + eps) -
                            np.arctan(arg+eps) +
                            dx[:, aa] * (1./ (1+arg**2.)) *
                            dydz/dxr**2. *
                            (r + dx[:, aa]**2./r)
                        )

                    if 'gxy' in self.survey.components.keys():
                        components['gxy'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dz_r) + dy[:, bb]**2./ (r*dz_r) +
                            dz[:, cc] / r  -
                            1. / (1+arg**2.+ eps) * (dz[:, cc]/r**2) * (r - dy[:, bb]**2./r)

                        )

                    if 'gxz' in self.survey.components.keys():
                        components['gxz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dy_r) + dz[:, cc]**2./ (r*dy_r) +
                            dy[:, bb] / r  -
                            1. / (1+arg**2.) * (dy[:, bb]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

                    arg = dx[:, aa]*dz[:, cc]/dyr

                    if ('gyy' in self.survey.components.keys()) or ("gzz" in self.survey.components.keys()) or ("guv" in self.survey.components.keys()):
                        gyy -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r*dz_r+ eps) +
                            dydz / (r*dx_r+ eps) -
                            np.arctan(arg+eps) +
                            dy[:, bb] * (1./ (1+arg**2.+ eps)) *
                            dxdz/dyr**2. *
                            (r + dy[:, bb]**2./r)
                        )

                    if 'gyz' in self.survey.components.keys():
                        components['gyz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dx_r) + dz[:, cc]**2./ (r*(dx_r)) +
                            dx[:, aa] / r  -
                            1. / (1+arg**2.) * (dx[:, aa]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

        if 'gyy' in self.survey.components.keys():
            components['gyy'] = gyy

        if 'gxx' in self.survey.components.keys():
            components['gxx'] = gxx

        if 'gzz' in self.survey.components.keys():
            components['gzz'] = -gxx - gyy

        if 'guv' in self.survey.components.keys():
            components['guv'] = -0.5*(gxx - gyy)

        return np.vstack([constants.G * 1e+8 * components[key] for key in list(components.keys())])


class DifferentialEquationSimulation(BaseSimulation):
    """
        Gravity in differential equations!
    """

    _depreciate_main_map = 'rhoMap'

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

        if self.solver is None:
            m1 = sp.linalg.interface.aslinearoperator(
                utils.sdiag(1 / A.diagonal())
            )
            u, info = sp.linalg.bicgstab(A, RHS, tol=1e-6, maxiter=1000, M=m1)

        else:
            print("Solving with Paradiso")
            Ainv = self.solver(A)
            u = Ainv * RHS

        gField = 4. * np.pi * NewtG * 1e+8 * self._Div * u

        return {'G': gField, 'u': u}
