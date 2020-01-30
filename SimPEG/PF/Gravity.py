from __future__ import print_function
from SimPEG import Problem, Mesh
from SimPEG import Utils
from SimPEG.Utils import mkvc, matutils, sdiag
from SimPEG import Props
import scipy as sp
import scipy.constants as constants
import os
import time
import numpy as np
from scipy.sparse import csr_matrix as csr


class GravityIntegral(Problem.LinearProblem):

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    # surveyPair = Survey.LinearSurvey
    forwardOnly = False  # Is TRUE, forward matrix not stored to memory
    actInd = None  #: Active cell indices provided
    silent = False
    memory_saving_mode = False
    parallelized = False
    n_cpu = None
    progress_index = -1
    gtgdiag = None

    aa = []

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        if getattr(self, 'actInd', None) is not None:

            if self.actInd.dtype == 'bool':
                inds = np.asarray([inds for inds,
                                  elem in enumerate(self.actInd, 1)
                                  if elem], dtype=int) - 1
            else:
                inds = self.actInd

        else:

            inds = np.asarray(range(self.mesh.nC))

        self.nC = len(inds)

        # Create active cell projector
        P = sp.sparse.csr_matrix(
            (np.ones(self.nC), (inds, range(self.nC))),
            shape=(self.mesh.nC, self.nC)
        )

        # Create vectors of nodal location
        # (lower and upper corners for each cell)
        bsw = (self.mesh.gridCC - self.mesh.h_gridded/2.)
        tne = (self.mesh.gridCC + self.mesh.h_gridded/2.)

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]

        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]

    def fields(self, m):

        model = self.rhoMap*m

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            fields = self.Intrgl_Fwr_Op(m=m)

            return mkvc(fields)

        else:
            vec = np.dot(self.G, model.astype(np.float32))

            return vec.astype(np.float64)


    def modelMap(self):
        """
            Call for general mapping of the problem
        """
        return self.rhoMap

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        if self.gtgdiag is None:

            if W is None:
                w = np.ones(self.G.shape[1])
            else:
                w = W.diagonal()

            dmudm = self.rhoMap.deriv(m)
            self.gtgdiag = np.zeros(dmudm.shape[1])

            for ii in range(self.G.shape[0]):

                self.gtgdiag += (w[ii]*self.G[ii, :]*dmudm)**2.

        return self.gtgdiag


    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """

        dmudm = self.rhoMap.deriv(m)
        return self.G.got(dmudm)

    def Jvec(self, m, v, f=None):
        dmudm = self.rhoMap.deriv(m)
        return self.G.dot(dmudm*v)

    def Jtvec(self, m, v, f=None):

        dmudm = self.rhoMap.deriv(m)
        return dmudm.T * (self.G.T.dot(v))

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:
            print("Begin linear forward calculation: ")
            start = time.time()

            self._G = self.Intrgl_Fwr_Op()
            print("Linear forward calculation ended in: " + str(time.time()-start) + " sec")

        return self._G


    def Intrgl_Fwr_Op(self, m=None):

        """

        Gravity forward operator in integral form

        flag        = 'z' | 'xyz'

        Return
        _G        = Linear forward modeling operation

        Created on March, 15th 2016

        @author: dominiquef

         """
        if m is not None:
            self.model = self.rhoMap*m
        self.rxLoc = self.survey.srcField.rxList[0].locs
        self.nD = int(self.rxLoc.shape[0])

        # if self.n_cpu is None:
        #     self.n_cpu = multiprocessing.cpu_count()

        # Switch to determine if the process has to be run in parallel
        job = Forward(
                rxLoc=self.rxLoc, Xn=self.Xn, Yn=self.Yn, Zn=self.Zn,
                n_cpu=self.n_cpu, forwardOnly=self.forwardOnly,
                model=self.model, components=self.survey.components,
                parallelized=self.parallelized
        )

        G = job.calculate()

        return G


class Forward(object):
    """
        Add docstring once it works
    """

    progress_index = -1
    parallelized = False
    rxLoc = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    forwardOnly = False
    model = None
    components = ['gz']

    def __init__(self, **kwargs):
        super(Forward, self).__init__()
        Utils.setKwargs(self, **kwargs)

    def calculate(self):

        self.nD = self.rxLoc.shape[0]
        self.nC = self.Xn.shape[0]

        if self.parallelized:
            if self.n_cpu is None:

                # By default take half the cores, turns out be faster
                # than running full threads
                self.n_cpu = int(multiprocessing.cpu_count()/2)

            pool = multiprocessing.Pool(self.n_cpu)

            result = pool.map(self.calcTrow, [self.rxLoc[ii, :] for ii in range(self.nD)])
            pool.close()
            pool.join()

        else:

            result = []
            for ii in range(self.nD):
                result += [self.calcTrow(self.rxLoc[ii, :])]
                self.progress(ii, self.nD)

        if self.forwardOnly:
            return mkvc(np.vstack(result))

        else:
            return np.vstack(result)

    def calcTrow(self, receiver_location):
        """
            Load in the active nodes of a tensor mesh and computes the magnetic
            forward relation between a cuboid and a given observation
            location outside the Earth [obsx, obsy, obsz]

            INPUT:
            xyzLoc:  [obsx, obsy, obsz] nC x 3 Array

            OUTPUT:
            Tx = [Txx Txy Txz]
            Ty = [Tyx Tyy Tyz]
            Tz = [Tzx Tzy Tzz]

        """
        eps = 1e-8

        NewtG = constants.G*1e+8

        dx = self.Xn - receiver_location[0]
        dy = self.Yn - receiver_location[1]
        dz = self.Zn - receiver_location[2]

        compDict = {key: np.zeros(self.Xn.shape[0]) for key in self.components}

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

                    if 'gx' in self.components:
                        compDict['gx'] += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz_r) +
                            dz[:, cc] * np.log(dy_r) -
                            dx[:, aa] * np.arctan(dydz /
                                                  dxr)
                        )

                    if 'gy' in self.components:
                        compDict['gy']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz_r) +
                            dz[:, cc] * np.log(dx_r) -
                            dy[:, bb] * np.arctan(dxdz /
                                                  dyr)
                        )

                    if 'gz' in self.components:
                        compDict['gz']  += (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy_r) +
                            dy[:, bb] * np.log(dx_r) -
                            dz[:, cc] * np.arctan(dxdy /
                                                  dzr)
                        )

                    arg = dy[:, bb] * dz[:, cc] / dxr

                    if ('gxx' in self.components) or ("gzz" in self.components) or ("guv" in self.components):
                        gxx -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r * dz_r + eps) +
                            dxdz / (r * dy_r + eps) -
                            np.arctan(arg+eps) +
                            dx[:, aa] * (1./ (1+arg**2.)) *
                            dydz/dxr**2. *
                            (r + dx[:, aa]**2./r)
                        )

                    if 'gxy' in self.components:
                        compDict['gxy'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dz_r) + dy[:, bb]**2./ (r*dz_r) +
                            dz[:, cc] / r  -
                            1. / (1+arg**2.+ eps) * (dz[:, cc]/r**2) * (r - dy[:, bb]**2./r)

                        )

                    if 'gxz' in self.components:
                        compDict['gxz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dy_r) + dz[:, cc]**2./ (r*dy_r) +
                            dy[:, bb] / r  -
                            1. / (1+arg**2.) * (dy[:, bb]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

                    arg = dx[:, aa]*dz[:, cc]/dyr

                    if ('gyy' in self.components) or ("gzz" in self.components) or ("guv" in self.components):
                        gyy -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dxdy / (r*dz_r+ eps) +
                            dydz / (r*dx_r+ eps) -
                            np.arctan(arg+eps) +
                            dy[:, bb] * (1./ (1+arg**2.+ eps)) *
                            dxdz/dyr**2. *
                            (r + dy[:, bb]**2./r)
                        )

                    if 'gyz' in self.components:
                        compDict['gyz'] -= (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            np.log(dx_r) + dz[:, cc]**2./ (r*(dx_r)) +
                            dx[:, aa] / r  -
                            1. / (1+arg**2.) * (dx[:, aa]/(r**2)) * (r - dz[:, cc]**2./r)

                        )

        if 'gyy' in self.components:
            compDict['gyy'] = gyy

        if 'gxx' in self.components:
            compDict['gxx'] = gxx

        if 'gzz' in self.components:
            compDict['gzz'] = -gxx - gyy

        if 'guv' in self.components:
            compDict['guv'] = -0.5 * (gxx - gyy)

        return np.vstack([NewtG * compDict[key] for key in list(compDict.keys())])


    def progress(self, ind, total):
        """
        progress(ind,prog,final)

        Function measuring the progress of a process and print to screen the %.
        Useful to estimate the remaining runtime of a large problem.

        Created on Dec, 20th 2015

        @author: dominiquef
        """
        arg = np.floor(ind/total*10.)
        if arg > self.progress_index:
            print("Done " + str(arg*10) + " %")
            self.progress_index = arg


class Problem3D_Diff(Problem.BaseProblem):
    """
        Gravity in differential equations!
    """

    _depreciate_main_map = 'rhoMap'

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    solver = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        self.mesh.setCellGradBC('dirichlet')

        self._Div = self.mesh.cellGrad

    @property
    def MfI(self): return self._MfI

    @property
    def Mfi(self): return self._Mfi

    def makeMassMatrices(self, m):
        self.model = m
        self._Mfi = self.mesh.getFaceInnerProduct()
        self._MfI = Utils.sdiag(1. / self._Mfi.diagonal())

    def getRHS(self, m):
        """


        """

        Mc = Utils.sdiag(self.mesh.vol)

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
                Utils.sdiag(1 / A.diagonal())
            )
            u, info = sp.linalg.bicgstab(A, RHS, tol=1e-6, maxiter=1000, M=m1)

        else:
            print("Solving with Paradiso")
            Ainv = self.solver(A)
            u = Ainv * RHS

        gField = 4. * np.pi * NewtG * 1e+8 * self._Div * u

        return {'G': gField, 'u': u}
