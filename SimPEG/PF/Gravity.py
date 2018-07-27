from __future__ import print_function
from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Props, Mesh
from SimPEG.Utils import mkvc
import scipy as sp
from . import BaseGrav as GRAV
# from . import Forward
import re
import numpy as np
import multiprocessing
import scipy.constants as constants
import time


class GravityIntegral(Problem.LinearProblem):

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    # surveyPair = Survey.LinearSurvey
    forwardOnly = False  # Is TRUE, forward matrix not stored to memory
    actInd = None  #: Active cell indices provided
    rxType = 'z'
    silent = False
    memory_saving_mode = False
    parallelized = False
    n_cpu = None
    progressIndex = -1
    gtgdiag = None

    aa = []

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fields(self, m):
        self.model = self.rhoMap*m

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            fields = self.Intrgl_Fwr_Op()

            return mkvc(fields)

        else:
            vec = np.dot(self.G, (self.model).astype(np.float32))

            return vec.astype(np.float64)

    def mapping(self):
        """
            Return rhoMap
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

    def getJ(self, m, f):
        """
            Sensitivity matrix
        """
        return self.G

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
            print("Begin linear forward calculation: " + self.rxType)
            start = time.time()
            self._G = self.Intrgl_Fwr_Op()
            print("Linear forward calculation ended in: " + str(time.time()-start) + " sec")
        return self._G

    def Intrgl_Fwr_Op(self, m=None, rxType='z'):

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
        if isinstance(self.mesh, Mesh.TreeMesh):
            # Get upper and lower corners of each cell
            bsw = (self.mesh.gridCC -
                   np.kron(self.mesh.vol.T**(1./3.)/2.,
                           np.ones(3)).reshape((self.mesh.nC, 3)))
            tne = (self.mesh.gridCC +
                   np.kron(self.mesh.vol.T**(1./3.)/2.,
                           np.ones(3)).reshape((self.mesh.nC, 3)))

            xn1, xn2 = bsw[:, 0], tne[:, 0]
            yn1, yn2 = bsw[:, 1], tne[:, 1]
            zn1, zn2 = bsw[:, 2], tne[:, 2]

        else:

            xn = self.mesh.vectorNx
            yn = self.mesh.vectorNy
            zn = self.mesh.vectorNz

            yn2, xn2, zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
            yn1, xn1, zn1 = np.meshgrid(yn[:-1], xn[:-1], zn[:-1])

        self.Yn = P.T*np.c_[Utils.mkvc(yn1), Utils.mkvc(yn2)]
        self.Xn = P.T*np.c_[Utils.mkvc(xn1), Utils.mkvc(xn2)]
        self.Zn = P.T*np.c_[Utils.mkvc(zn1), Utils.mkvc(zn2)]

        self.rxLoc = self.survey.srcField.rxList[0].locs
        self.nD = int(self.rxLoc.shape[0])

        # if self.n_cpu is None:
        #     self.n_cpu = multiprocessing.cpu_count()

        # Switch to determine if the process has to be run in parallel
        job = Forward(
                rxLoc=self.rxLoc, Xn=self.Xn, Yn=self.Yn, Zn=self.Zn,
                n_cpu=self.n_cpu, forwardOnly=self.forwardOnly,
                model=self.model, rxType=self.rxType,
                parallelized=self.parallelized
                )

        G = job.calculate()

        return G

    @property
    def mapPair(self):
        """
            Call for general mapping of the problem
        """
        return self.rhoMap


class Forward(object):
    """
        Add docstring once it works
    """

    progressIndex = -1
    parallelized = False
    rxLoc = None
    Xn, Yn, Zn = None, None, None
    n_cpu = None
    forwardOnly = False
    model = None
    rxType = 'z'

    def __init__(self, **kwargs):
        super(Forward, self).__init__()
        Utils.setKwargs(self, **kwargs)

    def calculate(self):

        self.nD = self.rxLoc.shape[0]

        if self.parallelized:
            if self.n_cpu is None:

                # By default take half the cores, turns out be faster
                # than running full threads
                self.n_cpu = int(multiprocessing.cpu_count()/2)

            pool = multiprocessing.Pool(self.n_cpu)

            # rowInd = np.linspace(0, self.nD, self.n_cpu+1).astype(int)

            # job_args = []

            # for ii in range(self.n_cpu):

            #     nRows = int(rowInd[ii+1]-rowInd[ii])
            #     job_args += [(rowInd[ii], nRows, m)]

            # result = pool.map(self.getTblock, job_args)

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

    def calcTrow(self, xyzLoc):
        """
        Load in the active nodes of a tensor mesh and computes the gravity tensor
        for a given observation location xyzLoc[obsx, obsy, obsz]

        INPUT:
        Xn, Yn, Zn: Node location matrix for the lower and upper most corners of
                    all cells in the mesh shape[nC,2]
        M
        OUTPUT:
        Tx = [Txx Txy Txz]
        Ty = [Tyx Tyy Tyz]
        Tz = [Tzx Tzy Tzz]

        where each elements have dimension 1-by-nC.
        Only the upper half 5 elements have to be computed since symetric.
        Currently done as for-loops but will eventually be changed to vector
        indexing, once the topography has been figured out.

        """

        NewtG = constants.G*1e+8  # Convertion from mGal (1e-5) and g/cc (1e-3)
        eps = 1e-8  # add a small value to the locations to avoid

        # Pre-allocate space for 1D array
        row = np.zeros((1, self.Xn.shape[0]))

        dz = xyzLoc[2] - self.Zn

        dy = self.Yn - xyzLoc[1]

        dx = self.Xn - xyzLoc[0]

        # Compute contribution from each corners
        for aa in range(2):
            for bb in range(2):
                for cc in range(2):

                    r = (
                            mkvc(dx[:, aa]) ** 2 +
                            mkvc(dy[:, bb]) ** 2 +
                            mkvc(dz[:, cc]) ** 2
                        ) ** (0.50)

                    if self.rxType == 'x':
                        row = row - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz[:, cc] + r + eps) +
                            dz[:, cc] * np.log(dy[:, bb] + r + eps) -
                            dx[:, aa] * np.arctan(dy[:, bb] * dz[:, cc] /
                                                  (dx[:, aa] * r + eps)))

                    elif self.rxType == 'y':
                        row = row - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz[:, cc] + r + eps) +
                            dz[:, cc] * np.log(dx[:, aa] + r + eps) -
                            dy[:, bb] * np.arctan(dx[:, aa] * dz[:, cc] /
                                                  (dy[:, bb] * r + eps)))

                    else:
                        row -= NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy[:, bb] + r + eps) +
                            dy[:, bb] * np.log(dx[:, aa] + r + eps) -
                            dz[:, cc] * np.arctan(dx[:, aa] * dy[:, bb] /
                                                  (dz[:, cc] * r + eps)))

        if self.forwardOnly:
            return np.dot(row, self.model)
        else:
            return row

    def progress(self, iter, nRows):
        """
        progress(self, iter, nRows)

        Function measuring the progress of a process and print to screen the %.
        Useful to estimate the remaining runtime of a large problem.

        Created on Dec, 20th 2015

        @author: dominiquef
        """
        arg = np.floor(iter/nRows*10.)
        if arg > self.progressIndex:
            print("Done " + str(arg*10) + " %")
            self.progressIndex = arg


def writeUBCobs(filename, survey, d=None):
    """
    writeUBCobs(filename,survey,d)

    Function writing an observation file in UBC-GRAV3D format.

    INPUT
    filename    : Name of out file including directory
    survey
    flag          : dobs | dpred

    OUTPUT
    Obsfile

    """

    rxLoc = survey.srcField.rxList[0].locs

    wd = survey.std

    if d is None:
        d = survey.dobs

    data = np.c_[rxLoc, d, wd]

    head = '%i' % len(d)
    np.savetxt(
        filename, data, fmt='%e', delimiter=' ',
        newline='\n', header=head, comments='')

    print("Observation file saved to: " + filename)


def readUBCgravObs(obs_file, gravGrad=False):

    """
    Read UBC grav file format

    INPUT:
    :param fileName, path to the UBC obs grav file

    OUTPUT:
    :param survey

    """

    fid = open(obs_file, 'r')

    if gravGrad:
        line = fid.readline()
        nComp = len(line.split(','))

    # First line has the number of rows
    line = fid.readline()
    ndat = int(line.split()[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()
    temp = np.array(line.split(), dtype=float)

    if gravGrad:
        d = np.zeros((ndat, nComp), dtype=float)

    else:
        d = np.zeros(ndat, dtype=float)

    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    for ii in range(ndat):

        temp = np.array(line.split(), dtype=float)
        locXYZ[ii, :] = temp[:3]

        if gravGrad:
            d[ii, :] = temp[3:]

        else:
            d[ii] = temp[3]
            wd[ii] = temp[4]

        line = fid.readline()

    rxLoc = GRAV.RxObs(locXYZ)
    srcField = GRAV.SrcField([rxLoc])
    survey = GRAV.LinearSurvey(srcField)
    survey.dobs = d
    survey.std = wd
    return survey


class Problem3D_PDE(Problem.BaseProblem):
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
        # rho = self.rhoMap*m
        self._Mfi = self.mesh.getFaceInnerProduct()
        self._MfI = Utils.sdiag(1./self._Mfi.diagonal())

    def getRHS(self, m):
        """


        """

        Mc = Utils.sdiag(self.mesh.vol)

#        rho = self.rhoMap*m
        rho = m
        return Mc*rho

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div.T*self.Mfi*self._Div

    def fields(self, m):
        """
            Return gravity potential (u) and field (G)
            u: defined on the cell nodes [nC x 1]
            gField: defined on the cell faces [nG x 1]

            After we compute u, then we update G.

            .. math ::

                \mathbf{G}_s =

        """
        from scipy.constants import G as NewtG

        self.makeMassMatrices(m)
        A = self.getA(m)
        RHS = self.getRHS(m)

        if self.solver is None:
            m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/A.diagonal()))
            u, info = sp.linalg.bicgstab(A, RHS, tol=1e-6, maxiter=1000, M=m1)

        else:
            print("Solving with Paradiso")
            Ainv = self.solver(A)
            u = Ainv*RHS

        gField = 4.*np.pi*NewtG*1e+8*self._Div*u

        nFx = self.mesh.nFx
        nFy = self.mesh.nFy

        aveF2CCgx = self.mesh.aveFx2CC * gField[0:nFx]
        aveF2CCgy = self.mesh.aveFy2CC * gField[nFx:(nFx+nFy)]
        aveF2CCgz = self.mesh.aveFz2CC * gField[(nFx+nFy):]

        ggx = 1e+4*self.mesh.cellGrad * aveF2CCgx
        ggy = 1e+4*self.mesh.cellGrad * aveF2CCgy
        ggz = 1e+4*self.mesh.cellGrad * aveF2CCgz

        return {'G': gField, 'ggx': ggx, 'ggy': ggy, 'ggz': ggz, 'u': u}
