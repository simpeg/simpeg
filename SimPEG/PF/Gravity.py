from __future__ import print_function
from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Props
from SimPEG.Utils import mkvc
import scipy.sparse as sp
from . import BaseGrav as GRAV
import re
import numpy as np



class GravityIntegral(Problem.LinearProblem):

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Specific density (g/cc)",
        default=1.
    )

    # surveyPair = Survey.LinearSurvey
    forwardOnly = False  # Is TRUE, forward matrix not stored to memory
    actInd = None  #: Active cell indices provided
    rType = 'z'
    silent = False
    memory_saving_mode = False


    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fields(self, m):
        self.model = self.rhoMap*m

        if self.forwardOnly:

            # Compute the linear operation without forming the full dense G
            fields = self.Intrgl_Fwr_Op(m=m)

            return fields

        else:
            vec = np.dot(self.F, (self.rhoMap*(m)).astype(np.float32))

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

            if W is None:
                W = sp.speye(self.F.shape[0])

            dmudm = self.rhoMap.deriv(m)

            if self.memory_saving_mode:
                wd = W.diagonal()
                JtJdiag = np.zeros_like(m)
                for ii in range(self.F.shape[0]):
                    JtJdiag += ((wd[ii] * self.F[ii, :]) * dmudm)**2.

                return JtJdiag

            else:
                return np.sum((W * self.F * dmudm)**2., axis=0)

    def getJ(self, m, f):
        """
            Sensitivity matrix
        """
        return self.F

    def Jvec(self, m, v, f=None):
        dmudm = self.rhoMap.deriv(m)
        return self.F.dot(dmudm*v)

    def Jtvec(self, m, v, f=None):
        dmudm = self.rhoMap.deriv(m)
        return dmudm.T * (self.F.T.dot(v))

    @property
    def F(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_F', None) is None:
            self._F = self.Intrgl_Fwr_Op('z')

        return self._F

    def Intrgl_Fwr_Op(self, m=None, rType='z'):

        """

        Gravity forward operator in integral form

        flag        = 'z' | 'xyz'

        Return
        _F        = Linear forward modeling operation

        Created on March, 15th 2016

        @author: dominiquef

         """
        # Find non-zero cells
        # inds = np.nonzero(actv)[0]
        if getattr(self, 'actInd', None) is not None:

            if self.actInd.dtype == 'bool':
                inds = np.asarray([inds for inds,
                                  elem in enumerate(self.actInd, 1)
                                  if elem], dtype=int) - 1
            else:
                inds = self.actInd

        else:

            inds = np.asarray(range(self.mesh.nC))

        nC = len(inds)

        # Create active cell projector
        P = sp.csr_matrix(
            (np.ones(nC), (inds, range(nC))),
            shape=(self.mesh.nC, nC)
        )

        # Create vectors of nodal location
        # (lower and upper corners for each cell)
        xn = self.mesh.vectorNx
        yn = self.mesh.vectorNy
        zn = self.mesh.vectorNz

        yn2, xn2, zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
        yn1, xn1, zn1 = np.meshgrid(yn[0:-1], xn[0:-1], zn[0:-1])

        Yn = P.T*np.c_[Utils.mkvc(yn1), Utils.mkvc(yn2)]
        Xn = P.T*np.c_[Utils.mkvc(xn1), Utils.mkvc(xn2)]
        Zn = P.T*np.c_[Utils.mkvc(zn1), Utils.mkvc(zn2)]

        rxLoc = self.survey.srcField.rxList[0].locs
        ndata = int(rxLoc.shape[0])

        # Pre-allocate space and create magnetization matrix if required
        # Pre-allocate space
        if self.forwardOnly:

            F = np.empty(ndata, dtype='float64')

        else:

            F = np.zeros((ndata, nC), dtype=np.float32)

        # Loop through all observations
        print("Begin linear forward calculation: " + self.rType)

        # Add counter to dsiplay progress. Good for large problems
        count = -1
        for ii in range(ndata):

            tx, ty, tz = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])

                        # Add counter to dsiplay progress. Good for large problems

            if self.forwardOnly:
                tx, ty, tz = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])

                if self.rType == 'x':
                    F[ii] = tx.dot(m)

                elif self.rType == 'y':
                    F[ii] = ty.dot(m)

                elif self.rType == 'z':
                    F[ii] = tz.dot(m)

            else:
                if self.rType == 'x':
                    F[ii, :] = tx

                elif self.rType == 'y':
                    F[ii, :] = ty

                elif self.rType == 'z':
                    F[ii, :] = tz

                else:
                    raise Exception('rType must be: "x", "y" or "z"')

            if not self.silent:
                # Display progress
                count = progress(ii, count, ndata)

        print("Done 100% ...forward operator completed!!\n")

        return F

    def mapPair(self):
        """
            Call for general mapping of the problem
        """
        return self.rhoMap


def get_T_mat(Xn, Yn, Zn, rxLoc):
    """
    Load in the active nodes of a tensor mesh and computes the gravity tensor
    for a given observation location rxLoc[obsx, obsy, obsz]

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
    from scipy.constants import G as NewtG

    NewtG = NewtG*1e+8  # Convertion from mGal (1e-5) and g/cc (1e-3)
    eps = 1e-8  # add a small value to the locations to avoid

    nC = Xn.shape[0]

    # Pre-allocate space for 1D array
    tx = np.zeros((1, nC))
    ty = np.zeros((1, nC))
    tz = np.zeros((1, nC))

    dz = rxLoc[2] - Zn

    dy = Yn - rxLoc[1]

    dx = Xn - rxLoc[0]

    # Compute contribution from each corners
    for aa in range(2):
        for bb in range(2):
            for cc in range(2):

                r = (
                        mkvc(dx[:, aa]) ** 2 +
                        mkvc(dy[:, bb]) ** 2 +
                        mkvc(dz[:, cc]) ** 2
                    ) ** (0.50)

                tx = tx - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                    dy[:, bb] * np.log(dz[:, cc] + r + eps) +
                    dz[:, cc] * np.log(dy[:, bb] + r + eps) -
                    dx[:, aa] * np.arctan(dy[:, bb] * dz[:, cc] /
                                          (dx[:, aa] * r + eps)))

                ty = ty - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                    dx[:, aa] * np.log(dz[:, cc] + r + eps) +
                    dz[:, cc] * np.log(dx[:, aa] + r + eps) -
                    dy[:, bb] * np.arctan(dx[:, aa] * dz[:, cc] /
                                          (dy[:, bb] * r + eps)))

                tz = tz - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                    dx[:, aa] * np.log(dy[:, bb] + r + eps) +
                    dy[:, bb] * np.log(dx[:, aa] + r + eps) -
                    dz[:, cc] * np.arctan(dx[:, aa] * dy[:, bb] /
                                          (dz[:, cc] * r + eps)))

    return tx, ty, tz


def progress(iter, prog, final):
    """
    progress(iter,prog,final)

    Function measuring the progress of a process and print to screen the %.
    Useful to estimate the remaining runtime of a large problem.

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    arg = np.floor(float(iter)/float(final)*10.)

    if arg > prog:

        strg = "Done " + str(arg*10) + " %"
        print(strg)
        prog = arg

    return prog


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
    np.savetxt(filename, data, fmt='%e', delimiter=' ', newline='\n', header=head,comments='')

    print("Observation file saved to: " + filename)


def plot_obs_2D(rxLoc, d=None, title='Gz Obs', vmin=None, vmax=None,
                levels=None, axs=None, marker=True, cmap='plasma'):
    """ Function plot_obs(rxLoc,d,wd)
    Generate a 2d interpolated plot from scatter points of data

    INPUT
    rxLoc       : Observation locations [x,y,z]
    d           : Data vector
    wd          : Uncertainty vector

    OUTPUT
    figure()

    Created on Dec, 27th 2015

    @author: dominiquef

    """

    from scipy.interpolate import griddata
    import pylab as plt

    # Create grid of points
    x = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), 100)
    y = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), 100)

    X, Y = np.meshgrid(x, y)

    # Interpolate
    d_grid = griddata(rxLoc[:, 0:2], d, (X, Y), method='linear')

    # Plot result
    if axs is None:
        fig, axs = plt.figure(), plt.subplot()

    if marker:
        axs.scatter(rxLoc[:, 0], rxLoc[:, 1], c='k', s=10)

    if d is not None:

        if (vmin is None):
            vmin = d.min()

        if (vmax is None):
            vmax = d.max()

        # Create grid of points
        x = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), 100)
        y = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), 100)

        X, Y = np.meshgrid(x, y)

        # Interpolate
        d_grid = griddata(rxLoc[:, 0:2], d, (X, Y), method='linear')
        im = axs.imshow(d_grid, extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im,fraction=0.02, ax=axs)

        if levels is None:
            axs.contour(X, Y, d_grid, 10, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            axs.contour(X, Y, d_grid, levels=levels, colors='k',
                        vmin=vmin, vmax=vmax)

    axs.set_title(title)
    plt.gca().set_aspect('equal', adjustable='box')

    return axs


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
        #rho = self.rhoMap*m
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
            gField: defined on the cell faces [nF x 1]

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
        nFz = self.mesh.nFz

        aveF2CCgx = self.mesh.aveFx2CC * gField[0:nFx]
        aveF2CCgy = self.mesh.aveFy2CC * gField[nFx:(nFx+nFy)]
        aveF2CCgz = self.mesh.aveFz2CC * gField[(nFx+nFy):]

        ggx = 1e+4*self.mesh.cellGrad * aveF2CCgx
        ggy = 1e+4*self.mesh.cellGrad * aveF2CCgy
        ggz = 1e+4*self.mesh.cellGrad * aveF2CCgz

        return {'G': gField, 'ggx': ggx, 'ggy': ggy, 'ggz': ggz, 'u': u}
