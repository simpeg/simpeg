from __future__ import print_function
from SimPEG import Problem
from SimPEG import Utils
from SimPEG import Props
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
    rtype = 'z'

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

    def fwr_op(self):
        # Add forward function
        # kappa = self.model.kappa TODO
        rho = self.rhoMap*self.model

        if self.forwardOnly:

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
            ndata = rxLoc.shape[0]

            # Pre-allocate space and create magnetization matrix if required
            # Pre-allocate space
            if self.rtype == 'z':

                fwr_d = np.zeros(self.survey.nRx)

            elif self.rtype == 'xyz':

                fwr_d = np.zeros(3*self.survey.nRx)

            else:

                print("""Flag must be either 'z' | 'xyz', please revised""")
                return

            # Add counter to dsiplay progress. Good for large problems
            count = -1
            for ii in range(ndata):

                tx, ty, tz = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])

                if self.rtype == 'z':
                    fwr_d[ii] = tz.dot(rho)

                elif self.rtype == 'xyz':
                    fwr_d[ii] = tx.dot(rho)
                    fwr_d[ii+ndata] = ty.dot(rho)
                    fwr_d[ii+2*ndata] = tz.dot(rho)

            # Display progress
                count = progress(ii, count, ndata)

            print("Done 100% ...forward operator completed!!\n")

            return fwr_d

        else:
            return self.G.dot(rho)

    def fields(self, m):
        self.model = m

        fields = self.fwr_op()

        return fields

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
            self._G = self.Intrgl_Fwr_Op('z')

        return self._G

    def Intrgl_Fwr_Op(self, flag):

        """

        Gravity forward operator in integral form

        flag        = 'z' | 'xyz'

        Return
        _G        = Linear forward modeling operation

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
        ndata = rxLoc.shape[0]

        # Pre-allocate space and create magnetization matrix if required
        # Pre-allocate space
        if flag == 'z':

            G = np.zeros((ndata, nC))

        elif flag == 'xyz':

            G = np.zeros((int(3*ndata), nC))

        else:

            print("""Flag must be either 'z' | 'xyz', please revised""")
            return

        # Loop through all observations
        print("Begin calculation of forward operator: " + flag)

        # Add counter to dsiplay progress. Good for large problems
        count = -1
        for ii in range(ndata):

            tx, ty, tz = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])

            if flag == 'z':

                G[ii, :] = tz

            elif flag == 'xyz':
                G[ii, :] = tx
                G[ii+ndata, :] = ty
                G[ii+2*ndata, :] = tz

            # Display progress
            count = progress(ii, count, ndata)

        print("Done 100% ...forward operator completed!!\n")

        return G


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
    eps = 1e-10  # add a small value to the locations to avoid

    nC = Xn.shape[0]

    # Pre-allocate space for 1D array
    tx = np.zeros((1, nC))
    ty = np.zeros((1, nC))
    tz = np.zeros((1, nC))

    dz = rxLoc[2] - Zn + eps

    dy = Yn - rxLoc[1] + eps

    dx = Xn - rxLoc[0] + eps

    # Compute contribution from each corners
    for aa in range(2):
        for bb in range(2):
            for cc in range(2):

                r = (
                        dx[:, aa] ** 2 +
                        dy[:, bb] ** 2 +
                        dz[:, cc] ** 2
                    ) ** (0.50)

                tx = tx - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                    dy[:, bb] * np.log(dz[:, cc] + r) +
                    dz[:, cc] * np.log(dy[:, bb] + r) -
                    dx[:, aa] * np.arctan(dy[:, bb] * dz[:, cc] /
                                          (dx[:, aa] * r)))

                ty = ty - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                    dx[:, aa] * np.log(dz[:, cc] + r) +
                    dz[:, cc] * np.log(dx[:, aa] + r) -
                    dy[:, bb] * np.arctan(dx[:, aa] * dz[:, cc] /
                                          (dy[:, bb] * r)))

                tz = tz - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                    dx[:, aa] * np.log(dy[:, bb] + r) +
                    dy[:, bb] * np.log(dx[:, aa] + r) -
                    dz[:, cc] * np.arctan(dx[:, aa] * dy[:, bb] /
                                          (dz[:, cc] * r)))

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


def writeUBCobs(filename, survey, d):
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

    data = np.c_[rxLoc, d, wd]

    head = '%i\n'%len(d)
    np.savetxt(filename, data, fmt='%e', delimiter=' ', newline='\n', header=head,comments='')

    print("Observation file saved to: " + filename)


def plot_obs_2D(rxLoc, d=None, varstr='Gz Obs', vmin=None, vmax=None,
                levels=None, fig=None):
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
    if fig is None:
        fig = plt.figure()

    plt.scatter(rxLoc[:, 0], rxLoc[:, 1], c='k', s=10)

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
        plt.imshow(d_grid, extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', vmin=vmin, vmax=vmax, cmap="plasma")
        plt.colorbar(fraction=0.02)

        if levels is None:
            plt.contour(X, Y, d_grid, 10, vmin=vmin, vmax=vmax, cmap="plasma")
        else:
            plt.contour(X, Y, d_grid, levels=levels, colors='r',
                        vmin=vmin, vmax=vmax, cmap="plasma")

    plt.title(varstr)
    plt.gca().set_aspect('equal', adjustable='box')

    return fig


def readUBCgravObs(obs_file):

    """
    Read UBC grav file format

    INPUT:
    :param fileName, path to the UBC obs grav file

    OUTPUT:
    :param survey

    """

    fid = open(obs_file, 'r')

    # First line has the number of rows
    line = fid.readline()
    ndat = np.array(line.split(), dtype=int)

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()
    temp = np.array(line.split(), dtype=float)

    d = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    for ii in range(ndat):

        temp = np.array(line.split(), dtype=float)
        locXYZ[ii, :] = temp[:3]
        d[ii] = temp[3]
        wd[ii] = temp[4]
        line = fid.readline()

    rxLoc = GRAV.RxObs(locXYZ)
    srcField = GRAV.SrcField([rxLoc])
    survey = GRAV.LinearSurvey(srcField)
    survey.dobs = d
    survey.std = wd
    return survey


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
        rho = self.rhoMap*m
        self._Mfi = self.mesh.getFaceInnerProduct()
        self._MfI = Utils.sdiag(1./self._Mfi.diagonal())

    def getRHS(self, m):
        """


        """

        Mc = Utils.sdiag(self.mesh.vol)

        rho = self.rhoMap*m

        return Mc*rho

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Gravity nodal problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return -self._Div.T*self.Mfi*self._Div

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
            m1 = sp.linalg.interface.aslinearoperator(Utils.sdiag(1/A.diagonal()))
            u, info = sp.linalg.bicgstab(A, RHS, tol=1e-6, maxiter=1000, M=m1)

        else:
            print("Solving with Paradiso")
            Ainv = self.solver(A)
            u = Ainv*RHS

        gField = 4.*np.pi*NewtG*1e+8*self._Div*u

        return {'G': gField, 'u': u}
