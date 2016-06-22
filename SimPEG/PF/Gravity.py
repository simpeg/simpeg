from SimPEG import *
import BaseGrav as GRAV
import re


class GravityIntegral(Problem.BaseProblem):

    # surveyPair = Survey.LinearSurvey

    storeG = True #: Store the forward matrix by default, otherwise just compute d
    actInd = None #: Active cell indices provided

    def __init__(self, mesh, mapping=None, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    def fwr_op(self):
        # Add forward function
        # kappa = self.curModel.kappa TODO
        sus = self.mapping*self.curModel
        return self.G.dot(sus)

    def fields(self, m):
        self.curModel = m
        total = np.zeros(self.survey.nRx)
        induced = self.fwr_op()
        # rem = self.rem

        if induced is not None:
            total += induced

        return total

        # return self.G.dot(self.mapping*(m))

    def Jvec(self, m, v, f=None):
        dmudm = self.mapping.deriv(m)
        return self.G.dot(dmudm*v)

    def Jtvec(self, m, v, f=None):
        dmudm = self.mapping.deriv(m)
        return dmudm.T * (self.G.T.dot(v))

    @property
    def G(self):
        if not self.ispaired:
            raise Exception('Need to pair!')

        if getattr(self, '_G', None) is None:
            self._G = self.Intrgl_Fwr_Op( 'z' )

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

            if self.actInd.dtype=='bool':
                inds = np.asarray([inds for inds, elem in enumerate(self.actInd, 1) if elem], dtype = int) - 1
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

        # Create vectors of nodal location (lower and upper corners for each cell)
        xn = self.mesh.vectorNx
        yn = self.mesh.vectorNy
        zn = self.mesh.vectorNz

        yn2, xn2, zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
        yn1, xn1, zn1 = np.meshgrid(yn[0:-1], xn[0:-1], zn[0:-1])

        Yn = P.T*np.c_[mkvc(yn1), mkvc(yn2)]
        Xn = P.T*np.c_[mkvc(xn1), mkvc(xn2)]
        Zn = P.T*np.c_[mkvc(zn1), mkvc(zn2)]

        rxLoc = self.survey.srcField.rxList[0].locs
        ndata = rxLoc.shape[0]

        # Pre-allocate space and create magnetization matrix if required
        # Pre-allocate space
        if flag == 'z':

            G = np.zeros((ndata, nC))

        elif flag == 'xyz':

            G = np.zeros((int(3*ndata), nC))

        else:

            print """Flag must be either 'z' | 'xyz', please revised"""
            return


        # Loop through all observations and create forward operator (ndata-by-nC)
        print "Begin calculation of forward operator: " + flag

        # Add counter to dsiplay progress. Good for large problems
        count = -1;
        for ii in range(ndata):

            if flag=='z':
                tt = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])
                G[ii, :] = tt

            elif flag == 'xyz':
                print "Sorry 3-component not implemented yet"

            # Display progress
            count = progress(ii, count, ndata)

        print "Done 100% ...forward operator completed!!\n"

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
    NewtG=6.6738e-3
    eps = 1e-10 # add a small value to the locations to avoid /0

    nC = Xn.shape[0]

    # Pre-allocate space for 1D array
    T = np.zeros((1,nC))

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

                T = T - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                    dx[:, aa] * np.log(dy[:, bb] + r) +
                    dy[:, bb] * np.log(dx[:, aa] + r) -
                    dz[:, cc] * np.arctan(
                        dx[:, aa] * dy[:, bb] / (dz[:, cc] * r)
                    )
                )

    return T


def progress(iter, prog, final):
    """
    progress(iter,prog,final)

    Function measuring the progress of a process and print to screen the %.
    Useful to estimate the remaining runtime of a large problem.

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    arg = np.floor(float(iter)/float(final)*10.)

    if  arg > prog:

        strg = "Done " + str(arg*10) + " %"
        print strg
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

    with file(filename, 'w') as fid:
        fid.write('%i\n' % len(d))
        np.savetxt(fid, data, fmt='%e', delimiter=' ', newline='\n')

    print "Observation file saved to: " + filename


def getActiveTopo(mesh, topo, flag):
    """
    getActiveTopo(mesh,topo)

    Function creates an active cell model from topography

    INPUT
    mesh        : Mesh in SimPEG format
    topo        : Scatter points defining topography [x,y,z]

    OUTPUT
    actv        : Active cell model

    """
    import scipy.interpolate as interpolation

    if flag == 'N':
        Zn = np.zeros((mesh.nNx, mesh.nNy))
        # wght    = np.zeros((mesh.nNx,mesh.nNy))
        cx = mesh.vectorNx
        cy = mesh.vectorNy

    F = interpolation.NearestNDInterpolator(topo[:, 0:2], topo[:, 2])
    [Y, X] = np.meshgrid(cy, cx)

    Zn = F(X, Y)

    actv = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))

    if flag == 'N':
        Nz = mesh.vectorNz[1:]

    for jj in range(mesh.nCy):

        for ii in range(mesh.nCx):

            temp = [kk for kk in range(len(Nz)) if np.all(Zn[ii:(ii+2), jj:(jj+2)] > Nz[kk]) ]
            actv[ii, jj, temp] = 1

    actv = mkvc(actv == 1)

    inds = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem], dtype = int) - 1

    return inds

def plot_obs_2D(survey,varstr):
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

    rxLoc   = survey.srcField.rxList[0].locs
    d       = survey.dobs
    wd      = survey.std

    # Create grid of points
    x = np.linspace(rxLoc[:,0].min(), rxLoc[:,0].max(), 100)
    y = np.linspace(rxLoc[:,1].min(), rxLoc[:,1].max(), 100)

    X, Y = np.meshgrid(x,y)

    # Interpolate
    d_grid = griddata(rxLoc[:,0:2],d,(X,Y), method ='linear')

    # Plot result
    plt.figure()
    plt.subplot()
    plt.imshow(d_grid, extent=[x.min(), x.max(), y.min(), y.max()],origin = 'lower')
    plt.colorbar(fraction=0.02)
    plt.contour(X,Y, d_grid,10)
    plt.scatter(rxLoc[:,0],rxLoc[:,1], c=d, s=20)
    plt.title(varstr)
    plt.gca().set_aspect('equal', adjustable='box')

def readUBCgravObs(obs_file):

    """
    Read UBC grav file format

    INPUT:
    :param fileName, path to the UBC obs grav file

    OUTPUT:
    :param survey

    """

    fid = open(obs_file,'r')

    # First line has the number of rows
    line = fid.readline()
    ndat = np.array(line.split(),dtype=int)

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()
    temp = np.array(line.split(),dtype=float)

    d  = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros( (ndat,3), dtype=float)

    for ii in range(ndat):

        temp = np.array(line.split(),dtype=float)
        locXYZ[ii,:] = temp[:3]
        d[ii] = temp[3]
        wd[ii] = temp[4]
        line = fid.readline()

    rxLoc = GRAV.RxObs(locXYZ)
    srcField = GRAV.SrcField([rxLoc])
    survey = GRAV.LinearSurvey(srcField)
    survey.dobs =  d
    survey.std =  wd
    return survey


def read_GRAVinv_inp(input_file):
    """Read input files for forward modeling MAG data with integral form
    INPUT:
    input_file: File name containing the forward parameter

    OUTPUT:
    mshfile
    obsfile
    topofile
    start model
    ref model
    weightfile
    chi_target
    as, ax ,ay, az
    upper, lower bounds
    lp, lqx, lqy, lqz

    # All files should be in the working directory, otherwise the path must
    # be specified.

    Created on Dec 21th, 2015

    @author: dominiquef
    """


    fid = open(input_file,'r')

    # Line 1
    line = fid.readline()
    l_input  = line.split('!')
    mshfile = l_input[0].rstrip()

    # Line 2
    line = fid.readline()
    l_input  = line.split('!')
    obsfile = l_input[0].rstrip()

    # Line 3
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    if l_input=='null':
        topofile = []

    else:
        topofile = l_input[0].rstrip()


    # Line 4
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    if l_input[0]=='VALUE':
        mstart = float(l_input[1])

    else:
        mstart = l_input[0].rstrip()

    # Line 5
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    if l_input[0]=='VALUE':
        mref = float(l_input[1])

    else:
        mref = l_input[0].rstrip()


    # Line 7
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    if l_input[0]=='DEFAULT':
        wgtfile = None

    else:
        wgtfile = l_input[0].rstrip()

    # Line 8
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    chi = float(l_input[0])

    # Line 9
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    val = np.array(l_input[0:4])
    alphas = val.astype(np.float)

    # Line 10
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    if l_input[0]=='VALUE':
        val   = np.array(l_input[1:3])
        bounds = val.astype(np.float)

    else:
        bounds = l_input[0].rstrip()

    # Line 11
    line = fid.readline()
    l_input = re.split('[!\s]',line)
    if l_input[0]=='VALUE':
        val   = np.array(l_input[1:6])
        lpnorms = val.astype(np.float)

    else:
        lpnorms = l_input[0].rstrip()

    return mshfile, obsfile, topofile, mstart, mref, wgtfile, chi, alphas, bounds, lpnorms

