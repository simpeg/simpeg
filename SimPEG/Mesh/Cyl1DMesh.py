import numpy as np
import scipy.sparse as sp
from scipy.constants import pi
from SimPEG.Utils import mkvc, ndgrid, sdiag

class Cyl1DMesh(object):
    """
    Cyl1DMesh is a mesh class for cylindrically symmetric 1D problems
    """

    _meshType = 'CYL1D'

    def __init__(self, h, z0=None):
        assert len(h) == 2, "len(h) must equal 2"
        if z0 is not None:
            assert z0.size == 1, "z0.size must equal 1"

        for i, h_i in enumerate(h):
            assert type(h_i) == np.ndarray, ("h[%i] is not a numpy array." % i)
            assert len(h_i.shape) == 1, ("h[%i] must be a 1D numpy array." % i)

        # Ensure h contains 1D vectors
        self._h = [mkvc(x.astype(float)) for x in h]

        if z0 is None:
            z0 = 0
        self._z0 = z0

    ####################################################
    # Mesh properties
    ####################################################

    def h():
        doc = "list containing the width of each cell"
        def fget(self):
            return self._h
        return locals()
    h = property(**h())

    def z0():
        doc = "The z-origin"
        def fget(self):
            return self._z0
        return locals()
    z0 = property(**z0())

    def hr():
        doc = "Width of the cells in the r direction"
        def fget(self):
            return self._h[0]
        return locals()
    hr = property(**hr())

    def hz():
        doc = "Width of the cells in the z direction"
        def fget(self):
            return self._h[1]
        return locals()
    hz = property(**hz())

    ####################################################
    # Counting
    ####################################################

    def nCx():
        doc = "Number of cells in the radial direction"
        fget = lambda self: self.hr.size
        return locals()
    nCx = property(**nCx())

    def nCz():
        doc = "Number of cells in the z direction"
        fget = lambda self: self.hz.size
        return locals()
    nCz = property(**nCz())

    def nC():
        doc = "Total number of cells"
        fget = lambda self: self.nCx * self.nCz
        return locals()
    nC = property(**nC())

    def nCv():
        doc = "Total number of cells in each direction"
        fget = lambda self: np.array([self.nCx, self.nCz])
        return locals()
    nCv = property(**nCv())

    def nNr():
        doc = "Number of nodes in the radial direction"
        fget = lambda self: self.hr.size
        return locals()
    nNr = property(**nNr())

    def nNz():
        doc = "Number of nodes in the radial direction"
        fget = lambda self: self.hz.size + 1
        return locals()
    nNz = property(**nNz())

    def nN():
        doc = "Total number of nodes"
        fget = lambda self: self.nNr * self.nNz
        return locals()
    nN = property(**nN())

    def nFr():
        doc = "Number of r faces"
        fget = lambda self: self.nNr * self.nCz
        return locals()
    nFr = property(**nFr())

    def nFz():
        doc = "Number of z faces"
        fget = lambda self: self.nNz * self.nCx
        return locals()
    nFz = property(**nFz())

    def nFv():
        doc = "Total number of faces in each direction"
        fget = lambda self: np.array([self.nFr, self.nFz])
        return locals()
    nFv = property(**nFv())

    def nF():
        doc = "Total number of faces"
        fget = lambda self: self.nFr + self.nFz
        return locals()
    nF = property(**nF())

    def nE():
        doc = "Number of edges"
        fget = lambda self: self.nN
        return locals()
    nE = property(**nE())

    ####################################################
    # Vectors & Grids
    ####################################################

    def vectorNr():
        doc = "Nodal grid vector (1D) in the r direction"
        fget = lambda self: self.hr.cumsum()
        return locals()
    vectorNr = property(**vectorNr())

    def vectorNz():
        doc = "Nodal grid vector (1D) in the z direction"
        fget = lambda self: np.r_[0, self.hz.cumsum()] + self._z0
        return locals()
    vectorNz = property(**vectorNz())

    def vectorCCr():
        doc = "Cell centered grid vector (1D) in the r direction"
        fget = lambda self: np.r_[0, self.hr.cumsum()[1:] - self.hr[1:]/2]
        return locals()
    vectorCCr = property(**vectorCCr())

    def vectorCCz():
        doc = "Cell centered grid vector (1D) in the z direction"
        fget = lambda self: self.hz.cumsum() - self.hz/2 + self._z0
        return locals()
    vectorCCz = property(**vectorCCz())

    def gridCC():
        doc = "Cell-centered grid"
        def fget(self):
            if self._gridCC is None:
                self._gridCC = ndgrid([self.vectorCCr, self.vectorCCz])
            return self._gridCC
        return locals()
    _gridCC = None
    gridCC = property(**gridCC())

    def gridN():
        doc = "Nodal grid"
        def fget(self):
            if self._gridN is None:
                self._gridN = ndgrid([self.vectorNr, self.vectorNz])
            return self._gridN
        return locals()
    _gridN = None
    gridN = property(**gridN())

    def gridFr():
        doc = "r face grid"
        def fget(self):
            if self._gridFr is None:
                self._gridFr = ndgrid([self.vectorNr, self.vectorCCz])
            return self._gridFr
        return locals()
    _gridFr = None
    gridFr = property(**gridFr())

    def gridFz():
        doc = "z face grid"
        def fget(self):
            if self._gridFz is None:
                self._gridFz = ndgrid([self.vectorCCr, self.vectorNz])
            return self._gridFz
        return locals()
    _gridFz = None
    gridFz = property(**gridFz())

    ####################################################
    # Geometries
    ####################################################

    def edge():
        doc = "Edge lengths"
        def fget(self):
            if self._edge is None:
                self._edge = 2*pi*self.gridN[:,0]
            return self._edge
        return locals()
    _edge = None
    edge = property(**edge())

    def area():
        doc = "Face areas"
        def fget(self):
            if self._area is None:
                areaR = np.kron(self.hz, 2*pi*self.vectorNr)
                areaZ = np.kron(np.ones_like(self.vectorNz),pi*(self.vectorNr**2 - np.r_[0, self.vectorNr[:-1]]**2))
                self._area = np.r_[areaR, areaZ]
            return self._area
        return locals()
    _area = None
    area = property(**area())

    def vol():
        doc = "Volume of each cell"
        def fget(self):
            if self._vol is None:
                az = pi*(self.vectorNr**2 - np.r_[0, self.vectorNr[:-1]]**2)
                self._vol = np.kron(self.hz,az)
            return self._vol
        return locals()
    _vol = None
    vol = property(**vol())

    ####################################################
    # Operators
    ####################################################

    def edgeCurl():
        doc = "The edgeCurl property."
        def fget(self):
            if self._edgeCurl is None:
                #1D Difference matricies
                dr = sp.spdiags((np.ones((self.nCx+1, 1))*[-1, 1]).T, [-1,0], self.nCx, self.nCx, format="csr")
                dz = sp.spdiags((np.ones((self.nCz+1, 1))*[-1, 1]).T, [0,1], self.nCz, self.nCz+1, format="csr")

                #2D Difference matricies
                Dr = sp.kron(sp.eye(self.nNz), dr)
                Dz = -sp.kron(dz, sp.eye(self.nCx))  #Not sure about this negative

                #Edge curl operator
                self._edgeCurl = sp.diags(1/self.area,0)*sp.vstack((Dz, Dr))*sp.diags(self.edge,0)
            return self._edgeCurl
        return locals()
    _edgeCurl = None
    edgeCurl = property(**edgeCurl())

    def aveE2CC():
        doc = "Averaging operator from cell edges to cell centres"
        def fget(self):
            if self._aveE2CC is None:
                az = sp.spdiags(0.5*np.ones((2, self.nNz)), [-1,0], self.nNz, self.nCz, format='csr')
                ar = sp.spdiags(0.5*np.ones((2, self.nCx)), [0, 1], self.nCx, self.nCx, format='csr')
                ar[0,0] = 1
                self._aveE2CC = sp.kron(az, ar).T
            return self._aveE2CC
        return locals()
    _aveE2CC = None
    aveE2CC = property(**aveE2CC())

    def aveF2CC():
        doc = "Averaging operator from cell faces to cell centres"
        def fget(self):
            if self._aveF2CC is None:
                az = sp.spdiags(0.5*np.ones((2, self.nNz)), [-1,0], self.nNz, self.nCz, format='csr')
                ar = sp.spdiags(0.5*np.ones((2, self.nCx)), [0, 1], self.nCx, self.nCx, format='csr')
                ar[0,0] = 1
                Afr = sp.kron(sp.eye(self.nCz),ar)
                Afz = sp.kron(az,sp.eye(self.nCx))
                self._aveF2CC = sp.vstack((Afr,Afz)).T
            return self._aveF2CC
        return locals()
    _aveF2CC = None
    aveF2CC = property(**aveF2CC())

    ####################################################
    # Methods
    ####################################################


    def getMass(self, materialProp=None, loc='e'):
        """ Produces mass matricies.

        :param None,float,numpy.ndarray materialProp: property to be averaged (see below)
        :param str loc: Average to location: 'e'-edges, 'f'-faces
        :rtype: scipy.sparse.csr.csr_matrix
        :return: M, the mass matrix

        materialProp can be::

            None            -> takes materialProp = 1 (default)
            float           -> a constant value for entire domain
            numpy.ndarray   -> if materialProp.size == self.nC
                                    3D property model
                               if materialProp.size = self.nCz
                                    1D (layered eath) property model
        """
        if materialProp is None:
            materialProp = np.ones(self.nC)
        elif type(materialProp) is float:
            materialProp = np.ones(self.nC)*materialProp
        elif materialProp.shape == (self.nCz,):
            materialProp = materialProp.repeat(self.nCx)
        materialProp = mkvc(materialProp)
        assert materialProp.shape == (self.nC,), "materialProp incorrect shape"

        if loc=='e':
            Av = self.aveE2CC
        elif loc=='f':
            Av = self.aveF2CC
        else:
            raise ValueError('Invalid loc')

        diag = Av.T * (self.vol * mkvc(materialProp))

        return sdiag(diag)

    def getEdgeMass(self, materialProp=None):
        """mass matrix for products of edge functions w'*M(materialProp)*e"""
        return self.getMass(loc='e', materialProp=materialProp)

    def getFaceMass(self, materialProp=None):
        """mass matrix for products of face functions w'*M(materialProp)*f"""
        return self.getMass(loc='f', materialProp=materialProp)

    def getInterpolationMat(self, loc, locType='fz'):
        """ Produces intrpolation matrix

        :param numpy.ndarray loc: Location of points to interpolate to
        :param str locType: What to interpolate (see below)
        :rtype: scipy.sparse.csr.csr_matrix
        :return: M, the intrpolation matrix

        locType can be::

            'fz'    -> z-component of field defined on faces
            'fr'    -> r-component of field defined on faces
            'et'    -> theta-component of field defined on edges
        """

        loc = np.atleast_2d(loc)

        assert np.all(loc[:,0]<=self.vectorNr.max()) & \
               np.all(loc[:,1]>=self.vectorNz.min()) & \
               np.all(loc[:,1]<=self.vectorNz.max()), \
               "Points outside of mesh"


        if locType=='fz':
            Q = sp.lil_matrix((loc.shape[0], self.nF), dtype=float)

            for i, iloc in enumerate(loc):
                # Point is on a z-interface
                if np.any(np.abs(self.vectorNz-iloc[1])<0.001):
                    dFz = self.gridFz-iloc          #Distance to z faces
                    dFz[dFz[:,0]>0,:] = np.inf      #Looking for next face to the left...
                    indL = np.argmin(np.sum(dFz**2, axis=1))  #Closest one
                    if self.gridFz[indL,0] == self.vectorCCr.max(): #Point in outer half cell (linear extrapolation)
                        zFL = self.gridFz[indL,:]
                        zFLL = self.gridFz[indL-1,:]
                        Q[i, indL+self.nFr] = (iloc[0] - zFLL[0])/(zFL[0] - zFLL[0])
                        Q[i, indL+self.nFr-1] = -(iloc[0] - zFL[0])/(zFL[0] - zFLL[0])
                    else:
                        zFL = self.gridFz[indL,:]
                        zFR = self.gridFz[indL+1,:]
                        Q[i,indL+self.nFr] = (zFR[0] - iloc[0])/(zFR[0] - zFL[0])
                        Q[i,indL+self.nFr+1] = (iloc[0] - zFL[0])/(zFR[0] - zFL[0])
                # Point is in a cell
                else:
                    dFz = self.gridFz-iloc
                    dFz[dFz>0] = np.inf
                    dFz = np.sum(dFz**2, axis=1)

                    indBL = np.argmin(dFz)         # Face below and to the left
                    indAL = indBL + self.nCx        # Face above and to the left

                    zF_BL = self.gridFz[indBL,:]
                    zF_AL = self.gridFz[indAL,:]

                    dzB = iloc[1] - zF_BL[1]         # z-distance to face below
                    dzA = zF_AL[1] - iloc[1]         # z-distance to face above

                    if self.gridFz[indBL,0] == self.vectorCCr.max(): #Point in outer half cell (linear extrapolation)
                        zF_BLL = self.gridFz[indBL-1,:]
                        zF_ALL = self.gridFz[indAL-1,:]

                        DZ = zF_AL[1] - zF_BL[1]
                        DR = zF_AL[0] - zF_ALL[0]

                        drL = iloc[0] - zF_AL[0]
                        drLL = iloc[0] - zF_ALL[0]

                        Q[i, indBL+self.nFr-1] = -(1 - dzB/DZ)*(drL/DR)
                        Q[i, indBL+self.nFr] = (1 - dzB/DZ)*(drLL/DR)
                        Q[i, indAL+self.nFr-1] = -(dzB/DZ)*(drL/DR)
                        Q[i, indAL+self.nFr] = (dzB/DZ)*(drLL/DR)
                    else:
                        indBR = indBL+1                 # Face below and to the right
                        indAR = indAL + 1               # Face above and to the right
                        zF_BR = self.gridFz[indBR,:]

                        drL = iloc[0] - zF_BL[0]         # r-distance to face on left
                        drR = zF_BR[0] - iloc[0]         # r-distance to face on right

                        drz = (drL + drR)*(dzB + dzA)
                        Q[i,indBL+self.nFr] = drR*dzA/drz
                        Q[i,indBR+self.nFr] = drL*dzA/drz
                        Q[i,indAL+self.nFr] = drR*dzB/drz
                        Q[i,indAR+self.nFr] = drL*dzB/drz

        elif locType=='fr':
            raise NotImplementedError('locType==fr')
        elif locType=='et':
            raise NotImplementedError('locType==et')
        else:
            raise ValueError('Invalid locType')
        return Q.tocsr()

    def getNearest(self, loc, locType):
        """ Returns the index of the closest face or edge to a given location

        :param numpy.ndarray loc: Test point
        :param str locType: Type of location desired (see below)
        :rtype: int
        :return: ind:

        locType can be::

            'fz'    -> location of nearest z-face
            'fr'    -> location of nearest r-face
            'et'    -> location of nearest edge
        """

        if locType=='et':
            dr = self.gridN[:,0] - loc[0]
            dz = self.gridN[:,1] - loc[1]
        elif locType=='fz':
            dr = self.gridFz[:,0] - loc[0]
            dz = self.gridFz[:,1] - loc[1]
        elif locType=='fr':
            dr = self.gridFr[:,0] - loc[0]
            dz = self.gridFr[:,1] - loc[1]
        else:
            raise ValueError('Invalid locType')
        R = np.sqrt(dr**2 + dz**2)
        ind = np.argmin(R)
        return ind
