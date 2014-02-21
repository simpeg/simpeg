import numpy as np
import scipy.sparse as sp
from scipy.constants import pi
from SimPEG.Utils import mkvc, ndgrid, sdiag, kron3, speye, ddx, av, avExtrap
from TensorMesh import BaseTensorMesh

class CylMesh(BaseTensorMesh):
    """
        CylMesh is a mesh class for cylindrical problems
    """

    _meshType = 'CYL'

    _unitDimensions = [1, 2*np.pi, 1]

    def __init__(self, h, x0=None):
        BaseTensorMesh.__init__(self, h, x0)
        assert self.dim == 3, "dim of mesh must equal 3, for a cylindrically symmetric mesh use [hx, 1, hz]"
        assert self.hy.sum() == 2*np.pi, "The 2nd dimension must sum to 2*pi"

    @property
    def nNx(self):
        """
        Number of nodes in the x-direction

        :rtype: int
        :return: nNx
        """
        if self.nCy == 1:
            return self.nCx
        return self.nCx + 1

    @property
    def nNy(self):
        """
        Number of nodes in the y-direction

        :rtype: int
        :return: nNy
        """
        if self.nCy == 1:
            return self.nCy - 1
        return self.nCy

    @property
    def nN(self):
        """
        Total number of nodes

        :rtype: int
        :return: nN
        """
        return (np.r_[self.nNx, self.nNy, self.nNz]).prod()

    @property
    def nFx(self):
        """
        Number of x-faces

        :rtype: int
        :return: nFx
        """
        return self.nC

    @property
    def vnFx(self):
        """
        Number of x-faces in each direction

        :rtype: numpy.array (dim, )
        :return: vnFx
        """
        return self.vnC

    @property
    def nFy(self):
        """
        Number of y-faces

        :rtype: int
        :return: nFy
        """
        return (self.vnC + np.r_[0,-1,0][:self.dim]).prod()

    @property
    def nEx(self):
        """
        Number of x-edges

        :rtype: int
        :return: nEx
        """
        return (self._n + np.r_[0,-1,1]).prod()

    @property
    def nEy(self):
        """
        Number of y-edges

        :rtype: int
        :return: nEy
        """
        return (self._n + np.r_[0,0,1]).prod()

    @property
    def nEz(self):
        """
        Number of z-edges

        :rtype: int
        :return: nEz
        """
        return (self._n + np.r_[0,-1,0]).prod()

    @property
    def vectorCCx(self):
        """Cell-centered grid vector (1D) in the x direction."""
        if self.nCy == 1:
            return np.r_[-self.hx[0]*0.5, self.hx[:-1].cumsum()] + self.hx*0.5 # - self.hx[0]/2
        return np.r_[0, self.hx[:-1].cumsum()] + self.hx*0.5

    @property
    def vectorCCy(self):
        """Cell-centered grid vector (1D) in the y direction."""
        return np.r_[0, self.hy[:-1]]

    @property
    def vectorNx(self):
        """Nodal grid vector (1D) in the x direction."""
        if self.nCy == 1:
            return self.hx.cumsum()# - self.hx[0]/2
        return np.r_[0, self.hx].cumsum()

    @property
    def vectorNy(self):
        """Nodal grid vector (1D) in the y direction."""
        if self.nCy == 1:
            return np.r_[0]
        return np.r_[0, self.hy[:-1].cumsum()] + self.hy[0]*0.5


    def getTensor(self, locType):
        """ Returns a tensor list.

        :param str locType: What tensor (see below)
        :rtype: list
        :return: list of the tensors that make up the mesh.

        locType can be::

            'Ex'    -> x-component of field defined on edges
            'Ey'    -> y-component of field defined on edges
            'Ez'    -> z-component of field defined on edges
            'Fx'    -> x-component of field defined on faces
            'Fy'    -> y-component of field defined on faces
            'Fz'    -> z-component of field defined on faces
            'N'     -> scalar field defined on nodes
            'CC'    -> scalar field defined on cell centers
        """

        if   locType is 'Fx':
            ten = [self.vectorNx , self.vectorCCy, self.vectorCCz]
        elif locType is 'Fy':
            ten = [self.vectorCCx, self.vectorNy , self.vectorCCz]
        elif locType is 'Fz':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorNz ]
        elif locType is 'Ex':
            ten = [self.vectorCCx, self.vectorNy , self.vectorNz ]
        elif locType is 'Ey':
            ten = [self.vectorNx , self.vectorCCy, self.vectorNz ]
        elif locType is 'Ez':
            ten = [self.vectorNx , self.vectorNy , self.vectorCCz]
        elif locType is 'CC':
            ten = [self.vectorCCx, self.vectorCCy, self.vectorCCz]
        elif locType is 'N':
            ten = [self.vectorNx , self.vectorNy , self.vectorNz ]

        return [t for t in ten if t is not None]

    @property
    def edge(self):
        """Edge lengths"""
        if getattr(self, '_edge', None) is None:
            if self.nCy == 1:
                self._edge = 2*pi*self.gridN[:,0]
            else:
                raise NotImplementedError('edges not yet implemented for 3D cyl mesh')
        return self._edge

    @property
    def area(self):
        """Face areas"""
        if getattr(self, '_area', None) is None:
            if self.nCy > 1:
                raise NotImplementedError('area not yet implemented for 3D cyl mesh')
            areaR = np.kron(self.hz, 2*pi*self.vectorNx)
            areaZ = np.kron(np.ones_like(self.vectorNz),pi*(self.vectorNx**2 - np.r_[0, self.vectorNx[:-1]]**2))
            self._area = np.r_[areaR, areaZ]
        return self._area

    @property
    def vol(self):
        """Volume of each cell"""
        if getattr(self, '_vol', None) is None:
            if self.nCy > 1:
                raise NotImplementedError('vol not yet implemented for 3D cyl mesh')
            az = pi*(self.vectorNx**2 - np.r_[0, self.vectorNx[:-1]]**2)
            self._vol = np.kron(self.hz,az)
        return self._vol

    ####################################################
    # Operators
    ####################################################

    @property
    def faceDiv(self):
        """Construct divergence operator (face-stg to cell-centres)."""
        if getattr(self, '_faceDiv', None) is None:
            n = self.vnC
            # Compute faceDivergence operator on faces
            D1 = self.faceDivx
            D3 = self.faceDivz
            if self.nCy == 1:
                D = sp.hstack((D1, D3), format="csr")
            elif self.nCy > 1:
                D2 = self.faceDivy
                D = sp.hstack((D1, D2, D3), format="csr")
            self._faceDiv = D
        return self._faceDiv

    @property
    def faceDivx(self):
        """Construct divergence operator in the x component (face-stg to cell-centres)."""
        if getattr(self, '_faceDivx', None) is None:
            D1 = kron3(speye(self.nCz), speye(self.nCy), ddx(self.nCx)[:,1:])
            S = self.r(self.area, 'F', 'Fx', 'V')
            V = self.vol
            self._faceDivx = sdiag(1/V)*D1*sdiag(S)
        return self._faceDivx

    @property
    def faceDivy(self):
        """Construct divergence operator in the y component (face-stg to cell-centres)."""
        raise NotImplementedError('Wrapping the ddx is not yet implemented.')
        if getattr(self, '_faceDivy', None) is None:
            # TODO: this needs to wrap to join up faces which are connected in the cylinder
            D2 = kron3(speye(self.nCz), ddx(self.nCy), speye(self.nCx))
            S = self.r(self.area, 'F', 'Fy', 'V')
            V = self.vol
            self._faceDivy = sdiag(1/V)*D2*sdiag(S)
        return self._faceDivy

    @property
    def faceDivz(self):
        """Construct divergence operator in the z component (face-stg to cell-centres)."""
        if getattr(self, '_faceDivz', None) is None:
            D3 = kron3(ddx(self.nCz), speye(self.nCy), speye(self.nCx))
            S = self.r(self.area, 'F', 'Fz', 'V')
            V = self.vol
            self._faceDivz = sdiag(1/V)*D3*sdiag(S)
        return self._faceDivz


    @property
    def cellGrad(self):
        """The cell centered Gradient, takes you to cell faces."""
        raise NotImplementedError('Cell Grad is not yet implemented.')


    @property
    def nodalGrad(self):
        """Construct gradient operator (nodes to edges)."""
        raise NotImplementedError('nodalGrad not yet implemented')

    @property
    def nodalLaplacian(self):
        """Construct laplacian operator (nodes to edges)."""
        raise NotImplementedError('nodalLaplacian not yet implemented')

    @property
    def edgeCurl(self):
        """The edgeCurl property."""
        if getattr(self, '_edgeCurl', None) is None:
            #1D Difference matricies
            dr = sp.spdiags((np.ones((self.nCx+1, 1))*[-1, 1]).T, [-1,0], self.nCx, self.nCx, format="csr")
            dz = sp.spdiags((np.ones((self.nCz+1, 1))*[-1, 1]).T, [0,1], self.nCz, self.nCz+1, format="csr")

            #2D Difference matricies
            Dr = sp.kron(sp.eye(self.nNz), dr)
            Dz = -sp.kron(dz, sp.eye(self.nCx))  #Not sure about this negative

            #Edge curl operator
            self._edgeCurl = sp.diags(1/self.area,0)*sp.vstack((Dz, Dr))*sp.diags(self.edge,0)
        return self._edgeCurl

    @property
    def aveE2CC(self):
        """Averaging operator from cell edges to cell centres"""
        if getattr(self, '_aveE2CC', None) is None:
            az = sp.spdiags(0.5*np.ones((2, self.nNz)), [-1,0], self.nNz, self.nCz, format='csr')
            ar = sp.spdiags(0.5*np.ones((2, self.nCx)), [0, 1], self.nCx, self.nCx, format='csr')
            ar[0,0] = 1
            self._aveE2CC = sp.kron(az, ar).T
        return self._aveE2CC

    @property
    def aveF2CC(self):
        """Averaging operator from cell faces to cell centres"""
        if getattr(self, '_aveF2CC', None) is None:
            az = sp.spdiags(0.5*np.ones((2, self.nNz)), [-1,0], self.nNz, self.nCz, format='csr')
            ar = sp.spdiags(0.5*np.ones((2, self.nCx)), [0, 1], self.nCx, self.nCx, format='csr')
            ar[0,0] = 1
            Afr = sp.kron(sp.eye(self.nCz),ar)
            Afz = sp.kron(az,sp.eye(self.nCx))
            self._aveF2CC = sp.vstack((Afr,Afz)).T
        return self._aveF2CC

    def getFaceMassDeriv(self):
        Av = self.aveF2CC
        return Av.T * sdiag(self.vol)

    def getEdgeMassDeriv(self):
        Av = self.aveE2CC
        return Av.T * sdiag(self.vol)


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

        assert np.all(loc[:,0]<=self.vectorNx.max()) & \
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
