import numpy as np
import scipy.sparse as sp
from scipy.constants import pi
from SimPEG.Utils import mkvc, ndgrid, sdiag, kron3, speye, ddx, av, avExtrap
from TensorMesh import BaseTensorMesh
from InnerProducts import InnerProducts


class CylMesh(BaseTensorMesh, InnerProducts):
    """
        CylMesh is a mesh class for cylindrical problems

        ::

            cs, nc, npad = 20., 30, 8
            hx = Utils.meshTensor([(cs,npad+10,-0.7), (cs,nc), (cs,npad,1.3)])
            hz = Utils.meshTensor([(cs,npad   ,-1.3), (cs,nc), (cs,npad,1.3)])
            mesh = Mesh.CylMesh([hx,1,hz], [0.,0,-hz.sum()/2.])
    """

    _meshType = 'CYL'

    _unitDimensions = [1, 2*np.pi, 1]

    def __init__(self, h, x0=None):
        BaseTensorMesh.__init__(self, h, x0)
        assert self.dim == 3, "dim of mesh must equal 3, for a cylindrically symmetric mesh use [hx, 1, hz]"
        assert self.hy.sum() == 2*np.pi, "The 2nd dimension must sum to 2*pi"

    @property
    def isSymmetric(self):
        return self.nCy == 1

    @property
    def nNx(self):
        """
        Number of nodes in the x-direction

        :rtype: int
        :return: nNx
        """
        if self.isSymmetric: return self.nCx
        return self.nCx + 1

    @property
    def nNy(self):
        """
        Number of nodes in the y-direction

        :rtype: int
        :return: nNy
        """
        if self.isSymmetric: return 0
        return self.nCy

    @property
    def vnFx(self):
        """
        Number of x-faces in each direction

        :rtype: numpy.array (dim, )
        :return: vnFx
        """
        return self.vnC

    @property
    def vnEy(self):
        """
        Number of y-edges in each direction

        :rtype: numpy.array (dim, )
        :return: vnEy or None if dim < 2
        """
        nNx = self.nNx if self.isSymmetric else self.nNx - 1
        return np.r_[nNx, self.nCy, self.nNz]

    @property
    def vnEz(self):
        """
        Number of z-edges in each direction

        :rtype: numpy.array (dim, )
        :return: vnEz or None if nCy > 1
        """
        if self.isSymmetric:
            return np.r_[self.nNx, self.nNy, self.nCz]
        else:
            return None

    @property
    def nEz(self):
        """
        Number of z-edges

        :rtype: int
        :return: nEz
        """
        if self.isSymmetric:
            return self.vnEz.prod()
        return (np.r_[self.nNx-1, self.nNy, self.nCz]).prod() + self.nCz

    @property
    def vectorCCx(self):
        """Cell-centered grid vector (1D) in the x direction."""
        return np.r_[0, self.hx[:-1].cumsum()] + self.hx*0.5

    @property
    def vectorCCy(self):
        """Cell-centered grid vector (1D) in the y direction."""
        return np.r_[0, self.hy[:-1]]

    @property
    def vectorNx(self):
        """Nodal grid vector (1D) in the x direction."""
        if self.isSymmetric:
            return self.hx.cumsum()
        return np.r_[0, self.hx].cumsum()

    @property
    def vectorNy(self):
        """Nodal grid vector (1D) in the y direction."""
        if self.isSymmetric:
            # There aren't really any nodes, but all the grids need
            # somewhere to live, why not zero?!
            return np.r_[0]
        return np.r_[0, self.hy[:-1].cumsum()] + self.hy[0]*0.5

    @property
    def edge(self):
        """Edge lengths"""
        if getattr(self, '_edge', None) is None:
            if self.isSymmetric:
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
            self._vol = np.kron(self.hz, az)
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
            if self.isSymmetric:
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
        # Nodal grad does not make sense for cylindrically symmetric mesh.
        if self.isSymmetric: return None
        raise NotImplementedError('nodalGrad not yet implemented')

    @property
    def nodalLaplacian(self):
        """Construct laplacian operator (nodes to edges)."""
        raise NotImplementedError('nodalLaplacian not yet implemented')

    @property
    def edgeCurl(self):
        """The edgeCurl property."""
        if self.nCy > 1:
            raise NotImplementedError('Edge curl not yet implemented for nCy > 1')
        if getattr(self, '_edgeCurl', None) is None:
            #1D Difference matricies
            dr = sp.spdiags((np.ones((self.nCx+1, 1))*[-1, 1]).T, [-1,0], self.nCx, self.nCx, format="csr")
            dz = sp.spdiags((np.ones((self.nCz+1, 1))*[-1, 1]).T, [0,1], self.nCz, self.nCz+1, format="csr")

            #2D Difference matricies
            Dr =  sp.kron(sp.identity(self.nNz), dr)
            Dz = -sp.kron(dz, sp.identity(self.nCx))

            A = self.area
            E = self.edge
            #Edge curl operator
            self._edgeCurl = sdiag(1/A)*sp.vstack((Dz, Dr))*sdiag(E)
        return self._edgeCurl

    # @property
    # def aveE2CC(self):
    #     """Averaging operator from cell edges to cell centers"""
    #     if getattr(self, '_aveE2CC', None) is None:
    #         if self.isSymmetric:
    #             az = sp.spdiags(0.5*np.ones((2, self.nNz)), [-1,0], self.nNz, self.nCz, format='csr')
    #             ar = sp.spdiags(0.5*np.ones((2, self.nCx)), [0, 1], self.nCx, self.nCx, format='csr')
    #             ar[0,0] = 1
    #             self._aveE2CC = (0.5)*sp.kron(az, ar).T
    #         else:
    #             raise NotImplementedError('wrapping in the averaging is not yet implemented')
    #     return self._aveE2CC

    @property
    def aveE2CC(self):
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, '_aveE2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if self.isSymmetric:
                avR = av(n[0])[:,1:]
                avR[0,0] = 1.
                self._aveE2CC = (0.5)*sp.kron(av(n[2]), avR, format="csr")
            else:
                raise NotImplementedError('wrapping in the averaging is not yet implemented')
                # self._aveE2CC = (1./3)*sp.hstack((kron3(av(n[2]), av(n[1]), speye(n[0])),
                #                                   kron3(av(n[2]), speye(n[1]), av(n[0])),
                #                                   kron3(speye(n[2]), av(n[1]), av(n[0]))), format="csr")
        return self._aveE2CC


    @property
    def aveF2CC(self):
        "Construct the averaging operator on cell faces to cell centers."
        if getattr(self, '_aveF2CC', None) is None:
            n = self.vnC
            if self.isSymmetric:
                avR = av(n[0])[:,1:]
                avR[0,0] = 1.
                self._aveF2CC = (0.5)*sp.hstack((sp.kron(speye(n[2]), avR), sp.kron(av(n[2]), speye(n[0]))), format="csr")
            else:
                raise NotImplementedError('wrapping in the averaging is not yet implemented')
                # self._aveF2CC = (1./3.)*sp.hstack((kron3(speye(n[2]), speye(n[1]), av(n[0])),
                #                                    kron3(speye(n[2]), av(n[1]), speye(n[0])),
                #                                    kron3(av(n[2]), speye(n[1]), speye(n[0]))), format="csr")
        return self._aveF2CC
