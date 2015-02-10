import numpy as np, scipy.sparse as sp
from SimPEG.Utils import ndgrid, mkvc, sdiag
from BaseMesh import BaseMesh


NUM, ACTIVE, NX, NY, NZ = range(5)
NUM, ACTIVE, PARENT, EDIR, ENODE0, ENODE1 = range(6)
NUM, ACTIVE, PARENT, FDIR, FEDGE0, FEDGE1, FEDGE2, FEDGE3 = range(8)
NUM, ACTIVE, PARENT, CFACE0, CFACE1, CFACE2, CFACE3, CFACE4, CFACE5 = range(9)

def SortByX0(grid):
    dtype=[('x',float),('y',float)]
    grid2 = np.zeros(grid.shape[0], dtype=dtype)
    grid2['x'][:] = grid[:,0]
    grid2['y'][:] = grid[:,1]
    P = np.argsort(grid2, order=['y','x'])
    return P


class TreeMesh(BaseMesh):

    def __init__(self, h_in, x0=None):
        assert type(h_in) in [list, tuple], 'h_in must be a list'
        assert len(h_in) > 1, "len(h_in) must be greater than 1"

        h = range(len(h_in))
        for i, h_i in enumerate(h_in):
            if type(h_i) in [int, long, float]:
                # This gives you something over the unit cube.
                h_i = np.ones(int(h_i))/int(h_i)
            assert isinstance(h_i, np.ndarray), ("h[%i] is not a numpy array." % i)
            assert len(h_i.shape) == 1, ("h[%i] must be a 1D numpy array." % i)
            h[i] = h_i[:] # make a copy.
        self.h = h

        if x0 is None:
            x0 = np.zeros(len(h))
        else:
            assert type(x0) in [list, tuple, np.ndarray], 'x0 must be an array'
            x0 = np.array(x0, dtype=float)
            assert len(x0) == self.dim, 'x0 must have the same dimensions as the mesh'

        # TODO: this has a lot of stuff which doesn't work for this style of mesh...
        BaseMesh.__init__(self, np.array([x.size for x in h]), x0)
        if self.dim == 2:
            self._init2D()
        else:
            self._init3D()

        self.isNumbered = False

    def _init2D(self):
        XY = ndgrid(*[np.r_[0, h.cumsum()] for h in self.h])

        nCx, nCy = [len(h) for h in self.h]

        vnC  = [nCx  , nCy  ]
        vnN  = [nCx+1, nCy+1]

        vnEx = [nCx  , nCy+1]
        vnEy = [nCx+1, nCy  ]

        vnFx = [nCx+1, nCy  ]
        vnFy = [nCx  , nCy+1]

        nC   = np.prod(vnC)
        nN   = np.prod(vnN)
        nFx  = np.prod(vnFx)
        nFy  = np.prod(vnFy)
        nF   = nFx + nFy
        nEx  = np.prod(vnEx)
        nEy  = np.prod(vnEy)
        nE   = nEx + nEy

        N = np.c_[np.arange(nN), np.ones(nN), XY]

        iN = np.arange(nN, dtype=int).reshape(vnN, order='F')

        # Pointers to the nodes for the edges
        pnEx = np.c_[mkvc(iN[:-1,:]), mkvc(iN[1:,:])]
        pnEy = np.c_[mkvc(iN[:,:-1]), mkvc(iN[:,1:])]

        iEx = np.arange(nEx, dtype=int).reshape(*vnEx, order='F')
        iEy = np.arange(nEy, dtype=int).reshape(*vnEy, order='F') + nEx

        zEx = np.zeros(nEx, dtype=int)
        zEy = np.zeros(nEy, dtype=int)

        Ex = np.c_[mkvc(iEx), zEx+1, zEx-1, zEx+0, pnEx]
        Ey = np.c_[mkvc(iEy), zEy+1, zEy-1, zEy+1, pnEy]

        # Pointers to the edges for the faces
        vFz = np.c_[mkvc(iEx[:,:-1]), mkvc(iEx[:,1:]), mkvc(iEy[:-1,:]), mkvc(iEy[1:,:])]

        iC = np.arange(nC, dtype=int)

        zC = np.zeros(nC, dtype=int)

        C = np.c_[iC, zC+1, zC-1, zC+2, vFz]

        self._nodes = N
        self._edges = np.r_[Ex, Ey]
        self._faces = C


    def _init3D(self):
        XYZ = ndgrid(*[np.r_[0, h.cumsum()] for h in self.h])

        nCx, nCy, nCz = [len(h) for h in self.h]

        vnC  = [nCx  , nCy  , nCz  ]
        vnN  = [nCx+1, nCy+1, nCz+1]

        vnEx = [nCx  , nCy+1, nCz+1]
        vnEy = [nCx+1, nCy  , nCz+1]
        vnEz = [nCx+1, nCy+1, nCz  ]

        vnFx = [nCx+1, nCy  , nCz  ]
        vnFy = [nCx  , nCy+1, nCz  ]
        vnFz = [nCx  , nCy  , nCz+1]

        nC   = np.prod(vnC)
        nN   = np.prod(vnN)
        nFx  = np.prod(vnFx)
        nFy  = np.prod(vnFy)
        nFz  = np.prod(vnFz)
        nF   = nFx + nFy + nFz
        nEx  = np.prod(vnEx)
        nEy  = np.prod(vnEy)
        nEz  = np.prod(vnEz)
        nE   = nEx + nEy + nEz

        N = np.c_[np.arange(XYZ.shape[0]), np.ones(XYZ.shape[0]), XYZ]

        iN = np.arange(nN, dtype=int).reshape(vnN, order='F')

        # Pointers to the nodes for the edges
        pnEx = np.c_[mkvc(iN[:-1,:,:]), mkvc(iN[1:,:,:])]
        pnEy = np.c_[mkvc(iN[:,:-1,:]), mkvc(iN[:,1:,:])]
        pnEz = np.c_[mkvc(iN[:,:,:-1]), mkvc(iN[:,:,1:])]

        iEx = np.arange(nEx, dtype=int).reshape(*vnEx, order='F')
        iEy = np.arange(nEy, dtype=int).reshape(*vnEy, order='F') + nEx
        iEz = np.arange(nEz, dtype=int).reshape(*vnEz, order='F') + nEx + nEy

        zEx = np.zeros(nEx, dtype=int)
        zEy = np.zeros(nEy, dtype=int)
        zEz = np.zeros(nEz, dtype=int)

        Ex = np.c_[mkvc(iEx), zEx+1, zEx-1, zEx+0, pnEx]
        Ey = np.c_[mkvc(iEy), zEy+1, zEy-1, zEy+1, pnEy]
        Ez = np.c_[mkvc(iEz), zEz+1, zEz-1, zEz+2, pnEz]

        # Pointers to the edges for the faces
        peFx = np.c_[                                       mkvc(iEy[:,:,:-1]), mkvc(iEy[:,:,1:]), mkvc(iEz[:,:-1,:]), mkvc(iEz[:,1:,:])]
        peFy = np.c_[mkvc(iEx[:,:,:-1]), mkvc(iEx[:,:,1:]),                                        mkvc(iEz[:-1,:,:]), mkvc(iEz[1:,:,:])]
        peFz = np.c_[mkvc(iEx[:,:-1,:]), mkvc(iEx[:,1:,:]), mkvc(iEy[:-1,:,:]), mkvc(iEy[1:,:,:])                                       ]

        iFx = np.arange(nFx, dtype=int).reshape(*vnFx, order='F')
        iFy = np.arange(nFy, dtype=int).reshape(*vnFy, order='F') + nFx
        iFz = np.arange(nFz, dtype=int).reshape(*vnFz, order='F') + nFx + nFy

        zFx = np.zeros(nFx, dtype=int)
        zFy = np.zeros(nFy, dtype=int)
        zFz = np.zeros(nFz, dtype=int)

        Fx = np.c_[mkvc(iFx), zFx+1, zFx-1, zFx+0, peFx]
        Fy = np.c_[mkvc(iFy), zFy+1, zFy-1, zFy+1, peFy]
        Fz = np.c_[mkvc(iFz), zFz+1, zFz-1, zFz+2, peFz]

        # Pointers to the faces for the cells
        pfCx = np.c_[mkvc(iFx[:-1,:,:]), mkvc(iFx[1:,:,:])]
        pfCy = np.c_[mkvc(iFy[:,:-1,:]), mkvc(iFy[:,1:,:])]
        pfCz = np.c_[mkvc(iFz[:,:,:-1]), mkvc(iFz[:,:,1:])]

        iC = np.arange(nC, dtype=int)

        zC = np.zeros(nC, dtype=int)

        C = np.c_[iC, zC+1, zC-1, pfCx, pfCy, pfCz]

        self._nodes = N
        self._edges = np.r_[Ex, Ey, Ez]
        self._faces = np.r_[Fx, Fy, Fz]
        self._cells = C

    @property
    def isNumbered(self):
        return self._numberedCC and self._numberedN and self._numberedEx and self._numberedEy
    @isNumbered.setter
    def isNumbered(self, value):
        assert value is False, 'Can only set to False.'
        self._numberedCC = False
        self._numberedN  = False
        self._numberedEx = False
        self._numberedEy = False
        for name in ['vol', 'area', 'edge', 'gridCC', 'gridN', 'gridEx', 'gridEy', 'gridEz', 'gridFx', 'gridFy', 'gridFz']:
            if hasattr(self, '_'+name):
                delattr(self, '_'+name)

    @property
    def nC(self):
        if self.dim == 2:
            return np.sum(self._faces[:,ACTIVE] == 1)
        return np.sum(self._cells[:,ACTIVE] == 1)

    @property
    def nN(self):
        return np.sum(self._nodes[:,ACTIVE] == 1)

    @property
    def nE(self):
        if self.dim == 2:
            return self.nEx + self.nEy
        return self.nEx + self.nEy + self.nEz

    @property
    def nF(self):
        if self.dim == 2:
            return self.nFx + self.nFy
        return self.nFx + self.nFy + self.nFz

    @property
    def nEx(self):
        return np.sum((self._edges[:,ACTIVE] == 1) & (self._edges[:,EDIR] == 0))

    @property
    def nEy(self):
        return np.sum((self._edges[:,ACTIVE] == 1) & (self._edges[:,EDIR] == 1))

    @property
    def nEz(self):
        if self.dim == 2:
            return None
        return np.sum((self._edges[:,ACTIVE] == 1) & (self._edges[:,EDIR] == 2))

    @property
    def nFx(self):
        if self.dim == 2:
            return self.nEy
        return np.sum((self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 0))

    @property
    def nFy(self):
        if self.dim == 2:
            return self.nEx
        return np.sum((self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 1))

    @property
    def nFz(self):
        if self.dim == 2:
            return None
        return np.sum((self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 2))

    @property
    def edge(self):
        if getattr(self, '_edge', None) is None:
            self.number()

            N = self._nodes
            E = self._edges
            activeEdges = E[:,ACTIVE] == 1
            e0xy = N[E[activeEdges,ENODE0],:][:,[NX,NY]]
            e1xy = N[E[activeEdges,ENODE1],:][:,[NX,NY]]

            A = np.sum((e1xy - e0xy)**2,axis=1)**0.5

            P = np.argsort(E[activeEdges,NUM])
            self._edge = A[P]

        return self._edge

    @property
    def area(self):
        if getattr(self, '_area', None) is None:
            self.number()
            if self.dim == 2:
                self._area = np.r_[self.edge[self.nEx:], self.edge[:self.nEx]]

        return self._area


    @property
    def vol(self):
        if getattr(self, '_vol', None) is None:
            self.number()

            N = self._nodes
            E = self._edges
            C = self._faces
            activeCells = C[:,ACTIVE] == 1
            nInds1 = E[C[activeCells,FEDGE0],:][:,[ENODE0,ENODE1]]
            nInds2 = E[C[activeCells,FEDGE1],:][:,[ENODE0,ENODE1]]
            n0 = N[nInds1[:,0],:][:,[NX,NY]] #   2------3       3------2
            n1 = N[nInds1[:,1],:][:,[NX,NY]] #   |      |  -->  |      |
            n3 = N[nInds2[:,0],:][:,[NX,NY]] #   |      |       |      |
            n2 = N[nInds2[:,1],:][:,[NX,NY]] #   0------1       0------1

            a = np.sum((n1 - n0)**2,axis=1)**0.5
            b = np.sum((n2 - n1)**2,axis=1)**0.5
            c = np.sum((n3 - n2)**2,axis=1)**0.5
            d = np.sum((n0 - n3)**2,axis=1)**0.5
            p = np.sum((n2 - n0)**2,axis=1)**0.5
            q = np.sum((n3 - n1)**2,axis=1)**0.5

            # Area of an arbitrary quadrilateral (in a plane)
            V = 0.25 * (4.0*(p**2)*(q**2) - (a**2 + c**2 - b**2 - d**2)**2)**0.5
            P = np.argsort(C[activeCells,NUM])
            self._vol = V[P]

        return self._vol

    @property
    def gridN(self):
        N = self._nodes
        activeNodes = N[:,ACTIVE] == 1
        Nx = N[activeNodes,NX]
        Ny = N[activeNodes,NY]

        P = SortByX0(np.c_[Nx, Ny])
        if not self._numberedN:
            cnt = np.zeros(P.size, dtype=int)
            cnt[P] = np.arange(P.size)
            self._nodes[activeNodes, NUM] = cnt
            self._numberedN = True

        return np.c_[Nx, Ny][P, :]

    @property
    def gridCC(self):
        N = self._nodes
        E = self._edges
        C = self._faces
        activeCells = C[:,ACTIVE] == 1
        nInds1 = E[C[activeCells,FEDGE0],:][:,[ENODE0,ENODE1]]
        nInds2 = E[C[activeCells,FEDGE1],:][:,[ENODE0,ENODE1]]
        Cx = (N[nInds1[:,0],NX] + N[nInds1[:,1],NX] + N[nInds2[:,0],NX] + N[nInds2[:,1],NX])/4.0
        Cy = (N[nInds1[:,0],NY] + N[nInds1[:,1],NY] + N[nInds2[:,0],NY] + N[nInds2[:,1],NY])/4.0

        P = SortByX0(np.c_[N[nInds1[:,0],NX], N[nInds1[:,0],NY]])
        if not self._numberedCC:
            cnt = np.zeros(P.size, dtype=int)
            cnt[P] = np.arange(P.size)
            self._faces[activeCells, NUM] = cnt
            self._numberedCC = True

        return np.c_[Cx, Cy][P, :]

    @property
    def gridEx(self):
        N = self._nodes
        E = self._edges
        C = self._faces
        activeEdges = (E[:,ACTIVE] == 1) & (E[:,EDIR] == 0)
        nInds = E[activeEdges,:][:,[ENODE0,ENODE1]]
        Ex = (N[nInds[:,0],NX] + N[nInds[:,1],NX])/2.0
        Ey = (N[nInds[:,0],NY] + N[nInds[:,1],NY])/2.0

        P = SortByX0(np.c_[N[nInds[:,0],NX], N[nInds[:,0],NY]])
        if not self._numberedEx:
            cnt = np.zeros(P.size, dtype=int)
            cnt[P] = np.arange(P.size)
            self._edges[activeEdges, NUM] = cnt
            self._numberedEx = True

        return np.c_[Ex, Ey][P, :]

    @property
    def gridEy(self):
        N = self._nodes
        E = self._edges
        C = self._faces
        activeEdges = (E[:,ACTIVE] == 1) & (E[:,EDIR] == 1)
        nInds = E[activeEdges,:][:,[ENODE0,ENODE1]]
        Ex = (N[nInds[:,0],NX] + N[nInds[:,1],NX])/2.0
        Ey = (N[nInds[:,0],NY] + N[nInds[:,1],NY])/2.0

        P = SortByX0(np.c_[N[nInds[:,0],NX], N[nInds[:,0],NY]])
        if not self._numberedEy:
            cnt = np.zeros(P.size, dtype=int)
            cnt[P] = np.arange(P.size)
            self._edges[activeEdges, NUM] = cnt + self.nEx
            self._numberedEy = True

        return np.c_[Ex, Ey][P, :]


    @property
    def gridFx(self):
        if self.dim == 2:
            return self.gridEy

    @property
    def gridFy(self):
        if self.dim == 2:
            return self.gridEx

    def _push(self, attr, rows):
        self.isNumbered = False
        rows = np.atleast_2d(rows)
        X = getattr(self, attr)
        offset = X.shape[0]
        rowNumer = np.arange(rows.shape[0], dtype=int) + offset
        rows[:,0] = rowNumer*0-1
        setattr(self, attr, np.vstack((X, rows)).astype(X.dtype))
        if rows.shape[0] == 1:
            return offset, rows.flatten()
        return rowNumer, rows

    def addNode(self, between):
        """Add a node between the node in list between"""
        between = np.array(between).flatten()
        nodes = self._nodes[between.astype(int), :]
        newNode = np.mean(nodes, axis=0)
        newNode[ACTIVE] = 1
        return self._push('_nodes', newNode)

    def refineEdge(self, index):
        e = self._edges[index,:]
        if e[ACTIVE] == 0:
            # search for the children up to one level deep
            subInds = np.argwhere(self._edges[:,PARENT] == index).flatten()
            return subInds, self._edges[subInds,:]

        self._edges[index, ACTIVE] = 0

        newNode, node = self.addNode(e[[ENODE0, ENODE1]])

        Es = np.zeros((2, 6))
        Es[:, ACTIVE]  = 1
        Es[:, PARENT]  = index
        Es[:, EDIR]    = e[EDIR]
        Es[0, ENODE0]  = e[ENODE0]
        Es[0, ENODE1]  = newNode
        Es[1, ENODE0]  = newNode
        Es[1, ENODE1]  = e[ENODE1]
        return self._push('_edges', Es)

    def refineFace(self, index):
        f = self._faces[index,:]
        if f[ACTIVE] == 0:
            # search for the children up to one level deep
            subInds = np.argwhere(self._faces[:,PARENT] == index).flatten()
            return subInds, self._faces[subInds,:]

        self._faces[index, ACTIVE] = 0

        #                                           new faces and edges
        #      2_______________3                    _______________
        #      |     e1-->     |                   |       |       |
        #   ^  |               | ^                 |   2   3   3   |        y            z            z
        #   |  |               | |                 |       |       |        ^            ^            ^
        #   |  |       x       | |      --->       |---0---+---1---|        |            |            |
        #   e2 |               | e3                |       |       |        |            |            |
        #      |               |                   |   0   2   1   |        z-----> x    y-----> x    x-----> y
        #      |_______________|                   |_______|_______|
        #      0      e0-->    1

        # Refine the outer edges
        E0i, E0 = self.refineEdge(f[FEDGE0])
        E1i, E1 = self.refineEdge(f[FEDGE1])
        E2i, E2 = self.refineEdge(f[FEDGE2])
        E3i, E3 = self.refineEdge(f[FEDGE3])

        nodeNums = self._edges[f[[FEDGE0, FEDGE1]],:][:,[ENODE0, ENODE1]]
        newNode, node = self.addNode(nodeNums)

        # Refine the inner edges
        nE = np.zeros((4,6))
        nE[:, ACTIVE] = 1
        nE[:, PARENT] = -1
        nE[:, EDIR] = [0,0,1,1] if f[FDIR] == 2 else [0,0,2,2] if f[FDIR] == 1 else [1,1,2,2]
        nE[0, ENODE0] = E2[0, ENODE1]
        nE[0, ENODE1] = newNode
        nE[1, ENODE0] = newNode
        nE[1, ENODE1] = E3[0, ENODE1]
        nE[2, ENODE0] = E0[0, ENODE1]
        nE[2, ENODE1] = newNode
        nE[3, ENODE0] = newNode
        nE[3, ENODE1] = E1[0, ENODE1]
        nEi, nE = self._push('_edges', nE)

        # Add four new faces
        Fs = np.zeros((4,8))
        Fs[:, ACTIVE] = 1
        Fs[:, PARENT] = index
        Fs[:, FDIR]   = f[FDIR]

        fInds = [FEDGE0,FEDGE1,FEDGE2,FEDGE3]
        Fs[0, fInds] = [E0i[0], nEi[0], E2i[0], nEi[2]]
        Fs[1, fInds] = [E0i[1], nEi[1], nEi[2], E3i[0]]
        Fs[2, fInds] = [nEi[0], E1i[0], E2i[1], nEi[3]]
        Fs[3, fInds] = [nEi[1], E1i[1], nEi[3], E3i[1]]

        return self._push('_faces', Fs)

    def _index(self, attr, index):
        index = [index] if np.isscalar(index) else list(index)
        C = getattr(self, attr)
        cSub = []
        iSub = []
        for I in index:
            if C[I, ACTIVE] == 1:
                iSub += [I]
                cSub += [C[I, :]]
            else:
                subInds = np.argwhere(C[:,PARENT] == I).flatten()
                i, c = self._index(attr, subInds)
                iSub += i
                cSub += [c]
        return iSub, np.vstack(cSub)

    @property
    def faceDiv(self):
        if getattr(self, '_faceDiv', None) is None:
            self.number()
            # TODO: Preallocate!
            I, J, V = [], [], []

            offset = np.r_[self.nFx, -self.nEx] # this switches from edge to face numbering
            C = self._faces
            activeCells = C[:,ACTIVE] == 1
            for cell in C[activeCells]:
                for sign, face in zip([-1,1,-1,1],[FEDGE0, FEDGE1, FEDGE2, FEDGE3]):
                    ij, jrow = self._index('_edges', cell[face])
                    I += [cell[NUM]]*len(ij)
                    J += list(jrow[:,0] + offset[jrow[:,EDIR]])
                    V += [sign]*len(ij)
            VOL = self.vol
            D = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.nF))
            S = self.area
            self._faceDiv = sdiag(1/VOL)*D*sdiag(S)
        return self._faceDiv

    def number(self):
        if self.isNumbered:
            return
        self._nodes[:,NUM] = -1
        self._edges[:,NUM] = -1
        self._faces[:,NUM] = -1
        self.gridCC
        self.gridN
        self.gridEx
        self.gridEy
        if self.dim > 2:
            self._cells[:,NUM] = -1
            self.gridEz
            self.gridFx
            self.gridFy
            self.gridFz

    def plotGrid(self, ax=None, text=True, showIt=False):
        import matplotlib.pyplot as plt


        axOpts = {'projection':'3d'} if self.dim == 3 else {}
        if ax is None: ax = plt.subplot(111, **axOpts)

        N = self._nodes
        E = self._edges
        C = self._faces

        plt.plot(N[:,1], N[:,2], 'b.')
        activeCells = C[:,ACTIVE] == 1
        for FEDGE in [FEDGE0, FEDGE1, FEDGE2, FEDGE3]:
            nInds = E[C[activeCells,FEDGE],:][:,[ENODE0,ENODE1]]
            eX = np.c_[N[nInds[:,0],NX], N[nInds[:,1],NX], [np.nan]*nInds.shape[0]]
            eY = np.c_[N[nInds[:,0],NY], N[nInds[:,1],NY], [np.nan]*nInds.shape[0]]
            plt.plot(eX.flatten(), eY.flatten(), 'b-')

        gridCC = self.gridCC
        if text:
            [ax.text(cc[0], cc[1],i) for i, cc in enumerate(gridCC)]
        plt.plot(gridCC[:,0], gridCC[:,1], 'r.')
        gridFx = self.gridFx
        gridFy = self.gridFy
        if text:
            [ax.text(cc[0], cc[1],i) for i, cc in enumerate(np.vstack((gridFx,gridFy)))]
        gridEx = self.gridEx
        gridEy = self.gridEy
        # if text:
        #     [ax.text(cc[0], cc[1],i) for i, cc in enumerate(np.vstack((gridEx,gridEy)))]

        # for E in self._edges:
        #     if E[ACTIVE] == 0: continue
        #     ex = N[E[[ENODE0,ENODE1]],NX]
        #     ey = N[E[[ENODE0,ENODE1]],NY]
        #     ax.plot(ex, ey, 'b-')
        #     ax.text(ex.mean(), ey.mean(), E[NUM])

        if showIt:
            plt.show()

if __name__ == '__main__':
    from SimPEG import Mesh, Utils
    import matplotlib.pyplot as plt

    tM = TreeMesh([np.ones(3),np.ones(2)])

    tM.refineFace(0)
    tM.refineFace(1)
    tM.refineFace(3)
    tM.refineFace(9)

    # print tM._faces
    # print tM._edges[0,:]
    # print tM.area


    # tM.number()
    # print tM._index('_edges',3)[1]


    # print tM._edges[:,[0,1,3, 4,5 ]]

    plt.subplot(211)
    plt.spy(tM.faceDiv)
    tM.plotGrid(ax=plt.subplot(212))

    # plt.figure(2)
    # plt.plot(SortByX0(tM.gridCC),'b.')
    plt.show()



