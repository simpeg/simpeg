import numpy as np, scipy.sparse as sp
from SimPEG.Utils import ndgrid, mkvc, sdiag
from BaseMesh import BaseMesh


NUM, ACTIVE, NX, NY, NZ = range(5) # Do not put anything after NZ
NUM, ACTIVE, PARENT, EDIR, ENODE0, ENODE1 = range(6)
NUM, ACTIVE, PARENT, FDIR, FEDGE0, FEDGE1, FEDGE2, FEDGE3 = range(8)
NUM, ACTIVE, PARENT, CFACE0, CFACE1, CFACE2, CFACE3, CFACE4, CFACE5 = range(9)

# The following classes are wrappers to make indexing easier

class TreeIndexer(object):
    def __init__(self, treeMesh, index=slice(None)):
        self.M = treeMesh
        if index == 'active':
            array = getattr(self.M, self._pointer)
            index = array[:,ACTIVE] == 1
        self.index = index

    @property
    def C(self): return getattr(self.M, '_cells', None)
    @property
    def F(self): return getattr(self.M, '_faces', None)
    @property
    def E(self): return getattr(self.M, '_edges', None)
    @property
    def N(self): return getattr(self.M, '_nodes', None)

    def sort(self, vec):
        self.M.number()
        P = np.argsort(self.num)
        if len(vec.shape) == 1:
            return vec[P]
        return vec[P,:]

    def _ind(self, column):
        array = getattr(self.M, self._pointer)
        ind = np.atleast_2d(array[self.index,:])[:,column]
        return self._SubTree(self.M, ind)

    def at(self, index=slice(None)):
        self.index = index
        return self

class TreeNode(TreeIndexer):
    _SubTree = None
    _pointer = '_nodes'
    @property
    def num(self): return self.N[self.index, NUM]
    @property
    def vec(self): return self.N[self.index,:][:,NX:]
    @property
    def x(self):   return self.N[self.index,:][:,NX]
    @property
    def y(self):   return self.N[self.index,:][:,NY]
    @property
    def z(self):   return self.N[self.index,:][:,NZ]

class TreeEdge(TreeIndexer):
    _SubTree = TreeNode
    _pointer = '_edges'

    @property
    def num(self):return self.E[self.index, NUM]
    @property
    def dir(self):return self.E[self.index, EDIR]
    @property
    def n0(self): return self._ind(ENODE0)
    @property
    def n1(self): return self._ind(ENODE1)
    @property
    def nodes(self):
        return [self.n0, self.n1]
    @property
    def length(self):
        return np.sum((self.n1.vec - self.n0.vec)**2,axis=1)**0.5
    @property
    def center(self):
        return (self.n0.vec + self.n1.vec)/2.0

class TreeFace(TreeIndexer):
    _SubTree = TreeEdge
    _pointer = '_faces'

    @property
    def num(self):return self.F[self.index, NUM]
    @property
    def dir(self):return self.F[self.index, FDIR]

    #            fX                                  fY                                 fZ
    #     n2___________n3                     n2___________n3                    n2___________n3
    #      |     e1    |                       |     e1    |                      |     e1    |
    #      |           |                       |           |                      |           |
    #   e2 |     x     | e3      z          e2 |     x     | e3      z         e2 |     x     | e3      y
    #      |           |         ^             |           |         ^            |           |         ^
    #      |___________|         |___> y       |___________|         |___> x      |___________|         |___> x
    #     n0    e0     n1                     n0    e0     n1                    n0    e0     n1

    @property
    def e0(self): return self._ind(FEDGE0)
    @property
    def e1(self): return self._ind(FEDGE1)
    @property
    def e2(self): return self._ind(FEDGE2)
    @property
    def e3(self): return self._ind(FEDGE3)
    @property
    def n0(self): return self.e0.n0
    @property
    def n1(self): return self.e0.n1
    @property
    def n2(self): return self.e1.n0
    @property
    def n3(self): return self.e1.n1
    @property
    def nodes(self):
        return [self.n0, self.n1, self.n2, self.n3]
    @property
    def center(self):
        return (self.n0.vec + self.n1.vec + self.n2.vec + self.n3.vec)/4.0
    @property
    def area(self):

        n0 = self.n0.vec #  2------3         3------2
        n1 = self.n1.vec #  |      |  --->   |      |
        n2 = self.n3.vec #  |      |         |      |
        n3 = self.n2.vec #  0------1         0------1

        a = np.sum((n1 - n0)**2,axis=1)**0.5
        b = np.sum((n2 - n1)**2,axis=1)**0.5
        c = np.sum((n3 - n2)**2,axis=1)**0.5
        d = np.sum((n0 - n3)**2,axis=1)**0.5
        p = np.sum((n2 - n0)**2,axis=1)**0.5
        q = np.sum((n3 - n1)**2,axis=1)**0.5

        # Area of an arbitrary quadrilateral (in a plane)
        V = 0.25 * (4.0*(p**2)*(q**2) - (a**2 + c**2 - b**2 - d**2)**2)**0.5
        return V

class TreeCell(TreeIndexer):
    _SubTree = TreeFace
    _pointer = '_cells'

    @property
    def num(self):return self.C[self.index, NUM]

    @property
    def fXm(self): return self._ind(CFACE0)
    @property
    def fXp(self): return self._ind(CFACE1)
    @property
    def fYm(self): return self._ind(CFACE2)
    @property
    def fYp(self): return self._ind(CFACE3)
    @property
    def fZm(self): return self._ind(CFACE4)
    @property
    def fZp(self): return self._ind(CFACE5)

    #                      fZp
    #                       |
    #                 6 ------eX3------ 7
    #                /|     |         / |
    #               /eZ2    .        / eZ3
    #             eY2 |        fYp eY3  |
    #             /   |            / fXp|
    #            4 ------eX2----- 5     |
    #            |fXm 2 -----eX1--|---- 3          z
    #           eZ0  /            |  eY1           ^   y
    #            | eY0   .  fYm  eZ1 /             |  /
    #            | /     |        | /              | /
    #            0 ------eX0------1                o----> x
    #                    |
    #                   fZm
    #
    #
    #            fX                                  fY                                 fZ
    #      2___________3                       2___________3                      2___________3
    #      |     e1    |                       |     e1    |                      |     e1    |
    #      |           |                       |           |                      |           |
    #   e2 |     x     | e3      z          e2 |     x     | e3      z         e2 |     x     | e3      y
    #      |           |         ^             |           |         ^            |           |         ^
    #      |___________|         |___> y       |___________|         |___> x      |___________|         |___> x
    #      0    e0     1                       0    e0     1                      0    e0     1
    #
    #  Mapping Nodes: numOnFace > numOnCell
    #
    #  fXm 0>0, 1>2, 2>4, 3>6              fYm 0>0, 1>1, 2>4, 3>5             fZm 0>0, 1>1, 2>2, 3>3
    #  fXp 0>1, 1>3, 2>5, 3>7              fYp 0>2, 1>3, 2>6, 3>7             fZp 0>4, 1>5, 2>6, 3>7

    @property
    def eX0(self): return self.fZm.e0
    @property
    def eX1(self): return self.fZm.e1
    @property
    def eX2(self): return self.fZp.e0
    @property
    def eX3(self): return self.fZp.e1

    @property
    def eY0(self): return self.fZm.e2
    @property
    def eY1(self): return self.fZm.e3
    @property
    def eY2(self): return self.fZp.e2
    @property
    def eY3(self): return self.fZp.e3

    @property
    def eZ0(self): return self.fXm.e2
    @property
    def eZ1(self): return self.fXp.e2
    @property
    def eZ2(self): return self.fXm.e3
    @property
    def eZ3(self): return self.fXp.e3

    @property
    def n0(self): return self.fZm.n0
    @property
    def n1(self): return self.fZm.n1
    @property
    def n2(self): return self.fZm.n2
    @property
    def n3(self): return self.fZm.n3
    @property
    def n4(self): return self.fZp.n0
    @property
    def n5(self): return self.fZp.n1
    @property
    def n6(self): return self.fZp.n2
    @property
    def n7(self): return self.fZp.n3

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
        return self._numbered
    @isNumbered.setter
    def isNumbered(self, value):
        assert value is False, 'Can only set to False.'
        self._numbered = False
        for name in ['vol', 'area', 'edge', 'gridCC', 'gridN', 'gridEx', 'gridEy', 'gridEz', 'gridFx', 'gridFy', 'gridFz']:
            if hasattr(self, '_'+name):
                delattr(self, '_'+name)

    def number(self):
        if self._numbered:
            return

        dtypeN = [('x',float),('y',float)]
        if self.dim == 3:
            dtypeN  += [('z',float)]
        dtypeV   = [('v', int)]

        N = TreeNode(self, 'active')
        E = TreeEdge(self, 'active')
        F = TreeFace(self, 'active')
        self._nodes[:,NUM] = -1
        self._edges[:,NUM] = -1
        self._faces[:,NUM] = -1
        if self.dim == 3:
            C = TreeCell(self, 'active')
            self._cells[:,NUM] = -1


        def doNumbering(indexer, nodes, dtype):
            grid = np.zeros(np.sum(indexer.index), dtype=dtype)
            grid['x'][:] = nodes.x
            grid['y'][:] = nodes.y
            if self.dim == 3:
                grid['z'][:] = nodes.z
            if 'v' in [d[0] for d in dtype]:
                grid['v'][:] = indexer.dir
            P = np.argsort(grid, order=[d[0] for d in reversed(dtype)])
            cnt = np.zeros(P.size, dtype=int)
            cnt[P] = np.arange(P.size)
            return cnt

        self._nodes[N.index, NUM] = doNumbering(N, N, dtypeN)

        self._edges[E.index, NUM] = doNumbering(E, E.n0, dtypeN + dtypeV)

        dtype = dtypeN if self.dim == 2 else (dtypeN + dtypeV)
        self._faces[F.index, NUM] = doNumbering(F, F.n0, dtype)

        if self.dim == 3:
            self._cells[C.index, NUM] = doNumbering(C, C.n0, dtypeN)

        self._numbered = True

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
        if self.dim == 2: return None
        return np.sum((self._edges[:,ACTIVE] == 1) & (self._edges[:,EDIR] == 2))

    @property
    def nFx(self):
        if self.dim == 2: return self.nEy
        return np.sum((self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 0))

    @property
    def nFy(self):
        if self.dim == 2: return self.nEx
        return np.sum((self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 1))

    @property
    def nFz(self):
        if self.dim == 2: return None
        return np.sum((self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 2))

    @property
    def edge(self):
        if getattr(self, '_edge', None) is None:
            E = TreeEdge(self, 'active')
            self._edge = E.sort(E.length)
        return self._edge

    @property
    def area(self):
        if getattr(self, '_area', None) is None:
            if self.dim == 2:
                self._area = np.r_[self.edge[self.nEx:], self.edge[:self.nEx]]
        return self._area

    @property
    def vol(self):
        if getattr(self, '_vol', None) is None:
            F = TreeFace(self, 'active')
            self._vol = F.sort(F.area)
        return self._vol

    @property
    def gridN(self):
        N = TreeNode(self, 'active')
        return N.sort(N.vec)

    @property
    def gridCC(self):
        F = TreeFace(self, 'active')
        return F.sort(F.center)

    @property
    def gridEx(self):
        E = TreeEdge(self, (self._edges[:,ACTIVE] == 1) & (self._edges[:,EDIR] == 0))
        return E.sort(E.center)

    @property
    def gridEy(self):
        E = TreeEdge(self, (self._edges[:,ACTIVE] == 1) & (self._edges[:,EDIR] == 1))
        return E.sort(E.center)

    @property
    def gridEz(self):
        if self.dim == 2: return None
        E = TreeEdge(self, (self._edges[:,ACTIVE] == 1) & (self._edges[:,EDIR] == 2))
        return E.sort(E.center)

    @property
    def gridFx(self):
        if self.dim == 2:
            return self.gridEy
        else:
            F = TreeFace(self, (self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 0))
            return F.sort(F.center)

    @property
    def gridFy(self):
        if self.dim == 2:
            return self.gridEx
        else:
            F = TreeFace(self, (self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 1))
            return F.sort(F.center)

    @property
    def gridFz(self):
        if self.dim == 2: return None
        F = TreeFace(self, (self._faces[:,ACTIVE] == 1) & (self._faces[:,FDIR] == 2))
        return F.sort(F.center)

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

        # Refine the outer edges
        E0i, E0 = self.refineEdge(f[FEDGE0])
        E1i, E1 = self.refineEdge(f[FEDGE1])
        E2i, E2 = self.refineEdge(f[FEDGE2])
        E3i, E3 = self.refineEdge(f[FEDGE3])

        nodeNums = self._edges[f[[FEDGE0, FEDGE1]],:][:,[ENODE0, ENODE1]]
        newNode, node = self.addNode(nodeNums)

        # Refine the inner edges
        #                                           new faces and edges
        #      2_______________3                    _______________
        #      |     e1-->     |                   |       |       |
        #   ^  |               | ^                 |   2   3   3   |        y            z            z
        #   |  |               | |                 |       |       |        ^            ^            ^
        #   |  |       +       | |      --->       |---0---+---1---|        |            |            |
        #   e2 |               | e3                |       |       |        |            |            |
        #      |               |                   |   0   2   1   |        z-----> x    y-----> x    x-----> y
        #      |_______________|                   |_______|_______|
        #      0      e0-->    1

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


    def refineCell(self, index):
        c = self._cells[index,:]
        if f[ACTIVE] == 0:
            # search for the children up to one level deep
            subInds = np.argwhere(self._cells[:,PARENT] == index).flatten()
            return subInds, self._cells[subInds,:]

        self._cells[index, ACTIVE] = 0

        # Refine the outer faces
        F0i, F0 = self.refineFace(c[CFACE0])
        F1i, F1 = self.refineFace(c[CFACE1])
        F2i, F2 = self.refineFace(c[CFACE2])
        F3i, F3 = self.refineFace(c[CFACE3])
        F4i, F4 = self.refineFace(c[CFACE4])
        F5i, F5 = self.refineFace(c[CFACE5])

        nodeNums = self._edges[f[[FEDGE0, FEDGE1]],:][:,[ENODE0, ENODE1]]
        newNode, node = self.addNode(nodeNums)


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

    print tM._nodes[:,NUM]
    tM.number()
    print tM._nodes[:,NUM]
    print tM._edges[:,NUM]

    print TreeFace(tM,[0]).e2.n0.x




    # print tM._faces
    # print tM._edges[0,:]
    # print tM.vol


    # tM.number()
    # print tM._index('_edges',3)[1]


    # print tM._edges[:,[0,1,3, 4,5 ]]

    plt.subplot(211)
    plt.spy(tM.faceDiv)
    tM.plotGrid(ax=plt.subplot(212))

    # plt.figure(2)
    # plt.plot(SortByX0(tM.gridCC),'b.')
    plt.show()
