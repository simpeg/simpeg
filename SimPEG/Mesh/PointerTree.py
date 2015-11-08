from SimPEG import np, sp, Utils, Solver, Mesh
import matplotlib.pyplot as plt
import matplotlib
import TreeUtils
import time

MAX_BITS = 20

def SortGrid(grid, offset=0):
    """
        Sorts a grid by the x0 location.
    """

    eps = 1e-7
    def mycmp(c1,c2):
        c1 = grid[c1-offset]
        c2 = grid[c2-offset]
        if c1.size == 2:
            if np.abs(c1[1] - c2[1]) < eps:
                return c1[0] - c2[0]
            return c1[1] - c2[1]
        elif c1.size == 3:
            if np.abs(c1[2] - c2[2]) < eps:
                if np.abs(c1[1] - c2[1]) < eps:
                    return c1[0] - c2[0]
                return c1[1] - c2[1]
            return c1[2] - c2[2]

    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return sorted(range(offset,grid.shape[0]+offset), key=K)


class NotBalancedException(Exception):
    pass

class Tree(object):
    def __init__(self, h_in, levels=3):
        assert type(h_in) is list, 'h_in must be a list'
        assert len(h_in) > 1, "len(h_in) must be greater than 1"

        h = range(len(h_in))
        for i, h_i in enumerate(h_in):
            if type(h_i) in [int, long, float]:
                # This gives you something over the unit cube.
                h_i = np.ones(int(h_i))/int(h_i)
            elif type(h_i) is list:
                h_i = Utils.meshTensor(h_i)
            assert isinstance(h_i, np.ndarray), ("h[%i] is not a numpy array." % i)
            assert len(h_i.shape) == 1, ("h[%i] must be a 1D numpy array." % i)
            assert len(h_i) == 2**levels, "must make h and levels match"
            h[i] = h_i[:] # make a copy.
        self.h = h

        self._levels = levels
        self._levelBits = int(np.ceil(np.sqrt(levels)))+1

        self.__dirty__ = True #: The numbering is dirty!

        self._cells = set()
        self._cells.add(0)

    @property
    def __dirty__(self):
        return self.__dirtyFaces__ or self.__dirtyEdges__ or self.__dirtyNodes__ or self.__dirtyHanging__

    @__dirty__.setter
    def __dirty__(self, val):
        assert val is True
        self.__dirtyFaces__ = True
        self.__dirtyEdges__ = True
        self.__dirtyNodes__ = True
        self.__dirtyHanging__ = True

    @property
    def levels(self): return self._levels

    @property
    def dim(self): return len(self.h)

    @property
    def nC(self): return len(self._cells)

    @property
    def nN(self):
        self.number()
        return len(self._nodes) - len(self._hangingN)

    @property
    def nF(self):
        return self.nFx + self.nFy + (0 if self.dim == 2 else self.nFz)

    @property
    def nFx(self):
        self.number()
        return len(self._facesX) - len(self._hangingFx)

    @property
    def nFy(self):
        self.number()
        return len(self._facesY) - len(self._hangingFy)

    @property
    def nFz(self):
        if self.dim == 2: return None
        self.number()
        return len(self._facesZ) - len(self._hangingFz)

    @property
    def nE(self):
        return self.nEx + self.nEy + (0 if self.dim == 2 else self.nEz)

    @property
    def nEx(self):
        if self.dim == 2:return self.nFy
        self.number()
        return len(self._edgesX) - len(self._hangingEx)

    @property
    def nEy(self):
        if self.dim == 2:return self.nFx
        self.number()
        return len(self._edgesY) - len(self._hangingEy)

    @property
    def nEz(self):
        if self.dim == 2: return None
        self.number()
        return len(self._edgesZ) - len(self._hangingEz)

    @property
    def _sortedCells(self):
        if getattr(self, '__sortedCells', None) is None:
            self.__sortedCells = sorted(self._cells)
        return self.__sortedCells

    @property
    def permuteCC(self):
        #TODO: cache these?
        P  = SortGrid(self.gridCC)
        return sp.identity(self.nC).tocsr()[P,:]

    @property
    def permuteF(self):
        #TODO: cache these?
        P = SortGrid(self.gridFx)
        P += SortGrid(self.gridFy, offset=self.nFx)
        if self.dim == 3:
            P += SortGrid(self.gridFz, offset=self.nFx+self.nFy)
        return sp.identity(self.nF).tocsr()[P,:]

    @property
    def permuteE(self):
        #TODO: cache these?
        if self.dim == 2:
            P = SortGrid(self.gridFy)
            P += SortGrid(self.gridFx, offset=self.nEx)
            return sp.identity(self.nE).tocsr()[P,:]
        if self.dim == 3:
            raise Exception()

    def _structureChange(self):
        deleteThese = [
                        '__sortedCells',
                        '_gridCC', '_gridN', '_gridFx', '_gridFy', '_gridFz', '_gridEx', '_gridEy', '_gridEz',
                        '_area', '_edge', '_vol',
                        '_faceDiv', '_edgeCurl', '_nodalGrad'
                      ]
        for p in deleteThese:
            if hasattr(self, p): delattr(self, p)
        self.__dirty__ = True

    def _index(self, pointer):
        assert len(pointer) is self.dim+1
        assert pointer[-1] <= self.levels
        return TreeUtils.index(self.dim, MAX_BITS, self._levelBits, pointer[:-1], pointer[-1])

    def _pointer(self, index):
        assert type(index) in [int, long]
        return TreeUtils.point(self.dim, MAX_BITS, self._levelBits, index)

    def __contains__(self, v):
        if type(v) in [int, long]:
            return v in self._cells
        return self._index(v) in self._cells

    def refine(self, function=None, recursive=True, cells=None, balance=True, _inRecursion=False):

        if not _inRecursion:
            self._structureChange()
            print 'Refining Mesh'

        cells = cells if cells is not None else sorted(self._cells)
        recurse = []
        tic = time.time()
        for cell in cells:
            p = self._pointer(cell)
            do = function(self._cellC(cell)) > p[-1]
            if do:
                recurse += self._refineCell(cell)

        print '   ', time.time() - tic

        if recursive and len(recurse) > 0:
            recurse += self.refine(function=function, recursive=True, cells=recurse, balance=balance, _inRecursion=True)

        if balance and not _inRecursion:
            self.balance()
        return recurse

    def _refineCell(self, pointer):
        pointer = self._asPointer(pointer)
        ind = self._asIndex(pointer)
        assert ind in self
        h = self._levelWidth(pointer[-1])/2 # halfWidth
        nL = pointer[-1] + 1 # new level
        add = lambda p:p[0]+p[1]
        added = []
        def addCell(p):
            i = self._index(p+[nL])
            self._cells.add(i)
            added.append(i)

        addCell(map(add, zip(pointer[:-1], [0,0,0][:self.dim])))
        addCell(map(add, zip(pointer[:-1], [h,0,0][:self.dim])))
        addCell(map(add, zip(pointer[:-1], [0,h,0][:self.dim])))
        addCell(map(add, zip(pointer[:-1], [h,h,0][:self.dim])))
        if self.dim == 3:
            addCell(map(add, zip(pointer[:-1], [0,0,h])))
            addCell(map(add, zip(pointer[:-1], [h,0,h])))
            addCell(map(add, zip(pointer[:-1], [0,h,h])))
            addCell(map(add, zip(pointer[:-1], [h,h,h])))
        self._cells.remove(ind)
        return added

    def corsen(self, function=None):
        self._structureChange()
        raise Exception('Not yet implemented')

    def _corsenCell(self, pointer):
        raise Exception('Not yet implemented')

    def _asPointer(self, ind):
        if type(ind) in [int, long]:
            return self._pointer(ind)
        if type(ind) is list:
            assert len(ind) == (self.dim + 1), str(ind) +' is not valid pointer'
            assert ind[-1] <= self.levels, str(ind) +' is not valid pointer'
            return ind
        if isinstance(ind, np.ndarray):
            return ind.tolist()
        raise Exception

    def _asIndex(self, pointer):
        if type(pointer) in [int, long]:
            return pointer
        if type(pointer) is list:
            return self._index(pointer)
        raise Exception


    def _childPointers(self, pointer, direction=0, positive=True):
        l = self._levelWidth(pointer[-1] + 1)

        if self.dim == 2:

            children = [
                            [pointer[0]    , pointer[1]    , pointer[-1] + 1],
                            [pointer[0] + l, pointer[1]    , pointer[-1] + 1],
                            [pointer[0]    , pointer[1] + l, pointer[-1] + 1],
                            [pointer[0] + l, pointer[1] + l, pointer[-1] + 1]
                       ]

        elif self.dim == 3:

            children = [
                            [pointer[0]    , pointer[1]    , pointer[2]    , pointer[-1] + 1],
                            [pointer[0] + l, pointer[1]    , pointer[2]    , pointer[-1] + 1],
                            [pointer[0]    , pointer[1] + l, pointer[2]    , pointer[-1] + 1],
                            [pointer[0] + l, pointer[1] + l, pointer[2]    , pointer[-1] + 1],
                            [pointer[0]    , pointer[1]    , pointer[2] + l, pointer[-1] + 1],
                            [pointer[0] + l, pointer[1]    , pointer[2] + l, pointer[-1] + 1],
                            [pointer[0]    , pointer[1] + l, pointer[2] + l, pointer[-1] + 1],
                            [pointer[0] + l, pointer[1] + l, pointer[2] + l, pointer[-1] + 1]
                       ]

        if direction == 0: ind = [0,2,4,6] if not positive else [1,3,5,7]
        if direction == 1: ind = [0,1,4,5] if not positive else [2,3,6,7]
        if direction == 2: ind = [0,1,2,3] if not positive else [4,5,6,7]

        return [children[_] for _ in ind[:(self.dim-1)*2]]


    def _parentPointer(self, pointer):
        mod = self._levelWidth(pointer[-1] - 1)
        return [p - (p % mod) for p in pointer[:-1]] + [pointer[-1]-1]

    def _cellN(self, p):
        p = self._asPointer(p)
        return [hi[:p[ii]].sum() for ii, hi in enumerate(self.h)]

    def _cellH(self, p):
        p = self._asPointer(p)
        w = self._levelWidth(p[-1])
        return [hi[p[ii]:p[ii]+w].sum() for ii, hi in enumerate(self.h)]

    def _cellC(self, p):
        return (np.array(self._cellH(p))/2.0 + self._cellN(p)).tolist()

    def _levelWidth(self, level):
        return 2**(self.levels - level)

    def _isInsideMesh(self, pointer):
        inside = True
        for p in pointer[:-1]:
            inside = inside and p >= 0 and p < 2**self.levels
        return inside

    def _getNextCell(self, ind, direction=0, positive=True, _lookUp=True):
        """
            Returns a None, int, list, or nested list
            The int is the cell number.

        """
        if direction >= self.dim: return None
        pointer = self._asPointer(ind)
        if pointer[-1] > self.levels: return None

        step = (1 if positive else -1) * self._levelWidth(pointer[-1])
        nextCell = [p if ii is not direction else p + step for ii, p in enumerate(pointer)]
        # raise Exception(pointer, nextCell)
        if not self._isInsideMesh(nextCell): return None

        # it might be the same size as me?
        if nextCell in self: return self._index(nextCell)

        if nextCell[-1] + 1 <= self.levels: # if I am not the smallest.
            children  = self._childPointers(pointer, direction=direction, positive=positive)
            nextCells = [self._getNextCell(child, direction=direction, positive=positive, _lookUp=False) for child in children]
            if nextCells[0] is not None:
                return nextCells

        if not _lookUp: return None

        # it might be bigger than me?
        return self._getNextCell(self._parentPointer(pointer),
                direction=direction, positive=positive)

    def balance(self, recursive=True, cells=None, _inRecursion=False):

        tic = time.time()
        if not _inRecursion:
            self._structureChange()
            print 'Balancing Mesh:'

        cells = cells if cells is not None else sorted(self._cells)

        # calcDepth = lambda i: lambda A: i if type(A) is not list else max(map(calcDepth(i+1), A))
        # flatten   = lambda A: A if calcDepth(0)(A) == 1 else flatten([_ for __ in A for _ in (__ if type(__) is list else [__])])

        recurse = set()

        for cell in cells:
            p = self._asPointer(cell)
            if p[-1] == self.levels: continue

            cs = range(6)
            cs[0] = self._getNextCell(cell, direction=0, positive=False)
            cs[1] = self._getNextCell(cell, direction=0, positive=True)
            cs[2] = self._getNextCell(cell, direction=1, positive=False)
            cs[3] = self._getNextCell(cell, direction=1, positive=True)
            cs[4] = self._getNextCell(cell, direction=2, positive=False) # this will be None if in 2D
            cs[5] = self._getNextCell(cell, direction=2, positive=True)  # this will be None if in 2D

            do = np.any([
                        type(c) is list and np.any([type(_) is list for _ in c])
                        for c in cs
                        if c is not None
                   ])
            # depth = calcDepth(0)(cs)
            # print depth, depth > 2, do, [jj for jj in flatten(cs) if jj is not None]
            # recurse += [jj for jj in flatten(cs) if jj is not None]

            if do and cell in self:
                newCells = self._refineCell(cell)
                recurse.update([_ for _ in cs if type(_) in [int, long]]) # only add the bigger ones!
                recurse.update(newCells)

        print '   ', len(cells), time.time() - tic
        if recursive and len(recurse) > 0:
            self.balance(cells=sorted(recurse), _inRecursion=True)

    @property
    def gridCC(self):
        if getattr(self, '_gridCC', None) is None:
            self._gridCC = np.zeros((len(self._cells),self.dim))
            for ii, ind in enumerate(self._sortedCells):
                p = self._asPointer(ind)
                self._gridCC[ii, :] = self._cellC(p)
        return self._gridCC

    @property
    def gridN(self):
        self.number()
        R = self._deflationMatrix(self._nodes, self._hangingN, self._n2i, withHanging=False)
        return R.T * self._gridN

    @property
    def gridFx(self):
        self.number()
        R = self._deflationMatrix(self._facesX, self._hangingFx, self._fx2i, withHanging=False)
        return R.T * self._gridFx

    @property
    def gridFy(self):
        self.number()
        R = self._deflationMatrix(self._facesY, self._hangingFy, self._fy2i, withHanging=False)
        return R.T * self._gridFy

    @property
    def gridFz(self):
        if self.dim < 3: return None
        self.number()
        R = self._deflationMatrix(self._facesZ, self._hangingFz, self._fz2i, withHanging=False)
        return R.T * self._gridFz

    @property
    def gridEx(self):
        if self.dim == 2: return self.gridFy
        self.number()
        R = self._deflationMatrix(self._edgesX, self._hangingEx, self._ex2i, withHanging=False)
        return R.T * self._gridEx

    @property
    def gridEy(self):
        if self.dim == 2: return self.gridFx
        self.number()
        R = self._deflationMatrix(self._edgesY, self._hangingEy, self._ey2i, withHanging=False)
        return R.T * self._gridEy

    @property
    def gridEz(self):
        if self.dim < 3: return None
        self.number()
        R = self._deflationMatrix(self._edgesZ, self._hangingEz, self._ez2i, withHanging=False)
        return R.T * self._gridEz

    @property
    def vol(self):
        if getattr(self, '_vol', None) is None:
            self._vol = np.zeros(len(self._cells))
            for ii, ind in enumerate(self._sortedCells):
                p = self._asPointer(ind)
                self._vol[ii] = np.prod(self._cellH(p))
        return self._vol

    @property
    def area(self):
        self.number()
        if getattr(self, '_area', None) is None:
            Rlist = [0]*self.dim
            Rlist[0] = self._deflationMatrix(self._facesX, self._hangingFx, self._fx2i, withHanging=False)
            Rlist[1] = self._deflationMatrix(self._facesY, self._hangingFy, self._fy2i, withHanging=False)
            if self.dim == 3:
                Rlist[2] = self._deflationMatrix(self._facesZ, self._hangingFz, self._fz2i, withHanging=False)
            R = sp.block_diag(Rlist)
            self._area = R.T * (
                                    np.r_[self._areaFxFull, self._areaFyFull] if self.dim == 2 else
                                    np.r_[self._areaFxFull, self._areaFyFull, self._areaFzFull]
                               )
        return self._area

    @property
    def edge(self):
        self.number()
        if self.dim == 2:
            return np.r_[self._area[self.nFx:], self._area[:self.nFx]]
        if getattr(self, '_edge', None) is None:
            R = sp.block_diag([
                self._deflationMatrix(self._edgesX, self._hangingEx, self._ex2i, withHanging=False),
                self._deflationMatrix(self._edgesY, self._hangingEy, self._ey2i, withHanging=False),
                self._deflationMatrix(self._edgesZ, self._hangingEz, self._ez2i, withHanging=False)
            ])
            self._edge = R.T * np.r_[self._edgeExFull, self._edgeEyFull, self._edgeEzFull]

        return self._edge

    def _onSameLevel(self, i0, i1):
        p0 = self._asPointer(i0)
        p1 = self._asPointer(i1)
        return p0[-1] == p1[-1]

    def _numberNodes(self, force=False):
        if not self.__dirtyNodes__ and not force: return

        self._nodes = set()

        for ind in self._cells:
            p = self._asPointer(ind)
            w = self._levelWidth(p[-1])
            if self.dim == 2:
                self._nodes.add(self._index([p[0]    , p[1]    , p[2]]))
                self._nodes.add(self._index([p[0] + w, p[1]    , p[2]]))
                self._nodes.add(self._index([p[0]    , p[1] + w, p[2]]))
                self._nodes.add(self._index([p[0] + w, p[1] + w, p[2]]))
            elif self.dim == 3:
                self._nodes.add(self._index([p[0]    , p[1]    , p[2]    , p[3]]))
                self._nodes.add(self._index([p[0] + w, p[1]    , p[2]    , p[3]]))
                self._nodes.add(self._index([p[0]    , p[1] + w, p[2]    , p[3]]))
                self._nodes.add(self._index([p[0] + w, p[1] + w, p[2]    , p[3]]))
                self._nodes.add(self._index([p[0]    , p[1]    , p[2] + w, p[3]]))
                self._nodes.add(self._index([p[0] + w, p[1]    , p[2] + w, p[3]]))
                self._nodes.add(self._index([p[0]    , p[1] + w, p[2] + w, p[3]]))
                self._nodes.add(self._index([p[0] + w, p[1] + w, p[2] + w, p[3]]))
        gridN = []
        self._n2i = dict()
        for ii, n in enumerate(sorted(self._nodes)):
            self._n2i[n] = ii
            gridN.append( self._cellN( self._pointer(n) ) )
        self._gridN = np.array(gridN)

        self.__dirtyNodes__ = False

    def _numberFaces(self, force=False):
        if not self.__dirtyFaces__ and not force: return

        self._facesX = set()
        self._facesY = set()
        if self.dim == 3:
            self._facesZ = set()

        for ind in self._cells:
            p = self._asPointer(ind)
            w = self._levelWidth(p[-1])

            if self.dim == 2:
                self._facesX.add(self._index([p[0]    , p[1]    , p[2]]))
                self._facesX.add(self._index([p[0] + w, p[1]    , p[2]]))
                self._facesY.add(self._index([p[0]    , p[1]    , p[2]]))
                self._facesY.add(self._index([p[0]    , p[1] + w, p[2]]))
            elif self.dim == 3:
                self._facesX.add(self._index([p[0]    , p[1]    , p[2]    , p[3]]))
                self._facesX.add(self._index([p[0] + w, p[1]    , p[2]    , p[3]]))
                self._facesY.add(self._index([p[0]    , p[1]    , p[2]    , p[3]]))
                self._facesY.add(self._index([p[0]    , p[1] + w, p[2]    , p[3]]))
                self._facesZ.add(self._index([p[0]    , p[1]    , p[2]    , p[3]]))
                self._facesZ.add(self._index([p[0]    , p[1]    , p[2] + w, p[3]]))

        gridFx = []
        areaFx = []
        self._fx2i = dict()
        for ii, fx in enumerate(sorted(self._facesX)):
            self._fx2i[fx] = ii
            p = self._pointer(fx)
            n, h = self._cellN(p), self._cellH(p)
            if self.dim == 2:
                gridFx.append( [n[0], n[1] + h[1]/2.0] )
                areaFx.append( h[1] )
            elif self.dim == 3:
                gridFx.append( [n[0], n[1] + h[1]/2.0, n[2] + h[2]/2.0] )
                areaFx.append( h[1]*h[2] )
        self._gridFx = np.array(gridFx)
        self._areaFxFull = np.array(areaFx)

        gridFy = []
        areaFy = []
        self._fy2i = dict()
        for ii, fy in enumerate(sorted(self._facesY)):
            self._fy2i[fy] = ii
            p = self._pointer(fy)
            n, h = self._cellN(p), self._cellH(p)
            if self.dim == 2:
                gridFy.append( [n[0] + h[0]/2.0, n[1]] )
                areaFy.append( h[0] )
            elif self.dim == 3:
                gridFy.append( [n[0] + h[0]/2.0, n[1], n[2] + h[2]/2.0] )
                areaFy.append( h[0]*h[2] )
        self._gridFy = np.array(gridFy)
        self._areaFyFull = np.array(areaFy)

        if self.dim == 2:
            self.__dirtyFaces__ = False
            return

        gridFz = []
        areaFz = []
        self._fz2i = dict()
        for ii, fz in enumerate(sorted(self._facesZ)):
            self._fz2i[fz] = ii
            p = self._pointer(fz)
            n, h = self._cellN(p), self._cellH(p)
            gridFz.append( [n[0] + h[0]/2.0, n[1] + h[1]/2.0, n[2]] )
            areaFz.append(h[0]*h[1])
        self._gridFz = np.array(gridFz)
        self._areaFzFull = np.array(areaFz)

        self.__dirtyFaces__ = False

    def _numberEdges(self, force=False):
        if self.dim == 2: return
        if not self.__dirtyEdges__ and not force: return

        self._edgesX = set()
        self._edgesY = set()
        self._edgesZ = set()

        for ind in self._cells:
            p = self._asPointer(ind)
            w = self._levelWidth(p[-1])
            self._edgesX.add(self._index([p[0]    , p[1]    , p[2]    , p[3]]))
            self._edgesX.add(self._index([p[0]    , p[1] + w, p[2]    , p[3]]))
            self._edgesX.add(self._index([p[0]    , p[1]    , p[2] + w, p[3]]))
            self._edgesX.add(self._index([p[0]    , p[1] + w, p[2] + w, p[3]]))

            self._edgesY.add(self._index([p[0]    , p[1]    , p[2]    , p[3]]))
            self._edgesY.add(self._index([p[0] + w, p[1]    , p[2]    , p[3]]))
            self._edgesY.add(self._index([p[0]    , p[1]    , p[2] + w, p[3]]))
            self._edgesY.add(self._index([p[0] + w, p[1]    , p[2] + w, p[3]]))

            self._edgesZ.add(self._index([p[0]    , p[1]    , p[2]    , p[3]]))
            self._edgesZ.add(self._index([p[0] + w, p[1]    , p[2]    , p[3]]))
            self._edgesZ.add(self._index([p[0]    , p[1] + w, p[2]    , p[3]]))
            self._edgesZ.add(self._index([p[0] + w, p[1] + w, p[2]    , p[3]]))

        gridEx = []
        edgeEx = []
        self._ex2i = dict()
        for ii, ex in enumerate(sorted(self._edgesX)):
            self._ex2i[ex] = ii
            p = self._pointer(ex)
            n, h = self._cellN(p), self._cellH(p)
            gridEx.append( [n[0] + h[0]/2.0, n[1], n[2]] )
            edgeEx.append( h[0] )
        self._gridEx = np.array(gridEx)
        self._edgeExFull = np.array(edgeEx)

        gridEy = []
        edgeEy = []
        self._ey2i = dict()
        for ii, ey in enumerate(sorted(self._edgesY)):
            self._ey2i[ey] = ii
            p = self._pointer(ey)
            n, h = self._cellN(p), self._cellH(p)
            gridEy.append( [n[0], n[1] + h[1]/2.0, n[2]] )
            edgeEy.append( h[1] )
        self._gridEy = np.array(gridEy)
        self._edgeEyFull = np.array(edgeEx)

        gridEz = []
        edgeEz = []
        self._ez2i = dict()
        for ii, ez in enumerate(sorted(self._edgesZ)):
            self._ez2i[ez] = ii
            p = self._pointer(ez)
            n, h = self._cellN(p), self._cellH(p)
            gridEz.append( [n[0], n[1], n[2] + h[2]/2.0] )
            edgeEz.append( h[2] )
        self._gridEz = np.array(gridEz)
        self._edgeEzFull = np.array(edgeEx)

        self.__dirtyEdges__ = False

    def _hanging(self, force=False):
        if not self.__dirtyHanging__ and not force: return

        self._numberNodes()
        self._numberFaces()
        self._numberEdges()

        self._hangingN  = dict()
        self._hangingFx = dict()
        self._hangingFy = dict()
        if self.dim == 3:
            self._hangingFz = dict()
            self._hangingEx = dict()
            self._hangingEy = dict()
            self._hangingEz = dict()

        # Compute from x faces
        for fx in self._facesX:
            p = self._pointer(fx)
            if p[-1] + 1 > self.levels: continue
            sl = p[-1] + 1 #: small level
            test = self._index(p[:-1] + [sl])
            if test not in self._facesX:
                # Return early without checking the other faces
                continue
            w = self._levelWidth(sl)

            if self.dim == 2:
                self._hangingFx[self._fx2i[test                                 ]] = ([self._fx2i[fx], 0.5], )
                self._hangingFx[self._fx2i[self._index([p[0]    , p[1] + w, sl])]] = ([self._fx2i[fx], 0.5], )

                n0, n1 = fx, self._index([p[0], p[1] + 2*w, p[-1]])
                self._hangingN[self._n2i[test                                   ]] = ([self._n2i[n0], 1.0], )
                self._hangingN[self._n2i[self._index([p[0]    , p[1] +   w, sl])]] = ([self._n2i[n0], 0.5], [self._n2i[n1], 0.5])
                self._hangingN[self._n2i[self._index([p[0]    , p[1] + 2*w, sl])]] = ([self._n2i[n1], 1.0], )

            elif self.dim == 3:
                ey0 = fx
                ey1 = self._index([p[0], p[1]      , p[2] + 2*w, p[-1]])
                ez0 = fx
                ez1 = self._index([p[0], p[1] + 2*w, p[2]      , p[-1]])

                n0  = fx
                n1  = self._index([p[0], p[1] + 2*w, p[2]      , p[-1]])
                n2  = self._index([p[0], p[1]      , p[2] + 2*w, p[-1]])
                n3  = self._index([p[0], p[1] + 2*w, p[2] + 2*w, p[-1]])

                self._hangingFx[self._fx2i[test                                           ]] = ([self._fx2i[fx], 0.25], )
                self._hangingFx[self._fx2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._fx2i[fx], 0.25], )
                self._hangingFx[self._fx2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._fx2i[fx], 0.25], )
                self._hangingFx[self._fx2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._fx2i[fx], 0.25], )

                self._hangingEy[self._ey2i[test                                           ]] = ([self._ey2i[ey0], 0.5], )
                self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._ey2i[ey0], 0.5], )
                self._hangingEy[self._ey2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._ey2i[ey0], 0.25], [self._ey2i[ey1], 0.25])
                self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._ey2i[ey0], 0.25], [self._ey2i[ey1], 0.25])
                self._hangingEy[self._ey2i[self._index([p[0], p[1]      , p[2] + 2*w, sl])]] = ([self._ey2i[ey1], 0.5], )
                self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2] + 2*w, sl])]] = ([self._ey2i[ey1], 0.5], )

                self._hangingEz[self._ez2i[test                                           ]] = ([self._ez2i[ez0], 0.5], )
                self._hangingEz[self._ez2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._ez2i[ez0], 0.5], )
                self._hangingEz[self._ez2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._ez2i[ez0], 0.25], [self._ez2i[ez1], 0.25])
                self._hangingEz[self._ez2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._ez2i[ez0], 0.25], [self._ez2i[ez1], 0.25])
                self._hangingEz[self._ez2i[self._index([p[0], p[1] + 2*w, p[2]      , sl])]] = ([self._ez2i[ez1], 0.5], )
                self._hangingEz[self._ez2i[self._index([p[0], p[1] + 2*w, p[2] +   w, sl])]] = ([self._ez2i[ez1], 0.5], )

                self._hangingN[ self._n2i[ test                                           ]] = ([self._n2i[n0],   1.0], )
                self._hangingN[ self._n2i[ self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._n2i[n0],   0.5], [self._n2i[n1], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0], p[1] + 2*w, p[2]      , sl])]] = ([self._n2i[n1],   1.0], )
                self._hangingN[ self._n2i[ self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._n2i[n0],   0.5], [self._n2i[n2], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._n2i[n0],   0.25], [self._n2i[n1], 0.25], [self._n2i[n2], 0.25], [self._n2i[n3], 0.25])
                self._hangingN[ self._n2i[ self._index([p[0], p[1] + 2*w, p[2] +   w, sl])]] = ([self._n2i[n1],   0.5], [self._n2i[n3], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0], p[1]      , p[2] + 2*w, sl])]] = ([self._n2i[n2],   1.0], )
                self._hangingN[ self._n2i[ self._index([p[0], p[1] +   w, p[2] + 2*w, sl])]] = ([self._n2i[n2],   0.5], [self._n2i[n3], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0], p[1] + 2*w, p[2] + 2*w, sl])]] = ([self._n2i[n3],   1.0], )

        # Compute from y faces
        for fy in self._facesY:
            p = self._pointer(fy)
            if p[-1] + 1 > self.levels: continue
            sl = p[-1] + 1 #: small level
            test = self._index(p[:-1] + [sl])
            if test not in self._facesY:
                # Return early without checking the other faces
                continue
            w = self._levelWidth(sl)

            if self.dim == 2:
                self._hangingFy[self._fy2i[test                                 ]] = ([self._fy2i[fy], 0.5], )
                self._hangingFy[self._fy2i[self._index([p[0] + w, p[1]    , sl])]] = ([self._fy2i[fy], 0.5], )

                n0, n1 = fy, self._index([p[0] + 2*w, p[1], p[-1]])
                self._hangingN[self._n2i[test                                   ]] = ([self._n2i[n0], 1.0], )
                self._hangingN[self._n2i[self._index([p[0] +   w, p[1]    , sl])]] = ([self._n2i[n0], 0.5], [self._n2i[n1], 0.5])
                self._hangingN[self._n2i[self._index([p[0] + 2*w, p[1]    , sl])]] = ([self._n2i[n1], 1.0], )

            elif self.dim == 3:
                ex0 = fy
                ex1 = self._index([p[0]      , p[1], p[2] + 2*w, p[-1]])
                ez0 = fy
                ez1 = self._index([p[0] + 2*w, p[1], p[2]      , p[-1]])

                n0  = fy
                n1  = self._index([p[0] + 2*w, p[1], p[2]      , p[-1]])
                n2  = self._index([p[0]      , p[1], p[2] + 2*w, p[-1]])
                n3  = self._index([p[0] + 2*w, p[1], p[2] + 2*w, p[-1]])

                self._hangingFy[self._fy2i[test                                           ]] = ([self._fy2i[fy], 0.25], )
                self._hangingFy[self._fy2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._fy2i[fy], 0.25], )
                self._hangingFy[self._fy2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._fy2i[fy], 0.25], )
                self._hangingFy[self._fy2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._fy2i[fy], 0.25], )

                self._hangingEx[self._ex2i[test                                           ]] = ([self._ex2i[ex0], 0.5], )
                self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._ex2i[ex0], 0.5], )
                self._hangingEx[self._ex2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._ex2i[ex0], 0.25], [self._ex2i[ex1], 0.25])
                self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._ex2i[ex0], 0.25], [self._ex2i[ex1], 0.25])
                self._hangingEx[self._ex2i[self._index([p[0]      , p[1], p[2] + 2*w, sl])]] = ([self._ex2i[ex1], 0.5], )
                self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2] + 2*w, sl])]] = ([self._ex2i[ex1], 0.5], )

                self._hangingEz[self._ez2i[test                                           ]] = ([self._ez2i[ez0], 0.5], )
                self._hangingEz[self._ez2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._ez2i[ez0], 0.5], )
                self._hangingEz[self._ez2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._ez2i[ez0], 0.25], [self._ez2i[ez1], 0.25])
                self._hangingEz[self._ez2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._ez2i[ez0], 0.25], [self._ez2i[ez1], 0.25])
                self._hangingEz[self._ez2i[self._index([p[0] + 2*w, p[1], p[2]      , sl])]] = ([self._ez2i[ez1], 0.5], )
                self._hangingEz[self._ez2i[self._index([p[0] + 2*w, p[1], p[2] +   w, sl])]] = ([self._ez2i[ez1], 0.5], )

                self._hangingN[ self._n2i[ test                                           ]] = ([self._n2i[n0],   1.0], )
                self._hangingN[ self._n2i[ self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._n2i[n0],   0.5], [self._n2i[n1], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0] + 2*w, p[1], p[2]      , sl])]] = ([self._n2i[n1],   1.0], )
                self._hangingN[ self._n2i[ self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._n2i[n0],   0.5], [self._n2i[n2], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._n2i[n0],   0.25], [self._n2i[n1], 0.25], [self._n2i[n2], 0.25], [self._n2i[n3], 0.25])
                self._hangingN[ self._n2i[ self._index([p[0] + 2*w, p[1], p[2] +   w, sl])]] = ([self._n2i[n1],   0.5], [self._n2i[n3], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0]      , p[1], p[2] + 2*w, sl])]] = ([self._n2i[n2],   1.0], )
                self._hangingN[ self._n2i[ self._index([p[0] +   w, p[1], p[2] + 2*w, sl])]] = ([self._n2i[n2],   0.5], [self._n2i[n3], 0.5])
                self._hangingN[ self._n2i[ self._index([p[0] + 2*w, p[1], p[2] + 2*w, sl])]] = ([self._n2i[n3],   1.0], )

        if self.dim == 2:
            self.__dirtyHanging__ = False
            return

        # Compute from z faces
        for fz in self._facesZ:
            p = self._pointer(fz)
            if p[-1] + 1 > self.levels: continue
            sl = p[-1] + 1 #: small level
            test = self._index(p[:-1] + [sl])
            if test not in self._facesZ:
                # Return early without checking the other faces
                continue
            w = self._levelWidth(sl)

            ex0 = fz
            ex1 = self._index([p[0]      , p[1] + 2*w, p[2], p[-1]])
            ey0 = fz
            ey1 = self._index([p[0] + 2*w, p[1]      , p[2], p[-1]])

            n0  = fz
            n1  = self._index([p[0] + 2*w, p[1]      , p[2], p[-1]])
            n2  = self._index([p[0]      , p[1] + 2*w, p[2], p[-1]])
            n3  = self._index([p[0] + 2*w, p[1] + 2*w, p[2], p[-1]])

            self._hangingFz[self._fz2i[test                                           ]] = ([self._fz2i[fz], 0.25], )
            self._hangingFz[self._fz2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._fz2i[fz], 0.25], )
            self._hangingFz[self._fz2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._fz2i[fz], 0.25], )
            self._hangingFz[self._fz2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._fz2i[fz], 0.25], )

            self._hangingEx[self._ex2i[test                                           ]] = ([self._ex2i[ex0], 0.5], )
            self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._ex2i[ex0], 0.5], )
            self._hangingEx[self._ex2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._ex2i[ex0], 0.25], [self._ex2i[ex1], 0.25])
            self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._ex2i[ex0], 0.25], [self._ex2i[ex1], 0.25])
            self._hangingEx[self._ex2i[self._index([p[0]      , p[1] + 2*w, p[2], sl])]] = ([self._ex2i[ex1], 0.5], )
            self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1] + 2*w, p[2], sl])]] = ([self._ex2i[ex1], 0.5], )

            self._hangingEy[self._ey2i[test                                           ]] = ([self._ey2i[ey0], 0.5], )
            self._hangingEy[self._ey2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._ey2i[ey0], 0.5], )
            self._hangingEy[self._ey2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._ey2i[ey0], 0.25], [self._ey2i[ey1], 0.25])
            self._hangingEy[self._ey2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._ey2i[ey0], 0.25], [self._ey2i[ey1], 0.25])
            self._hangingEy[self._ey2i[self._index([p[0] + 2*w, p[1]      , p[2], sl])]] = ([self._ey2i[ey1], 0.5], )
            self._hangingEy[self._ey2i[self._index([p[0] + 2*w, p[1] +   w, p[2], sl])]] = ([self._ey2i[ey1], 0.5], )

            self._hangingN[ self._n2i[ test                                           ]] = ([self._n2i[n0],   1.0], )
            self._hangingN[ self._n2i[ self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._n2i[n0],   0.5], [self._n2i[n1], 0.5])
            self._hangingN[ self._n2i[ self._index([p[0] + 2*w, p[1]      , p[2], sl])]] = ([self._n2i[n1],   1.0], )
            self._hangingN[ self._n2i[ self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._n2i[n0],   0.5], [self._n2i[n2], 0.5])
            self._hangingN[ self._n2i[ self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._n2i[n0],   0.25], [self._n2i[n1], 0.25], [self._n2i[n2], 0.25], [self._n2i[n3], 0.25])
            self._hangingN[ self._n2i[ self._index([p[0] + 2*w, p[1] +   w, p[2], sl])]] = ([self._n2i[n1],   0.5], [self._n2i[n3], 0.5])
            self._hangingN[ self._n2i[ self._index([p[0]      , p[1] + 2*w, p[2], sl])]] = ([self._n2i[n2],   1.0], )
            self._hangingN[ self._n2i[ self._index([p[0] +   w, p[1] + 2*w, p[2], sl])]] = ([self._n2i[n2],   0.5], [self._n2i[n3], 0.5])
            self._hangingN[ self._n2i[ self._index([p[0] + 2*w, p[1] + 2*w, p[2], sl])]] = ([self._n2i[n3],   1.0], )

        self.__dirtyHanging__ = False

    def number(self, force=False):
        if not self.__dirty__ and not force: return
        self._hanging()
        return

    def _deflationMatrix(self, theSet, theHang, theIndex, withHanging=True):
        reducedInd = dict() # final reduced index
        ii = 0
        I,J,V = [],[],[]
        for fx in sorted(theSet):
            if theIndex[fx] not in theHang:
                reducedInd[theIndex[fx]] = ii
                I += [theIndex[fx]]
                J += [ii]
                V += [1.0]
                ii += 1
        if withHanging:
            for hfkey in theHang.keys():
                hf = theHang[hfkey]
                I += [hfkey]*len(hf)
                J += [reducedInd[_[0]] for _ in hf]
                V += [_[1] for _ in hf]
        return sp.csr_matrix((V,(I,J)), shape=(len(theSet), len(reducedInd)))

    @property
    def faceDiv(self):
        if getattr(self, '_faceDiv', None) is None:
            self.number()

            # TODO: Preallocate!
            I, J, V = [], [], []
            PM = [-1,1]*self.dim # plus / minus
            offset = [0,0,self.nFx,self.nFx,self.nFx+self.nFy,self.nFx+self.nFy]

            for ii, ind in enumerate(self._sortedCells):

                p = self._pointer(ind)
                w = self._levelWidth(p[-1])

                if self.dim == 2:
                    faces = [
                                self._fx2i[self._index([ p[0]    , p[1]    , p[2]])],
                                self._fx2i[self._index([ p[0] + w, p[1]    , p[2]])],
                                self._fy2i[self._index([ p[0]    , p[1]    , p[2]])],
                                self._fy2i[self._index([ p[0]    , p[1] + w, p[2]])]
                            ]
                elif self.dim == 3:
                    faces = [
                                self._fx2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._fx2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
                                self._fy2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._fy2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
                                self._fz2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._fz2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])]
                            ]

                for off, pm, face in zip(offset,PM,faces):
                    I += [ii]
                    J += [face + off]
                    V += [pm]

            # total number of faces
            tnF = len(self._facesX) + len(self._facesY) + (0 if self.dim == 2 else len(self._facesZ))

            D = sp.csr_matrix((V,(I,J)), shape=(self.nC, tnF))
            Rlist = [0]*self.dim
            Rlist[0] = self._deflationMatrix(self._facesX, self._hangingFx, self._fx2i)
            Rlist[1] = self._deflationMatrix(self._facesY, self._hangingFy, self._fy2i)
            if self.dim == 3:
                Rlist[2] = self._deflationMatrix(self._facesZ, self._hangingFz, self._fz2i)
            R = sp.block_diag(Rlist)
            VOL = self.vol
            S = self.area
            self._faceDiv = Utils.sdiag(1.0/VOL)*D*R*Utils.sdiag(S)
        return self._faceDiv

    @property
    def edgeCurl(self):
        """Construct the 3D curl operator."""
        assert self.dim > 2, "Edge Curl only programed for 3D."

        if getattr(self, '_edgeCurl', None) is None:
            self.number()
            # TODO: Preallocate!
            I, J, V = [], [], []
            faceOffset = 0
            offset = [self.nEx]*2 + [self.nEx+self.nEy]*2
            PM = [1, -1, -1, 1]
            for ii, fx in  enumerate(sorted(self._facesX)):

                p = self._pointer(fx)
                w = self._levelWidth(p[-1])

                edges = [
                            self._ey2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                            self._ey2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])],
                            self._ez2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                            self._ez2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
                        ]

                for off, pm, edge in zip(offset,PM,edges):
                    I += [ii + faceOffset]
                    J += [edge + off]
                    V += [pm]

            faceOffset = self.nFx
            offset = [0]*2 + [self.nEx+self.nEy]*2
            PM = [-1, 1, 1, -1]
            for ii, fy in  enumerate(sorted(self._facesY)):

                p = self._pointer(fy)
                w = self._levelWidth(p[-1])

                edges = [
                            self._ex2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                            self._ex2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])],
                            self._ez2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                            self._ez2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
                        ]

                for off, pm, edge in zip(offset,PM,edges):
                    I += [ii + faceOffset]
                    J += [edge + off]
                    V += [pm]

            faceOffset = self.nFx + self.nFy
            offset = [0]*2 + [self.nEx]*2
            PM = [1, -1, -1, 1]
            for ii, fz in  enumerate(sorted(self._facesZ)):

                p = self._pointer(fz)
                w = self._levelWidth(p[-1])

                edges = [
                            self._ex2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                            self._ex2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
                            self._ey2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                            self._ey2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
                        ]

                for off, pm, edge in zip(offset,PM,edges):
                    I += [ii + faceOffset]
                    J += [edge + off]
                    V += [pm]

            tnF = len(self._facesX) + len(self._facesY) + len(self._facesZ)
            tnE = len(self._edgesX) + len(self._edgesY) + len(self._edgesZ)

            Rf = sp.block_diag([
                        self._deflationMatrix(self._facesX, self._hangingFx, self._fx2i),
                        self._deflationMatrix(self._facesY, self._hangingFy, self._fy2i),
                        self._deflationMatrix(self._facesZ, self._hangingFz, self._fz2i)
                    ])

            Re = sp.block_diag([
                        self._deflationMatrix(self._edgesX, self._hangingEx, self._ex2i),
                        self._deflationMatrix(self._edgesY, self._hangingEy, self._ey2i),
                        self._deflationMatrix(self._edgesZ, self._hangingEz, self._ez2i)
                    ])

            C = sp.csr_matrix((V,(I,J)), shape=(tnF, tnE))
            S = self.area
            L = self.edge
            self._edgeCurl = Utils.sdiag(1/S)*Rf.T*C*Re*Utils.sdiag(L)
        return self._edgeCurl


    def plotGrid(self, ax=None, showIt=False,
                grid=True,
                cells=True, cellLine=False,
                nodes=False,
                facesX=False, facesY=False, facesZ=False,
                edgesX=False, edgesY=False, edgesZ=False):

        # self.number()

        axOpts = {'projection':'3d'} if self.dim == 3 else {}
        if ax is None:
            ax = plt.subplot(111, **axOpts)
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure

        if grid:
            for ind in self._sortedCells:
                p = self._asPointer(ind)
                n = self._cellN(p)
                h = self._cellH(p)
                x = [n[0]    , n[0] + h[0], n[0] + h[0], n[0]       , n[0]]
                y = [n[1]    , n[1]       , n[1] + h[1], n[1] + h[1], n[1]]
                if self.dim == 2:
                    ax.plot(x,y, 'b-')
                elif self.dim == 3:
                    ax.plot(x,y, 'b-', zs=[n[2]]*5)
                    z = [n[2] + h[2], n[2] + h[2], n[2] + h[2], n[2] + h[2], n[2] + h[2]]
                    ax.plot(x,y, 'b-', zs=z)
                    sides = [0,0], [h[0],0], [0,h[1]], [h[0],h[1]]
                    for s in sides:
                        x = [n[0] + s[0], n[0] + s[0]]
                        y = [n[1] + s[1], n[1] + s[1]]
                        z = [n[2]       , n[2] + h[2]]
                        ax.plot(x,y, 'b-', zs=z)

        if self.dim == 2:
            if cells:
                ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r.')
            if cellLine:
                ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r:')
                ax.plot(self.gridCC[[0,-1],0], self.gridCC[[0,-1],1], 'ro')
            if nodes:
                ax.plot(self.gridN[:,0], self.gridN[:,1], 'ms')
                ax.plot(self.gridN[self._hangingN.keys(),0], self.gridN[self._hangingN.keys(),1], 'ms', ms=10, mfc='none', mec='m')
            if facesX:
                ax.plot(self.gridFx[self._hangingFx.keys(),0], self.gridFx[self._hangingFx.keys(),1], 'gs', ms=10, mfc='none', mec='g')
                ax.plot(self.gridFx[:,0], self.gridFx[:,1], 'g>')
            if facesY:
                ax.plot(self.gridFy[self._hangingFy.keys(),0], self.gridFy[self._hangingFy.keys(),1], 'gs', ms=10, mfc='none', mec='g')
                ax.plot(self.gridFy[:,0], self.gridFy[:,1], 'g^')
        elif self.dim == 3:
            if cells:
                ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r.', zs=self.gridCC[:,2])
            if cellLine:
                ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r:', zs=self.gridCC[:,2])
                ax.plot(self.gridCC[[0,-1],0], self.gridCC[[0,-1],1], 'ro', zs=self.gridCC[[0,-1],2])

            if nodes:
                ax.plot(self.gridN[:,0], self.gridN[:,1], 'ms', zs=self.gridN[:,2])
                ax.plot(self.gridN[self._hangingN.keys(),0], self.gridN[self._hangingN.keys(),1], 'ms', ms=10, mfc='none', mec='m', zs=self.gridN[self._hangingN.keys(),2])
                for key in self._hangingN.keys():
                    for hf in self._hangingN[key]:
                        ind = [key, hf[0]]
                        ax.plot(self.gridN[ind,0], self.gridN[ind,1], 'm:', zs=self.gridN[ind,2])

            if facesX:
                ax.plot(self.gridFx[:,0], self.gridFx[:,1], 'g>', zs=self.gridFx[:,2])
                ax.plot(self.gridFx[self._hangingFx.keys(),0], self.gridFx[self._hangingFx.keys(),1], 'gs', ms=10, mfc='none', mec='g', zs=self.gridFx[self._hangingFx.keys(),2])
                for key in self._hangingFx.keys():
                    for hf in self._hangingFx[key]:
                        ind = [key, hf[0]]
                        ax.plot(self.gridFx[ind,0], self.gridFx[ind,1], 'g:', zs=self.gridFx[ind,2])

            if facesY:
                ax.plot(self.gridFy[:,0], self.gridFy[:,1], 'g^', zs=self.gridFy[:,2])
                ax.plot(self.gridFy[self._hangingFy.keys(),0], self.gridFy[self._hangingFy.keys(),1], 'gs', ms=10, mfc='none', mec='g', zs=self.gridFy[self._hangingFy.keys(),2])
                for key in self._hangingFy.keys():
                    for hf in self._hangingFy[key]:
                        ind = [key, hf[0]]
                        ax.plot(self.gridFy[ind,0], self.gridFy[ind,1], 'g:', zs=self.gridFy[ind,2])

            if facesZ:
                ax.plot(self.gridFz[:,0], self.gridFz[:,1], 'g^', zs=self.gridFz[:,2])
                ax.plot(self.gridFz[self._hangingFz.keys(),0], self.gridFz[self._hangingFz.keys(),1], 'gs', ms=10, mfc='none', mec='g', zs=self.gridFz[self._hangingFz.keys(),2])
                for key in self._hangingFz.keys():
                    for hf in self._hangingFz[key]:
                        ind = [key, hf[0]]
                        ax.plot(self.gridFz[ind,0], self.gridFz[ind,1], 'g:', zs=self.gridFz[ind,2])

            if edgesX:
                ax.plot(self.gridEx[:,0], self.gridEx[:,1], 'k>', zs=self.gridEx[:,2])
                ax.plot(self.gridEx[self._hangingEx.keys(),0], self.gridEx[self._hangingEx.keys(),1], 'ks', ms=10, mfc='none', mec='k', zs=self.gridEx[self._hangingEx.keys(),2])
                for key in self._hangingEx.keys():
                    for hf in self._hangingEx[key]:
                        ind = [key, hf[0]]
                        ax.plot(self.gridEx[ind,0], self.gridEx[ind,1], 'k:', zs=self.gridEx[ind,2])


            if edgesY:
                ax.plot(self.gridEy[:,0], self.gridEy[:,1], 'k<', zs=self.gridEy[:,2])
                ax.plot(self.gridEy[self._hangingEy.keys(),0], self.gridEy[self._hangingEy.keys(),1], 'ks', ms=10, mfc='none', mec='k', zs=self.gridEy[self._hangingEy.keys(),2])
                for key in self._hangingEy.keys():
                    for hf in self._hangingEy[key]:
                        ind = [key, hf[0]]
                        ax.plot(self.gridEy[ind,0], self.gridEy[ind,1], 'k:', zs=self.gridEy[ind,2])

            if edgesZ:
                ax.plot(self.gridEz[:,0], self.gridEz[:,1], 'k^', zs=self.gridEz[:,2])
                ax.plot(self.gridEz[self._hangingEz.keys(),0], self.gridEz[self._hangingEz.keys(),1], 'ks', ms=10, mfc='none', mec='k', zs=self.gridEz[self._hangingEz.keys(),2])
                for key in self._hangingEz.keys():
                    for hf in self._hangingEz[key]:
                        ind = [key, hf[0]]
                        ax.plot(self.gridEz[ind,0], self.gridEz[ind,1], 'k:', zs=self.gridEz[ind,2])


        # ax.axis('equal')
        if showIt:plt.show()



if __name__ == '__main__':


    def function(xc):
        r = xc - np.array([0.5*128]*len(xc))
        dist = np.sqrt(r.dot(r))
        # if dist < 0.05:
        #     return 5
        if dist < 0.1*128:
            return 7
        if dist < 0.3*128:
            return 5
        if dist < 1.0*128:
            return 2
        else:
            return 0

    T = Tree([[(1,128)],[(1,128)],[(1,128)]],levels=7)
    T = Tree([128,128,128],levels=7)
    # T = Tree([[(1,16)],[(1,16)]],levels=4)
    # T = Tree([[(1,128)],[(1,128)]],levels=7)
    T.refine(lambda xc:1, balance=False)
    # T._index([0,0,0])
    # T._pointer(0)


    tic = time.time()
    # T.refine(function)#, balance=False)
    print time.time() - tic
    print T.nC

    print T.gridFz


    # T._refineCell([8,0,1])
    # T._refineCell([8,0,2])
    # T._refineCell([12,0,2])
    # T._refineCell([8,4,2])
    # T._refineCell([6,0,3])
    # T._refineCell([8,8,1])
    T._refineCell([0,0,0,1])
    T.__dirty__ = True


    print T.gridFx.shape[0], T.nFx


    ax = plt.subplot(211)
    ax.spy(T.edgeCurl)

    # print Mesh.TensorMesh([2,2,2]).edgeCurl.todense()
    # print T.edgeCurl.todense()
    # print Mesh.TensorMesh([2,2,2]).edgeCurl.todense() - T.edgeCurl.todense()
    # print T.gridEy - Mesh.TensorMesh([2,2,2]).gridEy

    # print T.edge
    # T.plotGrid(ax=ax)

    # R = deflationMatrix(T._facesX, T._hangingFx, T._fx2i)
    # print R

    ax = plt.subplot(212)#, projection='3d')
    ax.spy(Mesh.TensorMesh([2,2,2]).edgeCurl)

    # ax = plt.subplot(313)
    # ax.spy(T.faceDiv[:,:T.nFx] * R)


    # T.balance()
    # T.plotGrid(ax=ax)

    # cx = T._getNextCell([0,0,1],direction=0,positive=True)
    # print cx
    # # print [T._asPointer(_) for _ in cx]
    # cx = T._getNextCell([8,0,3],direction=0,positive=False)
    # print T._asPointer(cx)
    # cx = T._getNextCell([8,8,1],direction=1,positive=False)
    # print cx, #[T._asPointer(_) for _ in cx]
    # cm = T._getNextCell([64,80,4],direction=0,positive=False)
    # cy = T._getNextCell([64,80,4],direction=1,positive=True)
    # cp = T._getNextCell([64,80,4],direction=1,positive=False)

    # ax.plot( T._cellN([4,0,1])[0],T._cellN([4,0,1])[1], 'yd')
    # ax.plot( T._cellN(cx)[0],T._cellN(cx)[1], 'ys')
    # ax.plot( T._cellN(cm)[0],T._cellN(cm)[1], 'ys')
    # ax.plot( T._cellN(cy)[0],T._cellN(cy)[1], 'ys')
    # ax.plot( T._cellN(cp[0])[0],T._cellN(cp[0])[1], 'ys')
    # ax.plot( T._cellN(cp[1])[0],T._cellN(cp[1])[1], 'ys')





    # print T.nN

    plt.show()

