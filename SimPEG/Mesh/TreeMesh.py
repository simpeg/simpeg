#      ___                     ___          ___          ___          ___
#     /\  \         ___       /\__\        /\  \        /\  \        /\  \
#    /::\  \       /\  \     /::|  |      /::\  \      /::\  \      /::\  \
#   /:/\ \  \      \:\  \   /:|:|  |     /:/\:\  \    /:/\:\  \    /:/\:\  \
#  _\:\~\ \  \     /::\__\ /:/|:|__|__  /::\~\:\  \  /::\~\:\  \  /:/  \:\  \
# /\ \:\ \ \__\ __/:/\/__//:/ |::::\__\/:/\:\ \:\__\/:/\:\ \:\__\/:/__/_\:\__\
# \:\ \:\ \/__//\/:/  /   \/__/~~/:/  /\/__\:\/:/  /\:\~\:\ \/__/\:\  /\ \/__/
#  \:\ \:\__\  \::/__/          /:/  /      \::/  /  \:\ \:\__\   \:\ \:\__\
#   \:\/:/  /   \:\__\         /:/  /        \/__/    \:\ \/__/    \:\/:/  /
#    \::/  /     \/__/        /:/  /                   \:\__\       \::/  /
#     \/__/                   \/__/                     \/__/        \/__/
#      ___          ___       ___          ___          ___          ___
#     /\  \        /\  \     /\  \        /\  \        /\  \        /\  \
#    /::\  \      /::\  \    \:\  \      /::\  \      /::\  \      /::\  \
#   /:/\:\  \    /:/\:\  \    \:\  \    /:/\:\  \    /:/\:\  \    /:/\:\  \
#  /:/  \:\  \  /:/  \:\  \   /::\  \  /::\~\:\  \  /::\~\:\  \  /::\~\:\  \
# /:/__/ \:\__\/:/__/ \:\__\ /:/\:\__\/:/\:\ \:\__\/:/\:\ \:\__\/:/\:\ \:\__\
# \:\  \ /:/  /\:\  \  \/__//:/  \/__/\/_|::\/:/  /\:\~\:\ \/__/\:\~\:\ \/__/
#  \:\  /:/  /  \:\  \     /:/  /        |:|::/  /  \:\ \:\__\   \:\ \:\__\
#   \:\/:/  /    \:\  \    \/__/         |:|\/__/    \:\ \/__/    \:\ \/__/
#    \::/  /      \:\__\                 |:|  |       \:\__\       \:\__\
#     \/__/        \/__/                  \|__|        \/__/        \/__/
#
#
#
#                      .----------------.----------------.
#                     /|               /|               /|
#                    / |              / |              / |
#                   /  |     011     /  |    111      /  |
#                  /   |            /   |            /   |
#                 .----------------.----+-----------.    |
#                /|    . ---------/|----.----------/|----.
#               / |   /|         / |   /|         / |   /|
#              /  |  / | 001    /  |  / |  101   /  |  / |
#             /   | /  |       /   | /  |       /   | /  |
#            . -------------- .----------------.    |/   |
#            |    . ---+------|----.----+------|----.    |
#            |   /|    .______|___/|____.______|___/|____.
#            |  / |   /   010 |  / |   /    110|  / |   /
#            | /  |  /        | /  |  /        | /  |  /
#            . ---+---------- . ---+---------- .    | /
#            |    |/          |    |/          |    |/             z
#            |    . ----------|----.-----------|----.              ^   y
#            |   /    000     |   /     100    |   /               |  /
#            |  /             |  /             |  /                | /
#            | /              | /              | /                 o----> x
#            . -------------- . -------------- .
#
#
# Face Refinement:
#
#      2_______________3                    _______________
#      |               |                   |       |       |
#   ^  |               |                   | (0,1) | (1,1) |
#   |  |               |                   |       |       |
#   |  |       x       |        --->       |-------+-------|
#   t1 |               |                   |       |       |
#      |               |                   | (0,0) | (1,0) |
#      |_______________|                   |_______|_______|
#      0      t0-->    1
#
#
# Face and Edge naming conventions:
#
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

from SimPEG import np, sp, Utils, Solver
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx

import TreeUtils
from InnerProducts import InnerProducts
from BaseMesh import BaseMesh
import time

MAX_BITS = 20

class TreeMesh(BaseMesh, InnerProducts):
    def __init__(self, h_in, x0_in=None, levels=3):
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

        x0 = np.zeros(len(h))
        if x0_in is not None:
            assert len(h) == len(x0_in), "Dimension mismatch. x0 != len(h)"
            for i in range(len(h)):
                x_i, h_i = x0_in[i], h[i]
                if Utils.isScalar(x_i):
                    x0[i] = x_i
                elif x_i == '0':
                    x0[i] = 0.0
                elif x_i == 'C':
                    x0[i] = -h_i.sum()*0.5
                elif x_i == 'N':
                    x0[i] = -h_i.sum()
                else:
                    raise Exception("x0[%i] must be a scalar or '0' to be zero, 'C' to center, or 'N' to be negative." % i)

        BaseMesh.__init__(self, [len(_) for _ in h], x0)

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

        deleteThese = [
                        '__sortedCells',
                        '_gridCC', '_gridN', '_gridFx', '_gridFy', '_gridFz', '_gridEx', '_gridEy', '_gridEz',
                        '_area', '_edge', '_vol',
                        '_faceDiv', '_edgeCurl', '_nodalGrad',
                        '_aveF2CC', '_aveF2CCV', '_aveE2CC', '_aveE2CCV','_aveN2CC'
                      ]
        for p in deleteThese:
            if hasattr(self, p): delattr(self, p)

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
    def nhN(self):
        self.number()
        return len(self._hangingN)

    @property
    def nhF(self):
        return self.nhFx + self.nhFy + (0 if self.dim == 2 else self.nhFz)

    @property
    def nhFx(self):
        self.number()
        return len(self._hangingFx)

    @property
    def nhFy(self):
        self.number()
        return len(self._hangingFy)

    @property
    def nhFz(self):
        if self.dim == 2: return None
        self.number()
        return len(self._hangingFz)

    @property
    def nhE(self):
        return self.nhEx + self.nhEy + (0 if self.dim == 2 else self.nhEz)

    @property
    def nhEx(self):
        if self.dim == 2:return self.nhFy
        self.number()
        return len(self._hangingEx)

    @property
    def nhEy(self):
        if self.dim == 2:return self.nhFx
        self.number()
        return len(self._hangingEy)

    @property
    def nhEz(self):
        if self.dim == 2: return None
        self.number()
        return len(self._hangingEz)


    @property
    def ntN(self):
        self.number()
        return len(self._nodes)

    @property
    def ntF(self):
        return self.ntFx + self.ntFy + (0 if self.dim == 2 else self.ntFz)

    @property
    def ntFx(self):
        self.number()
        return len(self._facesX)

    @property
    def ntFy(self):
        self.number()
        return len(self._facesY)

    @property
    def ntFz(self):
        if self.dim == 2: return None
        self.number()
        return len(self._facesZ)

    @property
    def ntE(self):
        return self.ntEx + self.ntEy + (0 if self.dim == 2 else self.ntEz)

    @property
    def ntEx(self):
        if self.dim == 2:return self.ntFy
        self.number()
        return len(self._edgesX)

    @property
    def ntEy(self):
        if self.dim == 2:return self.ntFx
        self.number()
        return len(self._edgesY)

    @property
    def ntEz(self):
        if self.dim == 2: return None
        self.number()
        return len(self._edgesZ)

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
            P = SortGrid(self.gridEx)
            P += SortGrid(self.gridEy, offset=self.nEx)
            P += SortGrid(self.gridEz, offset=self.nEx+self.nEy)
            return sp.identity(self.nE).tocsr()[P,:]

    def _index(self, pointer):
        assert len(pointer) is self.dim+1
        assert pointer[-1] <= self.levels
        return TreeUtils.index(self.dim, MAX_BITS, self._levelBits, pointer[:-1], pointer[-1])

    def _pointer(self, index):
        assert type(index) in [int, long]
        return TreeUtils.point(self.dim, MAX_BITS, self._levelBits, index)

    def __contains__(self, v):
        return self._asIndex(v) in self._cells

    def refine(self, function=None, recursive=True, cells=None, balance=True, verbose=False, _inRecursion=False):

        if not _inRecursion:
            self.__dirty__ = True
            if verbose: print 'Refining Mesh'

        cells = cells if cells is not None else sorted(self._cells)
        recurse = []
        tic = time.time()
        for cell in cells:
            p = self._pointer(cell)
            do = function(self._cellC(cell)) > p[-1]
            if do:
                recurse += self._refineCell(cell)

        if verbose: print '   ', time.time() - tic

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
        self.__dirty__ = True
        raise Exception('Not yet implemented')


    def _corsenCell(self, pointer):
        raise Exception('Not yet implemented')

        # something like this: ??
        pointer = self._asPointer(pointer)
        ind = self._asIndex(pointer)
        assert ind in self

        parent = self._parentPointer(ind)
        children = _childPointers(parent)
        for child in children:
            self._cells.remove(self._asIndex(child))

        parentInd = self._asIndex(parent)
        self._cells.add(parentInd)
        return parentInd

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


    def _childPointers(self, pointer, direction=0, positive=True, returnAll=False):
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

        if returnAll:
            return children
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

    def balance(self, recursive=True, cells=None, verbose=False, _inRecursion=False):

        tic = time.time()
        if not _inRecursion:
            self.__dirty__ = True
            if verbose: print 'Balancing Mesh:'

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

        if verbose: print '   ', len(cells), time.time() - tic
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
        R = self._deflationMatrix('N', withHanging=False)
        return R.T * self._gridN

    @property
    def gridFx(self):
        self.number()
        R = self._deflationMatrix('Fx', withHanging=False)
        return R.T * self._gridFx

    @property
    def gridFy(self):
        self.number()
        R = self._deflationMatrix('Fy', withHanging=False)
        return R.T * self._gridFy

    @property
    def gridFz(self):
        if self.dim < 3: return None
        self.number()
        R = self._deflationMatrix('Fz', withHanging=False)
        return R.T * self._gridFz

    @property
    def gridEx(self):
        if self.dim == 2: return self.gridFy
        self.number()
        R = self._deflationMatrix('Ex', withHanging=False)
        return R.T * self._gridEx

    @property
    def gridEy(self):
        if self.dim == 2: return self.gridFx
        self.number()
        R = self._deflationMatrix('Ey', withHanging=False)
        return R.T * self._gridEy

    @property
    def gridEz(self):
        if self.dim < 3: return None
        self.number()
        R = self._deflationMatrix('Ez', withHanging=False)
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
            Rf = self._deflationMatrix('F', withHanging=False)
            self._area = Rf.T * (
                                    np.r_[self._areaFxFull, self._areaFyFull] if self.dim == 2 else
                                    np.r_[self._areaFxFull, self._areaFyFull, self._areaFzFull]
                               )
        return self._area

    @property
    def edge(self):
        self.number()
        if self.dim == 2:
            return np.r_[self.area[self.nFx:], self.area[:self.nFx]]
        if getattr(self, '_edge', None) is None:
            Re = self._deflationMatrix('E', withHanging=False)
            self._edge = Re.T * np.r_[self._edgeExFull, self._edgeEyFull, self._edgeEzFull]

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
        if self.dim == 2:
            self.__dirtyEdges__ = False
            return
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
        self._edgeEyFull = np.array(edgeEy)

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
        self._edgeEzFull = np.array(edgeEz)

        self.__dirtyEdges__ = False

    def _hanging(self, force=False):
        if not self.__dirtyHanging__ and not force: return

        self._numberNodes(force=force)
        self._numberFaces(force=force)
        self._numberEdges(force=force)

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
                chy0 = self._cellH([p[0]    , p[1]    , sl])[1]
                chy1 = self._cellH([p[0]    , p[1] + w, sl])[1]
                A = (chy0 + chy1)

                self._hangingFx[self._fx2i[test                                 ]] = ([self._fx2i[fx], chy0 / A], )
                self._hangingFx[self._fx2i[self._index([p[0]    , p[1] + w, sl])]] = ([self._fx2i[fx], chy1 / A], )

                n0, n1 = fx, self._index([p[0], p[1] + 2*w, p[-1]])
                self._hangingN[self._n2i[test                                   ]] = ([self._n2i[n0], 1.0], )
                self._hangingN[self._n2i[self._index([p[0]    , p[1] +   w, sl])]] = ([self._n2i[n0], 1.0 - chy0 / A], [self._n2i[n1], 1.0 - chy1 / A])
                self._hangingN[self._n2i[self._index([p[0]    , p[1] + 2*w, sl])]] = ([self._n2i[n1], 1.0], )

            elif self.dim == 3:

                chy0 = self._cellH([p[0]    , p[1]    , p[2]    , sl])[1]
                chy1 = self._cellH([p[0]    , p[1] + w, p[2]    , sl])[1]
                chz0 = self._cellH([p[0]    , p[1]    , p[2]    , sl])[2]
                chz1 = self._cellH([p[0]    , p[1]    , p[2] + w, sl])[2]
                lenY = chy0 + chy1
                lenZ = chz0 + chz1
                A = lenY * lenZ

                ey0 = fx
                ey1 = self._index([p[0], p[1]      , p[2] + 2*w, p[-1]])
                ez0 = fx
                ez1 = self._index([p[0], p[1] + 2*w, p[2]      , p[-1]])

                n0  = fx
                n1  = self._index([p[0], p[1] + 2*w, p[2]      , p[-1]])
                n2  = self._index([p[0], p[1]      , p[2] + 2*w, p[-1]])
                n3  = self._index([p[0], p[1] + 2*w, p[2] + 2*w, p[-1]])

                self._hangingFx[self._fx2i[test                                           ]] = ([self._fx2i[fx], chy0*chz0 / A ], )
                self._hangingFx[self._fx2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._fx2i[fx], chy1*chz0 / A ], )
                self._hangingFx[self._fx2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._fx2i[fx], chy0*chz1 / A ], )
                self._hangingFx[self._fx2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._fx2i[fx], chy1*chz1 / A ], )

                self._hangingEy[self._ey2i[test                                           ]] = ([self._ey2i[ey0], 1.0], )
                self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._ey2i[ey0], 1.0], )
                self._hangingEy[self._ey2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._ey2i[ey0], 0.5], [self._ey2i[ey1], 0.5])
                self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._ey2i[ey0], 0.5], [self._ey2i[ey1], 0.5])
                self._hangingEy[self._ey2i[self._index([p[0], p[1]      , p[2] + 2*w, sl])]] = ([self._ey2i[ey1], 1.0], )
                self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2] + 2*w, sl])]] = ([self._ey2i[ey1], 1.0], )

                self._hangingEz[self._ez2i[test                                           ]] = ([self._ez2i[ez0], 1.0], )
                self._hangingEz[self._ez2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._ez2i[ez0], 1.0], )
                self._hangingEz[self._ez2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._ez2i[ez0], 0.5], [self._ez2i[ez1], 0.5])
                self._hangingEz[self._ez2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._ez2i[ez0], 0.5], [self._ez2i[ez1], 0.5])
                self._hangingEz[self._ez2i[self._index([p[0], p[1] + 2*w, p[2]      , sl])]] = ([self._ez2i[ez1], 1.0], )
                self._hangingEz[self._ez2i[self._index([p[0], p[1] + 2*w, p[2] +   w, sl])]] = ([self._ez2i[ez1], 1.0], )

                # self._hangingEy[self._ey2i[test                                           ]] = ([self._ey2i[ey0], chy0 / lenY], )
                # self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._ey2i[ey0], chy1 / lenY], )
                # self._hangingEy[self._ey2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._ey2i[ey0], chy0 / lenY / 2.0], [self._ey2i[ey1], chy0 / lenY / 2.0])
                # self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._ey2i[ey0], chy1 / lenY / 2.0], [self._ey2i[ey1], chy1 / lenY / 2.0])
                # self._hangingEy[self._ey2i[self._index([p[0], p[1]      , p[2] + 2*w, sl])]] = ([self._ey2i[ey1], chy0 / lenY], )
                # self._hangingEy[self._ey2i[self._index([p[0], p[1] +   w, p[2] + 2*w, sl])]] = ([self._ey2i[ey1], chy1 / lenY], )

                # self._hangingEz[self._ez2i[test                                           ]] = ([self._ez2i[ez0], chz0 / lenZ], )
                # self._hangingEz[self._ez2i[self._index([p[0], p[1]      , p[2] +   w, sl])]] = ([self._ez2i[ez0], chz1 / lenZ], )
                # self._hangingEz[self._ez2i[self._index([p[0], p[1] +   w, p[2]      , sl])]] = ([self._ez2i[ez0], chz0 / lenZ / 2.0], [self._ez2i[ez1], chz0 / lenZ / 2.0])
                # self._hangingEz[self._ez2i[self._index([p[0], p[1] +   w, p[2] +   w, sl])]] = ([self._ez2i[ez0], chz1 / lenZ / 2.0], [self._ez2i[ez1], chz1 / lenZ / 2.0])
                # self._hangingEz[self._ez2i[self._index([p[0], p[1] + 2*w, p[2]      , sl])]] = ([self._ez2i[ez1], chz0 / lenZ], )
                # self._hangingEz[self._ez2i[self._index([p[0], p[1] + 2*w, p[2] +   w, sl])]] = ([self._ez2i[ez1], chz1 / lenZ], )

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
                chx0 = self._cellH([p[0]    , p[1]    , sl])[0]
                chx1 = self._cellH([p[0] + w, p[1]    , sl])[0]

                self._hangingFy[self._fy2i[test                                 ]] = ([self._fy2i[fy], chx0 / (chx0 + chx1)], )
                self._hangingFy[self._fy2i[self._index([p[0] + w, p[1]    , sl])]] = ([self._fy2i[fy], chx1 / (chx0 + chx1)], )

                n0, n1 = fy, self._index([p[0] + 2*w, p[1], p[-1]])
                self._hangingN[self._n2i[test                                   ]] = ([self._n2i[n0], 1.0], )
                self._hangingN[self._n2i[self._index([p[0] +   w, p[1]    , sl])]] = ([self._n2i[n0], 0.5], [self._n2i[n1], 0.5])
                self._hangingN[self._n2i[self._index([p[0] + 2*w, p[1]    , sl])]] = ([self._n2i[n1], 1.0], )

            elif self.dim == 3:

                chx0 = self._cellH([p[0]    , p[1]    , p[2]    , sl])[0]
                chx1 = self._cellH([p[0] + w, p[1]    , p[2]    , sl])[0]
                chz0 = self._cellH([p[0]    , p[1]    , p[2]    , sl])[2]
                chz1 = self._cellH([p[0]    , p[1]    , p[2] + w, sl])[2]
                lenX = chx0 + chx1
                lenZ = chz0 + chz1
                A = lenX * lenZ

                ex0 = fy
                ex1 = self._index([p[0]      , p[1], p[2] + 2*w, p[-1]])
                ez0 = fy
                ez1 = self._index([p[0] + 2*w, p[1], p[2]      , p[-1]])

                n0  = fy
                n1  = self._index([p[0] + 2*w, p[1], p[2]      , p[-1]])
                n2  = self._index([p[0]      , p[1], p[2] + 2*w, p[-1]])
                n3  = self._index([p[0] + 2*w, p[1], p[2] + 2*w, p[-1]])

                self._hangingFy[self._fy2i[test                                           ]] = ([self._fy2i[fy], chx0*chz0 / A ], )
                self._hangingFy[self._fy2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._fy2i[fy], chx1*chz0 / A ], )
                self._hangingFy[self._fy2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._fy2i[fy], chx0*chz1 / A ], )
                self._hangingFy[self._fy2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._fy2i[fy], chx1*chz1 / A ], )

                self._hangingEx[self._ex2i[test                                           ]] = ([self._ex2i[ex0], 1.0], )
                self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._ex2i[ex0], 1.0], )
                self._hangingEx[self._ex2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._ex2i[ex0], 0.5], [self._ex2i[ex1], 0.5])
                self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._ex2i[ex0], 0.5], [self._ex2i[ex1], 0.5])
                self._hangingEx[self._ex2i[self._index([p[0]      , p[1], p[2] + 2*w, sl])]] = ([self._ex2i[ex1], 1.0], )
                self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2] + 2*w, sl])]] = ([self._ex2i[ex1], 1.0], )

                self._hangingEz[self._ez2i[test                                           ]] = ([self._ez2i[ez0], 1.0], )
                self._hangingEz[self._ez2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._ez2i[ez0], 1.0], )
                self._hangingEz[self._ez2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._ez2i[ez0], 0.5], [self._ez2i[ez1], 0.5])
                self._hangingEz[self._ez2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._ez2i[ez0], 0.5], [self._ez2i[ez1], 0.5])
                self._hangingEz[self._ez2i[self._index([p[0] + 2*w, p[1], p[2]      , sl])]] = ([self._ez2i[ez1], 1.0], )
                self._hangingEz[self._ez2i[self._index([p[0] + 2*w, p[1], p[2] +   w, sl])]] = ([self._ez2i[ez1], 1.0], )

                # self._hangingEx[self._ex2i[test                                           ]] = ([self._ex2i[ex0], chx0 / lenX], )
                # self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._ex2i[ex0], chx1 / lenX], )
                # self._hangingEx[self._ex2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._ex2i[ex0], chx0 / lenX / 2.0], [self._ex2i[ex1], chx0 / lenX / 2.0])
                # self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._ex2i[ex0], chx1 / lenX / 2.0], [self._ex2i[ex1], chx1 / lenX / 2.0])
                # self._hangingEx[self._ex2i[self._index([p[0]      , p[1], p[2] + 2*w, sl])]] = ([self._ex2i[ex1], chx0 / lenX], )
                # self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1], p[2] + 2*w, sl])]] = ([self._ex2i[ex1], chx1 / lenX], )

                # self._hangingEz[self._ez2i[test                                           ]] = ([self._ez2i[ez0], chz0 / lenZ], )
                # self._hangingEz[self._ez2i[self._index([p[0]      , p[1], p[2] +   w, sl])]] = ([self._ez2i[ez0], chz1 / lenZ], )
                # self._hangingEz[self._ez2i[self._index([p[0] +   w, p[1], p[2]      , sl])]] = ([self._ez2i[ez0], chz0 / lenZ / 2.0], [self._ez2i[ez1], chz0 / lenZ / 2.0])
                # self._hangingEz[self._ez2i[self._index([p[0] +   w, p[1], p[2] +   w, sl])]] = ([self._ez2i[ez0], chz1 / lenZ / 2.0], [self._ez2i[ez1], chz1 / lenZ / 2.0])
                # self._hangingEz[self._ez2i[self._index([p[0] + 2*w, p[1], p[2]      , sl])]] = ([self._ez2i[ez1], chz0 / lenZ], )
                # self._hangingEz[self._ez2i[self._index([p[0] + 2*w, p[1], p[2] +   w, sl])]] = ([self._ez2i[ez1], chz1 / lenZ], )

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

            chx0 = self._cellH([p[0]    , p[1]    , p[2]    , sl])[0]
            chx1 = self._cellH([p[0] + w, p[1]    , p[2]    , sl])[0]
            chy0 = self._cellH([p[0]    , p[1]    , p[2]    , sl])[1]
            chy1 = self._cellH([p[0]    , p[1] + w, p[2]    , sl])[1]
            lenX = chx0 + chx1
            lenY = chy0 + chy1
            A = lenX * lenY

            ex0 = fz
            ex1 = self._index([p[0]      , p[1] + 2*w, p[2], p[-1]])
            ey0 = fz
            ey1 = self._index([p[0] + 2*w, p[1]      , p[2], p[-1]])

            n0  = fz
            n1  = self._index([p[0] + 2*w, p[1]      , p[2], p[-1]])
            n2  = self._index([p[0]      , p[1] + 2*w, p[2], p[-1]])
            n3  = self._index([p[0] + 2*w, p[1] + 2*w, p[2], p[-1]])

            self._hangingFz[self._fz2i[test                                           ]] = ([self._fz2i[fz], chx0*chy0 / A ], )
            self._hangingFz[self._fz2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._fz2i[fz], chx1*chy0 / A ], )
            self._hangingFz[self._fz2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._fz2i[fz], chx0*chy1 / A ], )
            self._hangingFz[self._fz2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._fz2i[fz], chx1*chy1 / A ], )

            self._hangingEx[self._ex2i[test                                           ]] = ([self._ex2i[ex0], 1.0], )
            self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._ex2i[ex0], 1.0], )
            self._hangingEx[self._ex2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._ex2i[ex0], 0.5], [self._ex2i[ex1], 0.5])
            self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._ex2i[ex0], 0.5], [self._ex2i[ex1], 0.5])
            self._hangingEx[self._ex2i[self._index([p[0]      , p[1] + 2*w, p[2], sl])]] = ([self._ex2i[ex1], 1.0], )
            self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1] + 2*w, p[2], sl])]] = ([self._ex2i[ex1], 1.0], )

            self._hangingEy[self._ey2i[test                                           ]] = ([self._ey2i[ey0], 1.0], )
            self._hangingEy[self._ey2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._ey2i[ey0], 1.0], )
            self._hangingEy[self._ey2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._ey2i[ey0], 0.5], [self._ey2i[ey1], 0.5])
            self._hangingEy[self._ey2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._ey2i[ey0], 0.5], [self._ey2i[ey1], 0.5])
            self._hangingEy[self._ey2i[self._index([p[0] + 2*w, p[1]      , p[2], sl])]] = ([self._ey2i[ey1], 1.0], )
            self._hangingEy[self._ey2i[self._index([p[0] + 2*w, p[1] +   w, p[2], sl])]] = ([self._ey2i[ey1], 1.0], )

            # self._hangingEx[self._ex2i[test                                           ]] = ([self._ex2i[ex0], chx0 / lenX], )
            # self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._ex2i[ex0], chx1 / lenX], )
            # self._hangingEx[self._ex2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._ex2i[ex0], chx0 / lenX / 2.0], [self._ex2i[ex1], chx0 / lenX / 2.0])
            # self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._ex2i[ex0], chx1 / lenX / 2.0], [self._ex2i[ex1], chx1 / lenX / 2.0])
            # self._hangingEx[self._ex2i[self._index([p[0]      , p[1] + 2*w, p[2], sl])]] = ([self._ex2i[ex1], chx0 / lenX], )
            # self._hangingEx[self._ex2i[self._index([p[0] +   w, p[1] + 2*w, p[2], sl])]] = ([self._ex2i[ex1], chx1 / lenX], )

            # self._hangingEy[self._ey2i[test                                           ]] = ([self._ey2i[ey0], chy0 / lenY], )
            # self._hangingEy[self._ey2i[self._index([p[0]      , p[1] +   w, p[2], sl])]] = ([self._ey2i[ey0], chy1 / lenY], )
            # self._hangingEy[self._ey2i[self._index([p[0] +   w, p[1]      , p[2], sl])]] = ([self._ey2i[ey0], chy0 / lenY / 2.0], [self._ey2i[ey1], chy0 / lenY / 2.0])
            # self._hangingEy[self._ey2i[self._index([p[0] +   w, p[1] +   w, p[2], sl])]] = ([self._ey2i[ey0], chy1 / lenY / 2.0], [self._ey2i[ey1], chy1 / lenY / 2.0])
            # self._hangingEy[self._ey2i[self._index([p[0] + 2*w, p[1]      , p[2], sl])]] = ([self._ey2i[ey1], chy0 / lenY], )
            # self._hangingEy[self._ey2i[self._index([p[0] + 2*w, p[1] +   w, p[2], sl])]] = ([self._ey2i[ey1], chy1 / lenY], )

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

    def number(self, balance=True, force=False):
        if not self.__dirty__ and not force: return
        if balance: self.balance()
        self._hanging(force=force)

    def _deflationMatrix(self, location, withHanging=True, asOnes=False):
        assert location in ['N','F','Fx','Fy'] + (['Fz','E','Ex','Ey','Ez'] if self.dim == 3 else [])

        args = dict()
        args['N'] =  (self._nodes,  self._hangingN,  self._n2i )
        args['Fx'] = (self._facesX, self._hangingFx, self._fx2i)
        args['Fy'] = (self._facesY, self._hangingFy, self._fy2i)
        if self.dim == 3:
            args['Fz'] = (self._facesZ, self._hangingFz, self._fz2i)
            args['Ex'] = (self._edgesX, self._hangingEx, self._ex2i)
            args['Ey'] = (self._edgesY, self._hangingEy, self._ey2i)
            args['Ez'] = (self._edgesZ, self._hangingEz, self._ez2i)
        if location in ['F', 'E']:
            Rlist = [self._deflationMatrix(location + subLoc, withHanging=withHanging, asOnes=asOnes) for subLoc in ['x','y','z'][:self.dim]]
            return sp.block_diag(Rlist)
        return self.__deflationMatrix(*args[location], withHanging=withHanging, asOnes=asOnes)

    def __deflationMatrix(self, theSet, theHang, theIndex, withHanging=True, asOnes=False):
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
                if asOnes:
                    V += [1.0]*len(hf)
                else:
                    V += [_[1] for _ in hf]
        return sp.csr_matrix((V,(I,J)), shape=(len(theSet), len(reducedInd)))

    @property
    def faceDiv(self):
        if getattr(self, '_faceDiv', None) is None:
            self.number()

            # TODO: Preallocate!
            I, J, V = [], [], []
            PM = [-1,1]*self.dim # plus / minus

            # TODO total number of faces?
            offset = [0]*2 + [self.ntFx]*2 + [self.ntFx+self.ntFy]*2

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

            D = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntF))
            R = self._deflationMatrix('F',asOnes=True)
            VOL = self.vol
            if self.dim == 2:
                S = np.r_[self._areaFxFull, self._areaFyFull]
            elif self.dim == 3:
                S = np.r_[self._areaFxFull, self._areaFyFull, self._areaFzFull]
            self._faceDiv = Utils.sdiag(1.0/VOL)*D*Utils.sdiag(S)*R
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
            offset = [self.ntEx]*2 + [self.ntEx+self.ntEy]*2
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

            faceOffset = self.ntFx
            offset = [0]*2 + [self.ntEx+self.ntEy]*2
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

            faceOffset = self.ntFx + self.ntFy
            offset = [0]*2 + [self.ntEx]*2
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

            Rf = self._deflationMatrix('F', withHanging=True, asOnes=False)
            Re = self._deflationMatrix('E')

            Rf_ave = Utils.sdiag(1./Rf.sum(axis=0)) * Rf.T

            C = sp.csr_matrix((V,(I,J)), shape=(self.ntF, self.ntE))
            S = np.r_[self._areaFxFull, self._areaFyFull, self._areaFzFull]
            L = np.r_[self._edgeExFull, self._edgeEyFull, self._edgeEzFull]
            self._edgeCurl = Rf_ave*Utils.sdiag(1.0/S)*C*Utils.sdiag(L)*Re
        return self._edgeCurl


    @property
    def nodalGrad(self):
        raise Exception('Not yet implemented!')
        # if getattr(self, '_nodalGrad', None) is None:
        #     self.number()
        #     # TODO: Preallocate!
        #     I, J, V = [], [], []
        #     # kinda a hack for the 2D gradient
        #     # because edges are not stored
        #     edges = self.faces if self.dim == 2 else self.edges
        #     for edge in edges:
        #         if self.dim == 3:
        #             I += [edge.num, edge.num]
        #         elif self.dim == 2 and edge.faceType == 'x':
        #             I += [edge.num + self.nFy, edge.num + self.nFy]
        #         elif self.dim == 2 and edge.faceType == 'y':
        #             I += [edge.num - self.nFx, edge.num - self.nFx]
        #         J += [edge.node0.num, edge.node1.num]
        #         V += [-1, 1]
        #     G = sp.csr_matrix((V,(I,J)), shape=(self.nE, self.nN))
        #     L = self.edge
        #     self._nodalGrad = Utils.sdiag(1/L)*G
        # return self._nodalGrad

    # @property
    # def aveE2CC(self):
    #     "Construct the averaging operator on cell edges to cell centers."
    #     if getattr(self, '_aveE2CC', None) is None:
            
    #         # TODO: preallocate
    #         I, J, V = [], [], []
            
    #         if self.dim == 2:
    #             raise NotImplementedError('aveE2CC not implemented yet')

    #         if self.dim == 3:
    #             PM = [1./(4.*self.dim)]*4*self.dim # plus / plus
    #             offset = [0]*4 + [self.ntEx]*4 + [self.ntEx+self.ntEy]*4

    #             for ii, ind in enumerate(self._sortedCells):
    #                 p = self._pointer(ind)
    #                 w = self._levelWidth(p[-1])

    #                 edges = [
    #                             self._ex2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
    #                             self._ex2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
    #                             self._ex2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])],
    #                             self._ex2i[self._index([ p[0]    , p[1] + w, p[2] + w, p[3]])],
    #                             self._ey2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
    #                             self._ey2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
    #                             self._ey2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])],
    #                             self._ey2i[self._index([ p[0] + w, p[1]    , p[2] + w, p[3]])],
    #                             self._ez2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
    #                             self._ez2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
    #                             self._ez2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
    #                             self._ez2i[self._index([ p[0] + w, p[1] + w, p[2]    , p[3]])]
    #                         ]

    #                 for off, pm, edge in zip(offset,PM,edges):
    #                     I += [ii]
    #                     J += [edge + off]
    #                     V += [pm]


    #         Av = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntE))
    #         Re = self._deflationMatrix('E',asOnes=False,withHanging=True)

    #         self._aveE2CC = Av*Re

    #     return self._aveE2CC

    @property
    def aveEx2CC(self):
        if getattr(self, '_aveEx2CC', None) is None:
            I, J, V = [], [], []
            
            if self.dim == 2:
                raise Exception('aveEx2CC not implemented in 2D')

            if self.dim == 3:
                PM = [1./4.]*4 

                for ii, ind in enumerate(self._sortedCells):
                    p = self._pointer(ind)
                    w = self._levelWidth(p[-1])

                    edgesx = [
                                self._ex2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._ex2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
                                self._ex2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])],
                                self._ex2i[self._index([ p[0]    , p[1] + w, p[2] + w, p[3]])],
                            ]

                    for pm, edge in zip(PM,edgesx):
                        I += [ii]
                        J += [edge]
                        V += [pm]

            Av = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntEx))
            Re = self._deflationMatrix('Ex',asOnes=False,withHanging=True)

            self._aveEx2CC = Av*Re
        return self._aveEx2CC
    
    @property
    def aveEy2CC(self):
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, '_aveEy2CC', None) is None:
            I, J, V = [], [], []
            
            if self.dim == 2:
                raise NotImplementedError('aveEy2CC not implemented in 2D')

            if self.dim == 3:
                PM = [1./4.]*4 # plus / plus

                for ii, ind in enumerate(self._sortedCells):
                    p = self._pointer(ind)
                    w = self._levelWidth(p[-1])

                    edgesy = [
                                self._ey2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._ey2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
                                self._ey2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])],
                                self._ey2i[self._index([ p[0] + w, p[1]    , p[2] + w, p[3]])],
                            ]

                    for pm, edge in zip(PM,edgesy):
                        I += [ii]
                        J += [edge]
                        V += [pm]

            Av = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntEy))
            Re = self._deflationMatrix('Ey',asOnes=False,withHanging=True)

            self._aveEy2CC = Av*Re
        return self._aveEy2CC

    @property
    def aveEz2CC(self):
        "Construct the averaging operator on cell edges to cell centers."
        # raise Exception('Not yet implemented!')
        if getattr(self, '_aveEz2CC', None) is None:
            I, J, V = [], [], []
            
            if self.dim == 2:
                raise Exception('There are no z edges in 2D')

            if self.dim == 3:
                PM = [1./4.]*4 # plus / plus

                for ii, ind in enumerate(self._sortedCells):
                    p = self._pointer(ind)
                    w = self._levelWidth(p[-1])

                    edgesz = [
                                self._ez2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._ez2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
                                self._ez2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
                                self._ez2i[self._index([ p[0] + w, p[1] + w, p[2]    , p[3]])],
                            ]

                    for pm, edge in zip(PM,edgesz):
                        I += [ii]
                        J += [edge]
                        V += [pm]


            Av = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntEz))
            Re = self._deflationMatrix('Ez',asOnes=False,withHanging=True)

            self._aveEz2CC = Av*Re

        return self._aveEz2CC

    @property
    def aveE2CC(self):
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, '_aveE2CC', None) is None:
            if self.dim == 2:
                raise Exception('aveE2CC not implemented in 2D')
            elif self.dim == 3:            
                self._aveE2CC = 1./self.dim*sp.hstack([self.aveEx2CC, self.aveEy2CC, self.aveEz2CC])
        return self._aveE2CC

    @property
    def aveE2CCV(self):
        "Construct the averaging operator on cell edges to cell centers."
        # raise Exception('Not yet implemented!')
        if getattr(self, '_aveE2CCV', None) is None:
            if self.dim == 2:
                raise Exception('aveE2CC not implemented in 2D')
            elif self.dim == 3:
                self._aveE2CCV = sp.block_diag([self.aveEx2CC, self.aveEy2CC, self.aveEz2CC])
        return self._aveE2CCV

    @property 
    def aveFx2CC(self):
        if getattr(self, '_aveFx2CC', None) is None:
            I, J, V = [], [], []
            PM = [1./2.]*self.dim # 0.5, 0.5
         
            for ii, ind in enumerate(self._sortedCells):
                p = self._pointer(ind)
                w = self._levelWidth(p[-1])

                if self.dim == 2:
                    facesx = [
                                self._fx2i[self._index([ p[0]    , p[1]    , p[2]])],
                                self._fx2i[self._index([ p[0] + w, p[1]    , p[2]])],
                            ]
            
                elif self.dim == 3:
                    facesx = [
                                self._fx2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._fx2i[self._index([ p[0] + w, p[1]    , p[2]    , p[3]])],
                            ]

                for pm, face in zip(PM,facesx):
                    I += [ii]
                    J += [face]
                    V += [pm]
           
            Av = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntFx))
            Rf = self._deflationMatrix('Fx',asOnes=True,withHanging=True)

            self._aveFx2CC = Av*Rf
        return self._aveFx2CC

    @property 
    def aveFy2CC(self):
        if getattr(self, '_aveFy2CC', None) is None:
            I, J, V = [], [], []
            PM = [1./2.]*2 # 0.5, 0.5
  
            for ii, ind in enumerate(self._sortedCells):
                p = self._pointer(ind)
                w = self._levelWidth(p[-1])

                if self.dim == 2:
                    facesy = [
                                self._fy2i[self._index([ p[0]    , p[1]    , p[2]])],
                                self._fy2i[self._index([ p[0]    , p[1] + w, p[2]])],
                            ]
                elif self.dim == 3:
                    facesy = [
                                self._fy2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._fy2i[self._index([ p[0]    , p[1] + w, p[2]    , p[3]])],
                            ]

                for pm, face in zip(PM,facesy):
                    I += [ii]
                    J += [face]
                    V += [pm]

            Av = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntFy))
            Rf = self._deflationMatrix('Fy',asOnes=True,withHanging=True)

            self._aveFy2CC = Av*Rf
        return self._aveFy2CC

    @property 
    def aveFz2CC(self):
        if getattr(self, '_aveFz2CC', None) is None:
            I, J, V = [], [], []
            PM = [1./2.]*2 # 0.5, 0.5

            for ii, ind in enumerate(self._sortedCells):
                p = self._pointer(ind)
                w = self._levelWidth(p[-1])

                if self.dim == 2: 
                    raise Exception('There are no z-faces in 2D')
                elif self.dim == 3:
                    facesz = [
                                self._fz2i[self._index([ p[0]    , p[1]    , p[2]    , p[3]])],
                                self._fz2i[self._index([ p[0]    , p[1]    , p[2] + w, p[3]])],
                            ]

                for pm, face in zip(PM,facesz):
                    I += [ii]
                    J += [face]
                    V += [pm]
           
            Av = sp.csr_matrix((V,(I,J)), shape=(self.nC, self.ntFz))
            Rf = self._deflationMatrix('Fz',asOnes=True,withHanging=True)
            self._aveFz2CC = Av*Rf
        return self._aveFz2CC

    @property
    def aveF2CC(self):
        "Construct the averaging operator on cell faces to cell centers."
        if getattr(self, '_aveF2CC', None) is None:
            if self.dim == 2:
                self._aveF2CC = 1./self.dim*sp.hstack([self.aveFx2CC, self.aveFy2CC])
            elif self.dim == 3:
                self._aveF2CC = 1./self.dim*sp.hstack([self.aveFx2CC, self.aveFy2CC, self.aveFz2CC])
        return self._aveF2CC

    @property
    def aveF2CCV(self):
        "Construct the averaging operator on cell faces to cell centers."
        if getattr(self, '_aveF2CCV', None) is None:
            if self.dim == 2:
                self._aveF2CCV = sp.block_diag([self.aveFx2CC, self.aveFy2CC])
            elif self.dim == 3:
                self._aveF2CCV = sp.block_diag([self.aveFx2CC, self.aveFy2CC, self.aveFz2CC])
        return self._aveF2CCV


    def _getFaceP(self, xFace, yFace, zFace):
        ind1, ind2, ind3 = [], [], []
        for ind in self._sortedCells:
            p = self._pointer(ind)
            w = self._levelWidth(p[-1])

            posX = 0 if xFace == 'fXm' else w
            posY = 0 if yFace == 'fYm' else w
            if self.dim == 3:
                posZ = 0 if zFace == 'fZm' else w

            ind1.append( self._fx2i[self._index([ p[0] + posX, p[1]] +  p[2:])]                   )
            ind2.append( self._fy2i[self._index([ p[0], p[1] + posY] +  p[2:])] + self.ntFx       )
            if self.dim == 3:
                ind3.append( self._fz2i[self._index([ p[0], p[1], p[2] + posZ, p[3]])] + self.ntFx + self.ntFy )

        if self.dim == 2:
            IND = np.r_[ind1, ind2]
        if self.dim == 3:
            IND = np.r_[ind1, ind2, ind3]

        PXXX = sp.coo_matrix((np.ones(self.dim*self.nC), (range(self.dim*self.nC), IND)), shape=(self.dim*self.nC, self.ntF)).tocsr()

        Rf = self._deflationMatrix('F', withHanging=True, asOnes=True)

        return PXXX * Rf

    def _getFacePxx(self):
        self.number()
        def Pxx(xFace, yFace):
            return self._getFaceP(xFace, yFace, None)
        return Pxx

    def _getFacePxxx(self):
        self.number()
        def Pxxx(xFace, yFace, zFace):
            return self._getFaceP(xFace, yFace, zFace)
        return Pxxx

    def _getEdgeP(self, xEdge, yEdge, zEdge):
        if self.dim == 2: raise Exception('Not implemented') # this should be a reordering of the face inner product?

        ind1, ind2, ind3 = [], [], []
        for ind in self._sortedCells:
            p = self._pointer(ind)
            w = self._levelWidth(p[-1])

            posX = [0,0] if xEdge == 'eX0' else [w, 0] if xEdge == 'eX1' else [0,w] if xEdge == 'eX2' else [w,w]
            posY = [0,0] if yEdge == 'eY0' else [w, 0] if yEdge == 'eY1' else [0,w] if yEdge == 'eY2' else [w,w]
            posZ = [0,0] if zEdge == 'eZ0' else [w, 0] if zEdge == 'eZ1' else [0,w] if zEdge == 'eZ2' else [w,w]

            ind1.append( self._ex2i[self._index([ p[0]          , p[1] + posX[0], p[2] + posX[1], p[3]])]                         )
            ind2.append( self._ey2i[self._index([ p[0] + posY[0], p[1]          , p[2] + posY[1], p[3]])] + self.ntEx             )
            ind3.append( self._ez2i[self._index([ p[0] + posZ[0], p[1] + posZ[1], p[2]          , p[3]])] + self.ntEx + self.ntEy )

        IND = np.r_[ind1, ind2, ind3]

        PXXX = sp.coo_matrix((np.ones(self.dim*self.nC), (range(self.dim*self.nC), IND)), shape=(self.dim*self.nC, self.ntE)).tocsr()

        Re = self._deflationMatrix('E')

        return PXXX * Re

    def _getEdgePxx(self):
        raise Exception('Not implemented') # this should be a reordering of the face inner product?
    def _getEdgePxxx(self):
        self.number()
        def Pxxx(xEdge, yEdge, zEdge):
            return self._getEdgeP(xEdge, yEdge, zEdge)
        return Pxxx


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
                ax.plot(self._gridN[:,0], self._gridN[:,1], 'ms')
                ax.plot(self._gridN[self._hangingN.keys(),0], self._gridN[self._hangingN.keys(),1], 'ms', ms=10, mfc='none', mec='m')
            if facesX:
                ax.plot(self._gridFx[self._hangingFx.keys(),0], self._gridFx[self._hangingFx.keys(),1], 'gs', ms=10, mfc='none', mec='g')
                ax.plot(self._gridFx[:,0], self._gridFx[:,1], 'g>')
            if facesY:
                ax.plot(self._gridFy[self._hangingFy.keys(),0], self._gridFy[self._hangingFy.keys(),1], 'gs', ms=10, mfc='none', mec='g')
                ax.plot(self._gridFy[:,0], self._gridFy[:,1], 'g^')
        elif self.dim == 3:
            if cells:
                ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r.', zs=self.gridCC[:,2])
            if cellLine:
                ax.plot(self.gridCC[:,0], self.gridCC[:,1], 'r:', zs=self.gridCC[:,2])
                ax.plot(self.gridCC[[0,-1],0], self.gridCC[[0,-1],1], 'ro', zs=self.gridCC[[0,-1],2])

            if nodes:
                ax.plot(self._gridN[:,0], self._gridN[:,1], 'ms', zs=self._gridN[:,2])
                ax.plot(self._gridN[self._hangingN.keys(),0], self._gridN[self._hangingN.keys(),1], 'ms', ms=10, mfc='none', mec='m', zs=self._gridN[self._hangingN.keys(),2])
                for key in self._hangingN.keys():
                    for hf in self._hangingN[key]:
                        ind = [key, hf[0]]
                        ax.plot(self._gridN[ind,0], self._gridN[ind,1], 'm:', zs=self._gridN[ind,2])

            if facesX:
                ax.plot(self._gridFx[:,0], self._gridFx[:,1], 'g>', zs=self._gridFx[:,2])
                ax.plot(self._gridFx[self._hangingFx.keys(),0], self._gridFx[self._hangingFx.keys(),1], 'gs', ms=10, mfc='none', mec='g', zs=self._gridFx[self._hangingFx.keys(),2])
                for key in self._hangingFx.keys():
                    for hf in self._hangingFx[key]:
                        ind = [key, hf[0]]
                        ax.plot(self._gridFx[ind,0], self._gridFx[ind,1], 'g:', zs=self._gridFx[ind,2])

            if facesY:
                ax.plot(self._gridFy[:,0], self._gridFy[:,1], 'g^', zs=self._gridFy[:,2])
                ax.plot(self._gridFy[self._hangingFy.keys(),0], self._gridFy[self._hangingFy.keys(),1], 'gs', ms=10, mfc='none', mec='g', zs=self._gridFy[self._hangingFy.keys(),2])
                for key in self._hangingFy.keys():
                    for hf in self._hangingFy[key]:
                        ind = [key, hf[0]]
                        ax.plot(self._gridFy[ind,0], self._gridFy[ind,1], 'g:', zs=self._gridFy[ind,2])

            if facesZ:
                ax.plot(self._gridFz[:,0], self._gridFz[:,1], 'g^', zs=self._gridFz[:,2])
                ax.plot(self._gridFz[self._hangingFz.keys(),0], self._gridFz[self._hangingFz.keys(),1], 'gs', ms=10, mfc='none', mec='g', zs=self._gridFz[self._hangingFz.keys(),2])
                for key in self._hangingFz.keys():
                    for hf in self._hangingFz[key]:
                        ind = [key, hf[0]]
                        ax.plot(self._gridFz[ind,0], self._gridFz[ind,1], 'g:', zs=self._gridFz[ind,2])

            if edgesX:
                ax.plot(self._gridEx[:,0], self._gridEx[:,1], 'k>', zs=self._gridEx[:,2])
                ax.plot(self._gridEx[self._hangingEx.keys(),0], self._gridEx[self._hangingEx.keys(),1], 'ks', ms=10, mfc='none', mec='k', zs=self._gridEx[self._hangingEx.keys(),2])
                for key in self._hangingEx.keys():
                    for hf in self._hangingEx[key]:
                        ind = [key, hf[0]]
                        ax.plot(self._gridEx[ind,0], self._gridEx[ind,1], 'k:', zs=self._gridEx[ind,2])


            if edgesY:
                ax.plot(self._gridEy[:,0], self._gridEy[:,1], 'k<', zs=self._gridEy[:,2])
                ax.plot(self._gridEy[self._hangingEy.keys(),0], self._gridEy[self._hangingEy.keys(),1], 'ks', ms=10, mfc='none', mec='k', zs=self._gridEy[self._hangingEy.keys(),2])
                for key in self._hangingEy.keys():
                    for hf in self._hangingEy[key]:
                        ind = [key, hf[0]]
                        ax.plot(self._gridEy[ind,0], self._gridEy[ind,1], 'k:', zs=self._gridEy[ind,2])

            if edgesZ:
                ax.plot(self._gridEz[:,0], self._gridEz[:,1], 'k^', zs=self._gridEz[:,2])
                ax.plot(self._gridEz[self._hangingEz.keys(),0], self._gridEz[self._hangingEz.keys(),1], 'ks', ms=10, mfc='none', mec='k', zs=self._gridEz[self._hangingEz.keys(),2])
                for key in self._hangingEz.keys():
                    for hf in self._hangingEz[key]:
                        ind = [key, hf[0]]
                        ax.plot(self._gridEz[ind,0], self._gridEz[ind,1], 'k:', zs=self._gridEz[ind,2])

        if showIt:plt.show()


    def plotImage(self, I, ax=None, showIt=True):
        if self.dim == 3: raise Exception()

        if ax is None: ax = plt.subplot(111)
        jet = cm = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=I.min(), vmax=I.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        ax.set_xlim((self.x0[0], self.h[0].sum()))
        ax.set_ylim((self.x0[1], self.h[1].sum()))
        for ii, node in enumerate(self._sortedCells):
            x0, sz = self._cellN(node), self._cellH(node)
            ax.add_patch(plt.Rectangle((x0[0], x0[1]), sz[0], sz[1], facecolor=scalarMap.to_rgba(I[ii]), edgecolor='k'))
            # if text: ax.text(self.center[0],self.center[1],self.num)
        scalarMap._A = []  # http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
        plt.colorbar(scalarMap)
        if showIt: plt.show()


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

if __name__ == '__main__':


    def function(xc):
        r = xc - np.array([0.5*128]*len(xc))
        dist = np.sqrt(r.dot(r))
        # if dist < 0.05:
        #     return 5
        if dist < 0.1*128:
            return 4
        if dist < 0.3*128:
            return 3
        if dist < 1.0*128:
            return 2
        else:
            return 0

    # T = Tree([[(1,128)],[(1,128)],[(1,128)]],levels=7)
    # T = Tree([128,128,128],levels=7)
    T = Tree([[(1,16)],[(1,16)]],levels=4)
    # T = Tree([[(1,128)],[(1,128)]],levels=7)
    # T.refine(lambda xc:1, balance=False)
    # T._index([0,0,0])
    # T._pointer(0)


    tic = time.time()
    T.refine(function)#, balance=False)
    print time.time() - tic
    print T.nC

    T.plotImage(np.random.rand(T.nC),showIt=True)

    print T.getFaceInnerProduct()
    # print T.gridFz


    # T._refineCell([8,0,1])
    # T._refineCell([8,0,2])
    # T._refineCell([12,0,2])
    # T._refineCell([8,4,2])
    # T._refineCell([6,0,3])
    # T._refineCell([8,8,1])
    # T._refineCell([0,0,0,1])
    # T.__dirty__ = True


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

