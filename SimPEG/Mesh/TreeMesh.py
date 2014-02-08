from SimPEG import np, sp, Utils, Solver
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx



def SortByX0():
    eps = 1e-7
    def mycmp(c1,c2):
        if np.abs(c1.x0[1] - c2.x0[1]) < eps:
            return c1.x0[0] - c2.x0[0]
        return c1.x0[1] - c2.x0[1]
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
    return K


class TreeFace(object):
    """docstring for TreeFace"""
    def __init__(self, mesh, x0=[0,0], faceType=None, dim=2, sz=1, depth=0, parent=None):
        self.mesh = mesh
        self.children = None
        self.numFace = None

        self.x0 = np.array(x0, dtype=float)
        self.faceType = faceType
        self.sz = np.array(sz, dtype=float)
        self.dim = dim
        self.depth = depth
        mesh.faces.add(self)
        if faceType is 'x': self.mesh.facesX.add(self)
        elif faceType is 'y': self.mesh.facesY.add(self)
        elif faceType is 'z': self.mesh.facesZ.add(self)
        self.tangent = np.zeros(dim)
        self.tangent[1 if faceType is 'x' else 0] = 1
        self.normal = np.zeros(dim)
        self.normal[0 if faceType is 'x' else 1] = 1

    @property
    def isleaf(self): return self.children is None

    @property
    def index(self):
        if not self.mesh.isNumbered: raise Exception('Mesh is not numbered.')
        if self.isleaf: return np.r_[self.numFace]
        return np.concatenate([face.index for face in self.children])


    @property
    def area(self):
        """area of the face"""
        return self.sz.prod()

    def refine(self):
        if not self.isleaf: return
        self.mesh.isNumbered = False

        self.children = np.empty(2,dtype=TreeFace)
        # Create refined x0's
        x0r_0 = self.x0
        x0r_1 = self.x0+0.5*self.tangent*self.sz
        self.children[0] = TreeFace(self.mesh, x0=x0r_0, faceType=self.faceType, dim=self.dim, sz=0.5*self.sz, depth=self.depth+1, parent=self)
        self.children[1] = TreeFace(self.mesh, x0=x0r_1, faceType=self.faceType, dim=self.dim, sz=0.5*self.sz, depth=self.depth+1, parent=self)
        self.mesh.faces.remove(self)
        if self.faceType is 'x':
            self.mesh.facesX.remove(self)
        elif self.faceType is 'y':
            self.mesh.facesY.remove(self)

    def viz(self, ax, text=True):
        if not self.isleaf: return
        ax.plot(np.r_[self.x0[0],self.x0[0]+self.tangent[0]*self.sz], np.r_[self.x0[1], self.x0[1]+self.tangent[1]*self.sz],'r-')
        if text: ax.text(self.x0[0]+0.5*self.tangent[0]*self.sz, self.x0[1]+0.5*self.tangent[1]*self.sz,self.numFace)

    @property
    def center(self):
        return self.x0 + 0.5*self.tangent*self.sz


class TreeNode(object):
    """docstring for TreeNode"""
    children = None #:
    numCell = None

    def __init__(self, mesh, x0=[0,0], dim=2, depth=0, sz=[1,1], parent=None, fXm=None, fXp=None, fYm=None, fYp=None, fZm=None, fZp=None):

        self.mesh = mesh
        self.x0 = np.array(x0, dtype=float)
        self.sz = np.array(sz, dtype=float)
        self.dim = dim
        self.depth = depth
        self.parent = parent
        if dim == 2:
            fXm = fXm if fXm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='x', dim=dim, sz=np.r_[sz[1]], depth=depth, parent=parent)
            fXp = fXp if fXp is not None else TreeFace(mesh, x0=np.r_[x0[0]+sz[0], x0[1]      ], faceType='x', dim=dim, sz=np.r_[sz[1]], depth=depth, parent=parent)
            fYm = fYm if fYm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='y', dim=dim, sz=np.r_[sz[0]], depth=depth, parent=parent)
            fYp = fYp if fYp is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]+sz[1]], faceType='y', dim=dim, sz=np.r_[sz[0]], depth=depth, parent=parent)
            self.faces = {"fXm":fXm, "fXp":fXp, "fYm":fYm, "fYp":fYp}

        elif dim == 3:
            fXm = fXm if fXm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]      ], faceType='x', dim=dim, sz=np.r_[sz[1], sz[2]], depth=depth, parent=parent)
            fXp = fXp if fXp is not None else TreeFace(mesh, x0=np.r_[x0[0]+sz[0], x0[1]      , x0[2]      ], faceType='x', dim=dim, sz=np.r_[sz[1], sz[2]], depth=depth, parent=parent)
            fYm = fYm if fYm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]      ], faceType='y', dim=dim, sz=np.r_[sz[0], sz[2]], depth=depth, parent=parent)
            fYp = fYp if fYp is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]+sz[1], x0[2]      ], faceType='y', dim=dim, sz=np.r_[sz[0], sz[2]], depth=depth, parent=parent)
            fZm = fZm if fZm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]      ], faceType='z', dim=dim, sz=np.r_[sz[0], sz[1]], depth=depth, parent=parent)
            fZp = fZp if fZp is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]+sz[2]], faceType='z', dim=dim, sz=np.r_[sz[0], sz[1]], depth=depth, parent=parent)
            self.faces = {"fXm":fXm, "fXp":fXp, "fYm":fYm, "fYp":fYp, "fZm":fZm, "fZp":fZp}

        mesh.cells.add(self)

    @property
    def parent(self):
        return self._parent
    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def branchdepth(self):
        if self.isleaf:
            return self.depth
        else:
            return np.max([node.branchdepth for node in self.children.flatten('F')])

    @property
    def center(self): return self.x0 + 0.5*self.sz

    def refine(self, function=None):
        if self.dim == 2:
            return self._refine2D(function=function)

    def _refine2D(self, function=None):
        if not self.isleaf and function is None: return

        if function is not None:
            do = function(self.center) > self.depth
            if not do: return

        self.mesh.isNumbered = False

        self.children = np.empty((2,2),dtype=TreeNode)
        x0, sz = self.x0, self.sz

        for faceName in self.faces:
            self.faces[faceName].refine()

        i, j = 0, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.faces['fXm'].children[0], None, self.faces['fYm'].children[0], None
        self.children[i,j] = TreeNode(self.mesh, x0=x0r,dim=self.dim, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        i, j = 1, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.children[0,0].faces['fXp'], self.faces['fXp'].children[0], self.faces['fYm'].children[1], None
        self.children[i,j] = TreeNode(self.mesh, x0=x0r,dim=self.dim, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        i, j = 0, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.faces['fXm'].children[1], None, self.children[0,0].faces['fYp'], self.faces['fYp'].children[0]
        self.children[i,j] = TreeNode(self.mesh, x0=x0r,dim=self.dim, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        i, j = 1, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.children[0,1].faces['fXp'], self.faces['fXp'].children[1], self.children[1,0].faces['fYp'], self.faces['fYp'].children[1]
        self.children[i,j] = TreeNode(self.mesh, x0=x0r,dim=self.dim, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        self.mesh.cells.remove(self)

        # pass the refine function to the children
        if function is not None:
            for child in self.children.flatten():
                child.refine(function)

    @property
    def isleaf(self): return self.children is None

    @property
    def faceIndex(self):
        I, J, V = np.empty(0,dtype=float), np.empty(0,dtype=float), np.empty(0,dtype=float)
        for face in self.faces:
            j = self.faces[face].index
            i = j*0+self.numCell
            v = j*0+1
            if 'p' in face:
                v *= -1
            I, J, V = np.r_[I,i], np.r_[J,j], np.r_[V,v]
        return I, J, V

    @property
    def vol(self): return self.sz.prod()


    def viz(self, ax, color='none', text=False):
        if not self.isleaf: return
        x0, sz = self.x0, self.sz
        ax.add_patch(plt.Rectangle((x0[0], x0[1]), sz[0], sz[1], facecolor=color, edgecolor='k'))
        if text: ax.text(self.center[0],self.center[1],self.numCell)




class TreeMesh(object):
    """TreeMesh"""
    def __init__(self, h, x0=None):

        assert type(h) is list, 'h must be a list'

        self.h = h
        if x0 is None:
            x0 = np.zeros(self.dim)
        else:
            assert type(x0) in [list, np.ndarray], 'x0 must be a numpy array or a list'
            assert len(x0) == self.dim, 'x0 must have the same dimensions as the mesh'
        self.x0 = np.array(x0, dtype=float)

        # set the sets for holding the faces and cells
        self.cells = set()
        self.faces = set()
        self.facesX = set()
        self.facesY = set()
        if self.dim == 3: self.facesZ = set()

        self.children = np.empty([hi.size for hi in h],dtype=TreeNode)
        for i in range(h[0].size):
            for j in range(h[1].size):
                fXm = None if i is 0 else self.children[i-1][j].faces['fXp']
                fYm = None if j is 0 else self.children[i][j-1].faces['fYp']
                x0i = (np.r_[x0[0], h[0][:i]]).sum()
                x0j = (np.r_[x0[1], h[1][:j]]).sum()
                self.children[i][j] = TreeNode(self, x0=[x0i, x0j], dim=len(h), depth=0, sz=[h[0][i], h[1][j]], fXm=fXm, fYm=fYm)

    isNumbered = Utils.dependentProperty('_isNumbered', False, ['_faceDiv'], 'Setting this to False will delete all operators.')

    @property
    def branchdepth(self):
        return np.max([node.branchdepth for node in self.children.flatten('F')])

    def refine(self, function):
        for node in self.children.flatten():
            node.refine(function)

    def number(self):
        if self.isNumbered: return

        self.sortedCells = sorted(self.cells,key=SortByX0())
        for i, sc in enumerate(self.sortedCells): sc.numCell = i

        self.sortedFaceX = sorted(self.facesX,key=SortByX0())
        for i, sfx in enumerate(self.sortedFaceX): sfx.numFace = i

        self.sortedFaceY = sorted(self.facesY,key=SortByX0())
        for i, sfy in enumerate(self.sortedFaceY): sfy.numFace = i + self.nFx

        if self.dim == 3:
            self.sortedFaceZ = sorted(self.facesZ,key=SortByX0())
            for i, sfz in enumerate(self.sortedFaceZ): sfz.numFace = i + self.nFx + self.nFy

        self.isNumbered = True

    @property
    def dim(self): return len(self.h)

    @property
    def nC(self): return len(self.cells)

    @property
    def nF(self): return len(self.faces)

    @property
    def nFx(self): return len(self.facesX)

    @property
    def nFy(self): return len(self.facesY)

    @property
    def nFz(self): return len(self.facesZ)

    @property
    def nE(self): return len(self.faces)

    @property
    def nEx(self):
        if self.dim == 2:
            return len(self.facesY)
        else: raise NotImplementedError('nEx')

    @property
    def nEy(self):
        if self.dim == 2:
            return len(self.facesX)
        else: raise NotImplementedError('nEy')

    @property
    def gridCC(self):
        if getattr(self, '_gridCC', None) is None:
            self.number()
            self._gridCC = np.empty((self.nC,self.dim))
            for ii, cell in enumerate(self.sortedCells):
                self._gridCC[ii,:] = cell.center
        return self._gridCC

    @property
    def gridFx(self):
        if getattr(self, '_gridFx', None) is None:
            self.number()
            self._gridFx = np.empty((self.nFx,self.dim))
            for ii, face in enumerate(self.sortedFaceX):
                self._gridFx[ii,:] = face.center
        return self._gridFx

    @property
    def gridFy(self):
        if getattr(self, '_gridFy', None) is None:
            self.number()
            self._gridFy = np.empty((self.nFy,self.dim))
            for ii, face in enumerate(self.sortedFaceY):
                self._gridFy[ii,:] = face.center
        return self._gridFy

    @property
    def vol(self):
        self.number()
        return np.array([cell.vol for cell in self.sortedCells])

    @property
    def area(self):
        self.number()
        return np.concatenate(([face.area for face in self.sortedFaceX],[face.area for face in self.sortedFaceY]))

    @property
    def faceDiv(self):
        if getattr(self, '_faceDiv', None) is None:
            self.number()
            I, J, V = np.empty(0), np.empty(0), np.empty(0)
            for cell in M.sortedCells:
                i, j, v = cell.faceIndex
                I, J, V = np.r_[I,i], np.r_[J,j], np.r_[V,v]

            VOL = self.vol
            D = sp.csr_matrix((V,(I,J)), shape=(M.nC, M.nF))
            S = self.area
            self._faceDiv = Utils.sdiag(1/VOL)*D*Utils.sdiag(S)
        return self._faceDiv


    def plotGrid(self, ax=None, text=True, plotC=True, plotF=False, showIt=False):
        if ax is None: ax = plt.subplot(111)

        if plotC: [node.viz(ax, text=text) for node in self.cells]
        if plotF: [node.viz(ax, text=text) for node in self.faces]
        ax.set_xlim((self.x0[0], self.h[0].sum()))
        ax.set_ylim((self.x0[1], self.h[1].sum()))
        if showIt: plt.show()


    def plotImage(self, I, ax=None, showIt=True):
        if self.dim == 2:
            self._plotImage2D(I, ax=ax, showIt=showIt)
        elif self.dim == 3:
            raise NotImplementedError('3D visualization is not yet implemented.')

    def _plotImage2D(self, I, ax=None, showIt=True):
        if ax is None: ax = plt.subplot(111)
        jet = cm = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=I.min(), vmax=I.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        ax.set_xlim((self.x0[0], self.h[0].sum()))
        ax.set_ylim((self.x0[1], self.h[1].sum()))
        for ii, node in enumerate(self.sortedCells):
            node.viz(ax=ax, color=scalarMap.to_rgba(I[ii]))
        scalarMap._A = []  # http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
        plt.colorbar(scalarMap)
        if showIt: plt.show()



if __name__ == '__main__':
    M = TreeMesh([np.ones(x) for x in [4,10]])

    def function(xc):
        r = xc - np.r_[2.,6.]
        dist = np.sqrt(r.dot(r))
        if dist < 1.0:
            return 3
        if dist < 1.5:
            return 2
        else:
            return 1

    M.refine(function)

    DIV = M.faceDiv
    # plt.subplot(211)
    # plt.spy(DIV)
    M.plotGrid(ax=plt.subplot(111),text=True)

    q = np.zeros(M.nC)
    q[208] = -1.0
    q[291] = 1.0
    b = Solver(-DIV*DIV.T).solve(q)
    plt.figure()
    M.plotImage(b)
    # plt.gca().invert_yaxis()
    print M.vol
    plt.show()
