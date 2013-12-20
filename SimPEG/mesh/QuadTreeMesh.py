from SimPEG import np, sp, utils
import matplotlib.pyplot as plt



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
        self.x0 = np.array(x0,dtype=float)
        self.faceType = faceType
        self.sz = sz
        self.dim = dim
        self.depth = depth
        mesh.faces.add(self)
        if faceType is 'x': self.mesh.faceX.add(self)
        elif faceType is 'y': self.mesh.faceY.add(self)
        self.tangent = np.zeros(dim)
        self.tangent[1 if faceType is 'x' else 0] = 1
        self.normal = np.zeros(dim)
        self.normal[0 if faceType is 'x' else 1] = 1

    @property
    def isleaf(self): return self.children is None

    @property
    def index(self):
        if self.isleaf: return np.r_[self.numFace]
        return np.concatenate([face.index for face in self.children])

    @property
    def area(self): return self.sz


    def refine(self):
        if not self.isleaf: return
        self.mesh.isNumbered = False

        self.children = np.empty(2,dtype=TreeFace)
        # Create refined x0's
        x0r_0 = self.x0
        x0r_1 = self.x0+0.5*self.tangent*self.sz
        self.children[0] = TreeFace(self.mesh, x0=x0r_0, faceType=self.faceType, dim=self.dim, sz=0.5*self.sz, depth=self.depth+1,parent=self)
        self.children[1] = TreeFace(self.mesh, x0=x0r_1, faceType=self.faceType, dim=self.dim, sz=0.5*self.sz, depth=self.depth+1,parent=self)
        self.mesh.faces.remove(self)
        if self.faceType is 'x':
            self.mesh.faceX.remove(self)
        elif self.faceType is 'y':
            self.mesh.faceY.remove(self)


    def viz(self, ax, text=True):
        if not self.isleaf: return
        ax.plot(np.r_[self.x0[0],self.x0[0]+self.tangent[0]*self.sz], np.r_[self.x0[1], self.x0[1]+self.tangent[1]*self.sz],'r-')
        if text: ax.text(self.x0[0]+0.5*self.tangent[0]*self.sz, self.x0[1]+0.5*self.tangent[1]*self.sz,self.numFace)



class TreeNode(object):
    """docstring for TreeNode"""
    def __init__(self, mesh, x0=[0,0], dim=2, depth=0, sz=[1,1], parent=None, fXm=None, fXp=None, fYm=None, fYp=None):

        fXm = fXm if fXm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='x', dim=dim, sz=sz[1], depth=depth, parent=parent)
        fXp = fXp if fXp is not None else TreeFace(mesh, x0=np.r_[x0[0]+sz[0], x0[1]      ], faceType='x', dim=dim, sz=sz[1], depth=depth, parent=parent)
        fYm = fYm if fYm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='y', dim=dim, sz=sz[0], depth=depth, parent=parent)
        fYp = fYp if fYp is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]+sz[1]], faceType='y', dim=dim, sz=sz[0], depth=depth, parent=parent)

        self.faces = {"fXm":fXm, "fXp":fXp, "fYm":fYm, "fYp":fYp}

        self.mesh = mesh
        self.x0 = np.array(x0, dtype=float)
        self.dim = dim
        self.depth = depth
        self.sz = np.array(sz, dtype=float)
        self.parent = parent
        self.children = None
        self.numCell = None
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
    def gridCC(self): return self.x0 + 0.5*self.sz

    def refine(self, function=None):
        if not self.isleaf and function is None: return

        if function is not None:
            do = function(self.gridCC) > self.depth
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


    def viz(self, ax, text=True):
        if self.isleaf:
            x0, sz = self.x0, self.sz
            corners = np.c_[np.r_[x0[0]      , x0[1]      ],
                            np.r_[x0[0]+sz[0], x0[1]      ],
                            np.r_[x0[0]+sz[0], x0[1]+sz[1]],
                            np.r_[x0[0]      , x0[1]+sz[1]],
                            np.r_[x0[0]      , x0[1]      ]].T
            ax.plot(corners[:,0],corners[:,1], 'b')
            if self.numCell is not None and text:
                ax.text(x0[0]+sz[0]/2,x0[1]+sz[1]/2,'%d'%self.numCell)
        else:
            [node.viz(ax) for node in self.children.flatten('F')]




class QuadTreeMesh(object):
    """docstring for QuadTreeMesh"""
    def __init__(self, h, x0=[0,0]):
        self.faces = set()
        self.faceX = set()
        self.faceY = set()
        self.cells = set()
        self.x0 = np.array(x0,dtype=float)
        self.children = np.empty([hi.size for hi in h],dtype=TreeNode)
        for i in range(h[0].size):
            for j in range(h[1].size):
                fXm = None if i is 0 else self.children[i-1][j].faces['fXp']
                fYm = None if j is 0 else self.children[i][j-1].faces['fYp']
                x0i = (np.r_[x0[0], h[0][:i]]).sum()
                x0j = (np.r_[x0[1], h[1][:j]]).sum()
                self.children[i][j] = TreeNode(self, x0=[x0i, x0j], dim=len(h), depth=0, sz=[h[0][i], h[1][j]], fXm=fXm, fYm=fYm)

    isNumbered = utils.dependentProperty('_isNumbered', False, ['_faceDiv'], 'Setting this to False will delete all operators.')

    @property
    def branchdepth(self):
        return np.max([node.branchdepth for node in self.children.flatten('F')])

    def refine(self, function):
        for node in self.children.flatten():
            node.refine(function)

    def number(self):
        if self.isNumbered: return

        sortedCells = sorted(M.cells,key=SortByX0())
        sortedFaceX = sorted(M.faceX,key=SortByX0())
        sortedFaceY = sorted(M.faceY,key=SortByX0())
        nFx = len(sortedFaceX)
        for i, sc in enumerate(sortedCells): sc.numCell = i
        for i, sfx in enumerate(sortedFaceX): sfx.numFace = i
        for i, sfy in enumerate(sortedFaceY): sfy.numFace = i + nFx

        self.sortedCells = sortedCells
        self.sortedFaceX = sortedFaceX
        self.sortedFaceY = sortedFaceY

        self.isNumbered = True

    def viz(self, ax=None, text=True):
        if ax is None: ax = plt.subplot(111)
        # [node.viz(ax, text=text) for node in self.cells]
        [node.viz(ax, text=text) for node in self.faces]

    @property
    def nC(self): return len(self.cells)
    @property
    def nF(self): return len(self.faces)
    @property
    def nFx(self): return len(self.facesX)
    @property
    def nFy(self): return len(self.facesY)

    @property
    def gridCC(self):
        if getattr(self, '_gridCC', None) is None:
            self.number()
            self._gridCC = np.empty((self.nC,self.dim))
            for ii, cell in enumerate(self.sortedCells):
                self._gridCC[ii,:] = cell.gridCC
        return self._gridCC

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
            self._faceDiv = utils.sdiag(1/VOL)*D*utils.sdiag(S)
        return self._faceDiv



if __name__ == '__main__':
    M = QuadTreeMesh([np.ones(x) for x in [4,10]])

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
    plt.subplot(211)
    plt.spy(DIV)

    M.viz(ax=plt.subplot(212),text=False)
    # plt.gca().invert_yaxis()
    plt.show()
