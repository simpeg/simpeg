import numpy as np
import matplotlib.pyplot as plt


class TreeFace(object):
    """docstring for TreeFace"""
    def __init__(self, mesh, x0=[0,0], faceType=None, dim=2, sz=1, depth=0, parent=None):
        self.mesh = mesh
        self.children = None
        self.x0 = np.array(x0,dtype=float)
        self.faceType = faceType
        self.sz = sz
        self.dim = dim
        mesh.faces.add(self)
        self.tangent = np.zeros(dim)
        self.tangent[1 if faceType is 'x' else 0] = 1
        self.normal = np.zeros(dim)
        self.normal[0 if faceType is 'x' else 1] = 1

    @property
    def isleaf(self): return self.children is None

    def refine(self):
        if not self.isleaf: return
        self.children = np.empty(2,dtype=TreeFace)
        # Create refined x0's
        x0r_0 = self.x0
        x0r_1 = self.x0+self.tangent*self.sz/2
        self.children[0] = TreeFace(self.mesh, x0=x0r_0, faceType=self.faceType, dim=self.dim, sz=self.sz/2, depth=self.depth+1,parent=self)
        self.children[1] = TreeFace(self.mesh, x0=x0r_1, faceType=self.faceType, dim=self.dim, sz=self.sz/2, depth=self.depth+1,parent=self)
        self.mesh.faces.remove(self)


    def viz(self, ax):
        ax.plot(np.r_[self.x0[0],self.x0[0]+self.tangent[0]*self.sz], np.r_[self.x0[1], self.x0[1]+self.tangent[1]*self.sz],'rx-')


class TreeNode(object):
    """docstring for TreeNode"""
    def __init__(self, mesh, x0=[0,0], dim=2, depth=0, sz=[1,1], parent=None, fXm=None, fXp=None, fYm=None, fYp=None):

        self.fXm = fXm if fXm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='x', dim=dim, sz=sz[1], depth=depth, parent=parent)
        self.fXp = fXp if fXp is not None else TreeFace(mesh, x0=np.r_[x0[0]+sz[0], x0[1]      ], faceType='x', dim=dim, sz=sz[1], depth=depth, parent=parent)
        self.fYm = fYm if fYm is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='y', dim=dim, sz=sz[0], depth=depth, parent=parent)
        self.fYp = fYp if fYp is not None else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]+sz[1]], faceType='y', dim=dim, sz=sz[0], depth=depth, parent=parent)

        self.mesh = mesh
        self.x0 = np.array(x0,dtype=float)
        self.dim = dim
        self.depth = depth
        self.sz = np.array(sz,dtype=float)
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


    def refine(self):
        if not self.isleaf: return

        x0, sz = self.x0, self.sz
        corners = [np.r_[x0[0]        , x0[1]        ],
                   np.r_[x0[0]+sz[0]/2, x0[1]        ],
                   np.r_[x0[0]        , x0[1]+sz[1]/2],
                   np.r_[x0[0]+sz[0]/2, x0[1]+sz[1]/2]]

        children = [TreeNode(self.mesh, x0=corners[i],dim=self.dim, depth=self.depth+1, sz=sz/2, parent=self) for i in range(2*self.dim)]
        self.mesh.cells.remove(self)
        self.mesh.cells = self.mesh.cells.union(set(children))
        self.children = np.array(children,dtype=TreeNode).reshape([2,2],order='F')



    def countCell(self, COUNTER, column, maxColumns):
        if self.isleaf:
            if self.numCell is None:
                self.numCell = COUNTER
                return COUNTER + 1
            else:
                return COUNTER
        if column < maxColumns/2:
            COUNTER = self.children[0,0].countCell(COUNTER, column, maxColumns/2)
            COUNTER = self.children[1,0].countCell(COUNTER, column, maxColumns/2)
            return COUNTER
        else:
            column = column - maxColumns/2
            COUNTER = self.children[0,1].countCell(COUNTER, column, maxColumns/2)
            COUNTER = self.children[1,1].countCell(COUNTER, column, maxColumns/2)
            return COUNTER


    @property
    def isleaf(self): return self.children is None

    def viz(self, ax):
        if self.isleaf:
            x0, sz = self.x0, self.sz
            corners = np.c_[np.r_[x0[0]      , x0[1]      ],
                            np.r_[x0[0]+sz[0], x0[1]      ],
                            np.r_[x0[0]+sz[0], x0[1]+sz[1]],
                            np.r_[x0[0]      , x0[1]+sz[1]],
                            np.r_[x0[0]      , x0[1]      ]].T
            ax.plot(corners[:,0],corners[:,1], 'b')
            self.fXm.viz(ax)
            if self.numCell is not None:
                ax.text(x0[0]+sz[0]/2,x0[1]+sz[1]/2,'%d'%self.numCell)
        else:
            [node.viz(ax) for node in self.children.flatten('F')]




class QuadTreeMesh(object):
    """docstring for QuadTreeMesh"""
    def __init__(self, cells, sz):
        self.faces = set()
        self.cells = set()

        self.children = np.empty(cells,dtype=TreeNode)
        for i in range(cells[0]):
            for j in range(cells[1]):
                fXm = None if i is 0 else self.children[i-1][j].fXp
                fYm = None if j is 0 else self.children[i][j-1].fYp
                self.children[i][j] = TreeNode(self, x0=[i*sz[0],j*sz[1]], dim=2, depth=0, sz=sz, fXm=fXm, fYm=fYm)


    @property
    def branchdepth(self):
        return np.max([node.branchdepth for node in self.children.flatten('F')])


    def number(self):
        COUNTER = 0
        for col in range(self.children.shape[1]):
            coldepth = np.max([node.branchdepth for node in self.children[:,col]])
            maxColumns = 2**coldepth
            for COL in range(maxColumns):
                for row in range(self.children.shape[0]):
                    COUNTER = self.children[row,col].countCell(COUNTER, COL, maxColumns)

    def viz(self):
        ax = plt.subplot(111)
        [node.viz(ax) for node in self.cells]
        [node.viz(ax) for node in self.faces]


if __name__ == '__main__':
    M = QuadTreeMesh([4,6],[1,2])
    M.children[0,0].refine()
    M.children[1,1].refine()
    for ii in range(3):
        M.children[ii,ii].refine()
        M.children[ii,ii].children[0,0].refine()


    eps = 1e-7
    def mycmp(c1,c2):
        if np.abs(c1.x0[1] - c2.x0[1]) < eps:
            return c1.x0[0] - c2.x0[0]
        return c1.x0[1] - c2.x0[1]

    def cmp_to_key(mycmp):
        'Convert a cmp= function into a key= function'
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

    # M.number()
    sortCells = sorted(M.cells,key=cmp_to_key(mycmp))
    for i, sc in enumerate(sortCells):
        sc.numCell = i

    print M.children[0,0].fXp is M.children[1,0].fXm
    M.viz()
    # plt.gca().invert_yaxis()
    plt.show()
