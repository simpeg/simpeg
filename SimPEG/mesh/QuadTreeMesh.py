import numpy as np
import matplotlib.pyplot as plt



class TreeNode(object):
    """docstring for TreeNode"""
    def __init__(self, mesh, x0=[0,0], dim=2, depth=0, sz=[1,1], parent=None):
        self.mesh = mesh
        self.x0 = np.array(x0,dtype=float)
        self.dim = dim
        self.depth = depth
        self.sz = np.array(sz,dtype=float)
        self.parent = parent
        self.children = None
        self.numCell = None

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
        if self.isleaf:
            x0, sz = self.x0, self.sz
            corners = [np.r_[x0[0]        , x0[1]        ],
                       np.r_[x0[0]+sz[0]/2, x0[1]        ],
                       np.r_[x0[0]        , x0[1]+sz[1]/2],
                       np.r_[x0[0]+sz[0]/2, x0[1]+sz[1]/2]]

            children = np.array([TreeNode(self.mesh, x0=corners[i],dim=self.dim, depth=self.depth+1, sz=sz/2, parent=self) for i in range(2*self.dim)])
            self.children = children.reshape([2,2],order='F')



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
            ax.plot(corners[:,1],corners[:,0], 'b')
            if self.numCell is not None:
                ax.text(x0[1]+sz[1]/2,x0[0]+sz[0]/2,'%d'%self.numCell)
        else:
            [node.viz(ax) for node in self.children.flatten('F')]




class QuadTreeMesh(object):
    """docstring for QuadTreeMesh"""
    def __init__(self, cells, sz):
        children = range(cells[0])
        for i, row in enumerate(children):
            children[i] = [TreeNode(self, x0=[i*sz[0],j*sz[1]], dim=2, depth=0, sz=sz) for j in range(cells[1])]
        self.children = np.array(children,dtype=TreeNode)


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
        for row in self.children:
            for node in row:
                node.viz(ax)


if __name__ == '__main__':
    M = QuadTreeMesh([2,3],[1,2])
    M.children[0,0].refine()
    M.children[1,1].refine()
    M.children[0,0].children[0,0].refine()
    M.children[0,0].children[0,0].children[0,1].refine()
    M.number()


    M.viz()
    plt.gca().invert_yaxis()
    plt.show()
