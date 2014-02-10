from SimPEG import np, sp, Utils, Solver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx



def SortByX0():
    eps = 1e-7
    def mycmp(c1,c2):
        if c1.x0.size == 2:
            if np.abs(c1.x0[1] - c2.x0[1]) < eps:
                return c1.x0[0] - c2.x0[0]
            return c1.x0[1] - c2.x0[1]
        elif c1.x0.size == 3:
            if np.abs(c1.x0[2] - c2.x0[2]) < eps:
                if np.abs(c1.x0[1] - c2.x0[1]) < eps:
                    return c1.x0[0] - c2.x0[0]
                return c1.x0[1] - c2.x0[1]
            return c1.x0[2] - c2.x0[2]

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


class TreeObject(object):
    """docstring for TreeObject"""

    children = None #: children of the tree object
    num = None

    def __init__(self, mesh, parent):
        self.mesh = mesh
        self._parent = parent

    @property
    def parent(self): return self._parent

    @property
    def dim(self): return self.mesh.dim

    @property
    def isleaf(self): return self.children is None

    @property
    def center(self): return self.x0


class TreeNode(TreeObject):
    """docstring for TreeNode"""
    def __init__(self, mesh, x0=[0,0], depth=0, parent=None):
        TreeObject.__init__(self, mesh, parent)
        self.x0 = np.array(x0, dtype=float)
        self.mesh.nodes.add(self)


class TreeEdge(TreeObject):
    """docstring for TreeEdge"""
    def __init__(self, mesh, x0=[0,0], edgeType=None, sz=[1,], depth=0,
                 node0=None, node1=None,
                 parent=None):
        TreeObject.__init__(self, mesh, parent)

        self.x0 = np.array(x0, dtype=float)
        self.depth = depth
        self.edgeType = edgeType
        self.sz = np.array(sz, dtype=float)

        mesh.edges.add(self)
        if   edgeType is 'x': mesh.edgesX.add(self)
        elif edgeType is 'y': mesh.edgesY.add(self)
        elif edgeType is 'z': mesh.edgesZ.add(self)

        self.node0 = node0
        self.node1 = node1


class TreeFace(TreeObject):
    """docstring for TreeFace"""
    def __init__(self, mesh, x0=[0,0], faceType=None, sz=[1,], depth=0,
                 node0=None, node1=None, node2=None, node3=None,
                 parent=None):
        TreeObject.__init__(self, mesh, parent)

        self.x0 = np.array(x0, dtype=float)
        self.depth = depth
        self.faceType = faceType
        self.sz = np.array(sz, dtype=float)

        mesh.faces.add(self)
        if   faceType is 'x': mesh.facesX.add(self)
        elif faceType is 'y': mesh.facesY.add(self)
        elif faceType is 'z': mesh.facesZ.add(self)
        # Add the nodes:
        self.node0 = node0 if isinstance(node0,TreeNode) else TreeNode(mesh, x0=self.x0)
        self.node1 = node1 if isinstance(node1,TreeNode) else TreeNode(mesh, x0=self.x0 + self.tangent0*self.sz[0])
        if self.dim == 3:
            self.node2 = node2 if isinstance(node1,TreeNode) else TreeNode(mesh, x0=self.x0 + self.tangent1*self.sz[1])
            self.node3 = node3 if isinstance(node1,TreeNode) else TreeNode(mesh, x0=self.x0 + self.tangent0*self.sz[0] + self.tangent1*self.sz[1])

    @property
    def tangent0(self):
        if   self.faceType is 'x': t = np.r_[0,1.,0]
        elif self.faceType is 'y': t = np.r_[1.,0,0]
        elif self.faceType is 'z': t = np.r_[1.,0,0]
        return t[:self.dim]

    @property
    def tangent1(self):
        if self.dim == 2: return
        if   self.faceType is 'x': t = np.r_[0,0,1.]
        elif self.faceType is 'y': t = np.r_[0,0,1.]
        elif self.faceType is 'z': t = np.r_[0,1.,0]
        return t[:self.dim]

    @property
    def normal(self):
        if   self.faceType is 'x': n = np.r_[1.,0,0]
        elif self.faceType is 'y': n = np.r_[0,1.,0]
        elif self.faceType is 'z': n = np.r_[0,0,1.]
        return n[:self.dim]

    @property
    def index(self):
        if not self.mesh.isNumbered: raise Exception('Mesh is not numbered.')
        if self.isleaf: return np.r_[self.num]
        return np.concatenate([face.index for face in self.children])

    @property
    def area(self):
        """area of the face"""
        return self.sz.prod()

    def refine(self):
        if not self.isleaf: return
        self.mesh.isNumbered = False
        if self.dim == 2:
            self._refine2D()
        elif self.dim == 3:
            self._refine3D()

    def _refine2D(self):
        self.children = np.empty(2,dtype=TreeFace)
        # Create refined x0's
        x0r_0 = self.x0
        x0r_1 = self.x0+0.5*self.tangent0*self.sz
        self.children[0] = TreeFace(self.mesh, x0=x0r_0, faceType=self.faceType, sz=0.5*self.sz, depth=self.depth+1, parent=self, node0=self.node0)
        self.children[1] = TreeFace(self.mesh, x0=x0r_1, faceType=self.faceType, sz=0.5*self.sz, depth=self.depth+1, parent=self, node0=self.children[0].node1, node1=self.node1)
        self.mesh.faces.remove(self)
        if self.faceType is 'x':
            self.mesh.facesX.remove(self)
        elif self.faceType is 'y':
            self.mesh.facesY.remove(self)

    def _refine3D(self):
        self.children = np.empty((2,2),dtype=TreeFace)
        # Create refined x0's
        x0r_0 = self.x0
        x0r_1 = self.x0+0.5*self.tangent0*self.sz[0]
        x0r_2 = self.x0+0.5*self.tangent1*self.sz[1]
        x0r_3 = self.x0+0.5*self.tangent0*self.sz[0]+0.5*self.tangent1*self.sz[1]

        # TODO: Set nodes
        self.children[0,0] = TreeFace(self.mesh, x0=x0r_0, faceType=self.faceType, sz=0.5*self.sz, depth=self.depth+1, parent=self)
        self.children[1,0] = TreeFace(self.mesh, x0=x0r_1, faceType=self.faceType, sz=0.5*self.sz, depth=self.depth+1, parent=self)
        self.children[0,1] = TreeFace(self.mesh, x0=x0r_2, faceType=self.faceType, sz=0.5*self.sz, depth=self.depth+1, parent=self)
        self.children[1,1] = TreeFace(self.mesh, x0=x0r_3, faceType=self.faceType, sz=0.5*self.sz, depth=self.depth+1, parent=self)
        self.mesh.faces.remove(self)
        if self.faceType is 'x':
            self.mesh.facesX.remove(self)
        elif self.faceType is 'y':
            self.mesh.facesY.remove(self)
        elif self.faceType is 'z':
            self.mesh.facesZ.remove(self)

    def plotGrid(self, ax, text=True):
        if not self.isleaf: return
        if self.dim == 2:
            line = np.c_[self.node0.x0, self.node1.x0].T
            ax.plot(line[:,0], line[:,1],'r-')
            if text: ax.text(self.center[0], self.center[1],self.num)
        elif self.dim == 3:
            line = np.c_[self.node0.x0, self.node1.x0, self.node3.x0, self.node2.x0, self.node0.x0].T
            ax.plot(line[:,0], line[:,1],'r-', zs=line[:,2])
            if text: ax.text(self.center[0], self.center[1], self.center[2], self.num)

    @property
    def center(self):
        if self.dim == 2:
            return self.x0 + 0.5*self.tangent0*self.sz[0]
        elif self.dim == 3:
            return self.x0 + 0.5*self.tangent0*self.sz[0] + 0.5*self.tangent1*self.sz[1]


class TreeCell(TreeObject):
    """docstring for TreeCell"""
    children = None #:

    def __init__(self, mesh, x0=[0,0], depth=0, sz=[1,1],
                 fXm=None, fXp=None,
                 fYm=None, fYp=None,
                 fZm=None, fZp=None,
                 parent=None):
        TreeObject.__init__(self, mesh, parent)

        self.x0 = np.array(x0, dtype=float)
        self.sz = np.array(sz, dtype=float)
        self.depth = depth
        if self.dim == 2:
            #
            #      2___________3
            #      |    fYp    |
            #      |           |
            #   fXm|     x     |fXp      y
            #      |           |         ^
            #      |___________|         |___> x
            #      0    fYm    1
            #
            N = {}
            N["n0"] = getattr(fXm, 'node0', None) or getattr(fYm, 'node0', None)
            N["n1"] = getattr(fXp, 'node0', None) or getattr(fYm, 'node1', None)
            N["n2"] = getattr(fXm, 'node1', None) or getattr(fYp, 'node0', None)
            N["n3"] = getattr(fXp, 'node1', None) or getattr(fYp, 'node1', None)

            fXm = fXm if isinstance(fXm, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='x', sz=np.r_[sz[1]], depth=depth, parent=parent, node0=N['n0'], node1=N['n2'])
            N["n0"], N["n2"] = fXm.node0, fXm.node1

            fXp = fXp if isinstance(fXp, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]+sz[0], x0[1]      ], faceType='x', sz=np.r_[sz[1]], depth=depth, parent=parent, node0=N['n1'], node1=N['n3'])
            N["n1"], N["n3"] = fXp.node0, fXp.node1

            fYm = fYm if isinstance(fYm, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      ], faceType='y', sz=np.r_[sz[0]], depth=depth, parent=parent, node0=N['n0'], node1=N['n1'])
            N["n0"], N["n1"] = fYm.node0, fYm.node1

            fYp = fYp if isinstance(fYp, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]+sz[1]], faceType='y', sz=np.r_[sz[0]], depth=depth, parent=parent, node0=N['n2'], node1=N['n3'])
            N["n2"], N["n3"] = fYp.node0, fYp.node1

            self.faces = {"fXm":fXm, "fXp":fXp, "fYm":fYm, "fYp":fYp}
            self.nodes = N

        elif self.dim == 3:
            #                      fZp
            #                       |
            #                 6 --------------- 7
            #                /|     |         / |
            #               / |     .        /  |
            #              /  |        fYp  /   |
            #             /   |            / fXp|
            #            4 -------------- 5     |
            #            |fXm 2 ----------|---- 3          z
            #            |   /            |   /            ^   y
            #            |  /  fYm  .     |  /             |  /
            #            | /        |     | /              | /
            #            0 -------------- 1                o----> x
            #                       |
            #                      fZm
            #
            #
            #            fX                                  fY                                 fZ
            #      2___________3                       2___________3                      2___________3
            #      |    eYp    |                       |    eXp    |                      |    eXp    |
            #      |           |                       |           |                      |           |
            #   eZm|     x     |eZp      z          eZm|     x     |eZp      z         eYm|     x     |eYp      y
            #      |           |         ^             |           |         ^            |           |         ^
            #      |___________|         |___> y       |___________|         |___> x      |___________|         |___> x
            #      0    eYm    1                       0    eXm    1                      0    eXm    1
            #
            #  Mappings: numOnFace > numOnCell
            #
            #  fXm 0>0, 1>2, 2>4, 3>6              fYm 0>0, 1>1, 2>4, 3>5             fZm 0>0, 1>1, 2>2, 3>3
            #  fXp 0>1, 1>3, 2>5, 3>7              fYp 0>2, 1>3, 2>6, 3>7             fZp 0>4, 1>5, 2>6, 3>7

            N = {}
            N["n0"] = getattr(fXm, 'node0', None) or getattr(fYm, 'node0', None) or getattr(fZm, 'node0', None)
            N["n1"] = getattr(fXp, 'node0', None) or getattr(fYm, 'node1', None) or getattr(fZm, 'node1', None)
            N["n2"] = getattr(fXm, 'node1', None) or getattr(fYp, 'node0', None) or getattr(fZm, 'node2', None)
            N["n3"] = getattr(fXp, 'node1', None) or getattr(fYp, 'node1', None) or getattr(fZm, 'node3', None)
            N["n4"] = getattr(fXm, 'node2', None) or getattr(fYm, 'node2', None) or getattr(fZp, 'node0', None)
            N["n5"] = getattr(fXp, 'node2', None) or getattr(fYm, 'node3', None) or getattr(fZp, 'node1', None)
            N["n6"] = getattr(fXm, 'node3', None) or getattr(fYp, 'node2', None) or getattr(fZp, 'node2', None)
            N["n7"] = getattr(fXp, 'node3', None) or getattr(fYp, 'node3', None) or getattr(fZp, 'node3', None)


            fXm = fXm if isinstance(fXm, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]      ], faceType='x', sz=np.r_[sz[1], sz[2]], depth=depth, parent=parent, node0=N["n0"], node1=N["n2"], node2=N["n4"], node3=N["n6"])
            N["n0"], N["n2"], N["n4"], N["n6"] = fXm.node0, fXm.node1, fXm.node2, fXm.node3

            fXp = fXp if isinstance(fXp, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]+sz[0], x0[1]      , x0[2]      ], faceType='x', sz=np.r_[sz[1], sz[2]], depth=depth, parent=parent, node0=N["n1"], node1=N["n3"], node2=N["n5"], node3=N["n7"])
            N["n1"], N["n3"], N["n5"], N["n7"] = fXp.node0, fXp.node1, fXp.node2, fXp.node3

            fYm = fYm if isinstance(fYm, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]      ], faceType='y', sz=np.r_[sz[0], sz[2]], depth=depth, parent=parent, node0=N["n0"], node1=N["n1"], node2=N["n4"], node3=N["n5"])
            N["n0"], N["n1"], N["n4"], N["n5"] = fYm.node0, fYm.node1, fYm.node2, fYm.node3

            fYp = fYp if isinstance(fYp, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]+sz[1], x0[2]      ], faceType='y', sz=np.r_[sz[0], sz[2]], depth=depth, parent=parent, node0=N["n2"], node1=N["n3"], node2=N["n6"], node3=N["n7"])
            N["n2"], N["n3"], N["n6"], N["n7"] = fYp.node0, fYp.node1, fYp.node2, fYp.node3

            fZm = fZm if isinstance(fZm, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]      ], faceType='z', sz=np.r_[sz[0], sz[1]], depth=depth, parent=parent, node0=N["n0"], node1=N["n1"], node2=N["n2"], node3=N["n3"])
            N["n0"], N["n1"], N["n2"], N["n3"] = fZm.node0, fZm.node1, fZm.node2, fZm.node3

            fZp = fZp if isinstance(fZp, TreeFace) else TreeFace(mesh, x0=np.r_[x0[0]      , x0[1]      , x0[2]+sz[2]], faceType='z', sz=np.r_[sz[0], sz[1]], depth=depth, parent=parent, node0=N["n4"], node1=N["n5"], node2=N["n6"], node3=N["n7"])
            N["n4"], N["n5"], N["n6"], N["n7"] = fZp.node0, fZp.node1, fZp.node2, fZp.node3

            self.faces = {"fXm":fXm, "fXp":fXp, "fYm":fYm, "fYp":fYp, "fZm":fZm, "fZp":fZp}

        mesh.cells.add(self)

    @property
    def branchdepth(self):
        if self.isleaf:
            return self.depth
        else:
            return np.max([node.branchdepth for node in self.children.flatten('F')])

    @property
    def center(self): return self.x0 + 0.5*self.sz

    def refine(self, function=None):
        if not self.isleaf and function is None: return

        if function is not None:
            do = function(self.center) > self.depth
            if not do: return

        if self.dim == 2:
            return self._refine2D()
        elif self.dim == 3:
            return self._refine3D()

        # pass the refine function to the children
        if function is not None:
            for child in self.children.flatten():
                child.refine(function)

    def _refine2D(self):

        self.mesh.isNumbered = False

        self.children = np.empty((2,2),dtype=TreeCell)
        x0, sz = self.x0, self.sz

        for faceName in self.faces:
            self.faces[faceName].refine()

        i, j = 0, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.faces['fXm'].children[0], None, self.faces['fYm'].children[0], None
        self.children[i,j] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        i, j = 1, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.children[0,0].faces['fXp'], self.faces['fXp'].children[0], self.faces['fYm'].children[1], None
        self.children[i,j] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        i, j = 0, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.faces['fXm'].children[1], None, self.children[0,0].faces['fYp'], self.faces['fYp'].children[0]
        self.children[i,j] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        i, j = 1, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1]]
        fXm, fXp, fYm, fYp = self.children[0,1].faces['fXp'], self.faces['fXp'].children[1], self.children[1,0].faces['fYp'], self.faces['fYp'].children[1]
        self.children[i,j] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp)

        self.mesh.cells.remove(self)


    def _refine3D(self):

        self.mesh.isNumbered = False

        self.children = np.empty((2,2,2),dtype=TreeCell)
        x0, sz = self.x0, self.sz

        for faceName in self.faces:
            self.faces[faceName].refine()

        i, j, k = 0, 0, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        i, j, k = 1, 0, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        i, j, k = 0, 1, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        i, j, k = 1, 1, 0
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        # Upper layer
        i, j, k = 0, 0, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        i, j, k = 1, 0, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        i, j, k = 0, 1, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        i, j, k = 1, 1, 1
        x0r = np.r_[x0[0] + 0.5*i*sz[0], x0[1] + 0.5*j*sz[1], x0[2] + 0.5*k*sz[2]]
        fXm, fXp, fYm, fYp, fZm, fZp =  None, None, None, None, None, None
        self.children[i,j,k] = TreeCell(self.mesh, x0=x0r, depth=self.depth+1, sz=0.5*sz, parent=self, fXm=fXm, fXp=fXp, fYm=fYm, fYp=fYp, fZm=fZm, fZp=fZp)

        self.mesh.cells.remove(self)

    @property
    def faceIndex(self):
        #TODO: preallocate
        I, J, V = np.empty(0,dtype=float), np.empty(0,dtype=float), np.empty(0,dtype=float)
        for face in self.faces:
            j = self.faces[face].index
            i = j*0+self.num
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
        if text: ax.text(self.center[0],self.center[1],self.num)

    def plotGrid(self, ax, text=False):
        if not self.isleaf: return
        if self.dim == 2:
            ax.plot(self.center[0],self.center[1],'b.')
            if text: ax.text(self.center[0],self.center[1],self.num)
        elif self.dim == 3:
            ax.plot([self.center[0]],[self.center[1]],'b.', zs=[self.center[2]])
            if text: ax.text(self.center[0], self.center[1], self.center[2], self.num)


class TreeMesh(object):
    """TreeMesh"""
    def __init__(self, h_in, x0=None):
        assert type(h_in) is list, 'h_in must be a list'
        h = range(len(h_in))
        for i, h_i in enumerate(h_in):
            if type(h_i) in [int, long, float]:
                # This gives you something over the unit cube.
                h_i = np.ones(int(h_i))/int(h_i)
            assert type(h_i) == np.ndarray, ("h[%i] is not a numpy array." % i)
            assert len(h_i.shape) == 1, ("h[%i] must be a 1D numpy array." % i)
            h[i] = h_i[:] # make a copy.
        self.h = h

        if x0 is None:
            x0 = np.zeros(self.dim)
        else:
            assert type(x0) in [list, np.ndarray], 'x0 must be a numpy array or a list'
            x0 = np.array(x0, dtype=float)
            assert len(x0) == self.dim, 'x0 must have the same dimensions as the mesh'
        self.x0 = x0

        # set the sets for holding the cells, nodes, faces, and edges
        self.cells  = set()
        self.nodes  = set()
        self.faces  = set()
        self.facesX = set()
        self.facesY = set()
        if self.dim == 3:
            self.facesZ = set()
            self.edges  = set()
            self.edgesX = set()
            self.edgesY = set()
            self.edgesZ = set()

        self.children = np.empty([hi.size for hi in h],dtype=TreeCell)

        if self.dim == 2:
            for i in range(h[0].size):
                for j in range(h[1].size):
                    fXm = None if i is 0 else self.children[i-1][j].faces['fXp']
                    fYm = None if j is 0 else self.children[i][j-1].faces['fYp']
                    x0i = (np.r_[x0[0], h[0][:i]]).sum()
                    x0j = (np.r_[x0[1], h[1][:j]]).sum()
                    self.children[i][j] = TreeCell(self, x0=[x0i, x0j], depth=0, sz=[h[0][i], h[1][j]], fXm=fXm, fYm=fYm)

        elif self.dim == 3:
            for i in range(h[0].size):
                for j in range(h[1].size):
                    for k in range(h[2].size):
                        fXm = None if i is 0 else self.children[i-1][j][k].faces['fXp']
                        fYm = None if j is 0 else self.children[i][j-1][k].faces['fYp']
                        fZm = None if k is 0 else self.children[i][j][k-1].faces['fZp']
                        x0i = (np.r_[x0[0], h[0][:i]]).sum()
                        x0j = (np.r_[x0[1], h[1][:j]]).sum()
                        x0k = (np.r_[x0[2], h[2][:k]]).sum()
                        self.children[i][j] = TreeCell(self, x0=[x0i, x0j, x0k], depth=0, sz=[h[0][i], h[1][j], h[2][k]], fXm=fXm, fYm=fYm, fZm=fZm)

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
        for i, sC in enumerate(self.sortedCells): sC.num = i

        self.sortedNodes = sorted(self.nodes,key=SortByX0())
        for i, sN in enumerate(self.sortedNodes): sN.num = i

        self.sortedFaceX = sorted(self.facesX,key=SortByX0())
        for i, sFx in enumerate(self.sortedFaceX): sFx.num = i

        self.sortedFaceY = sorted(self.facesY,key=SortByX0())
        for i, sFy in enumerate(self.sortedFaceY): sFy.num = i + self.nFx

        if self.dim == 3:
            self.sortedFaceZ = sorted(self.facesZ,key=SortByX0())
            for i, sFz in enumerate(self.sortedFaceZ): sFz.num = i + self.nFx + self.nFy

            self.sortedEdgeX = sorted(self.edgesX,key=SortByX0())
            for i, sEx in enumerate(self.sortedEdgeX): sEx.num = i

            self.sortedEdgeY = sorted(self.edgesY,key=SortByX0())
            for i, sEy in enumerate(self.sortedEdgeY): sEy.num = i + self.nEx

            self.sortedEdgeZ = sorted(self.edgesZ,key=SortByX0())
            for i, sEz in enumerate(self.sortedEdgeZ): sEz.num = i + self.nEx + self.nEy

        self.isNumbered = True

    @property
    def dim(self): return len(self.h)

    @property
    def nC(self): return len(self.cells)

    @property
    def nN(self): return len(self.nodes)

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
    def gridN(self):
        if getattr(self, '_gridN', None) is None:
            self.number()
            self._gridN = np.empty((self.nN,self.dim))
            for ii, node in enumerate(self.sortedNodes):
                self._gridN[ii,:] = node.center
        return self._gridN

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
    def gridFz(self):
        if self.dim == 2: return None
        if getattr(self, '_gridFz', None) is None:
            self.number()
            self._gridFz = np.emptz((self.nFz,self.dim))
            for ii, face in enumerate(self.sortedFaceZ):
                self._gridFz[ii,:] = face.center
        return self._gridFz

    @property
    def gridEx(self):
        if self.dim == 2: return self.gridFy
        else: raise NotImplementedError('Edge Grid not yet implemented')

    @property
    def gridEy(self):
        if self.dim == 2: return self.gridFx
        else: raise NotImplementedError('Edge Grid not yet implemented')

    @property
    def gridEz(self):
        if self.dim == 2: return None
        else: raise NotImplementedError('Edge Grid not yet implemented')

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

    def plotGrid(self, ax=None, text=True, plotC=True, plotF=True, showIt=False):
        axOpts = {'projection':'3d'} if self.dim == 3 else {}
        if ax is None: ax = plt.subplot(111, **axOpts)

        if plotC: [c.plotGrid(ax, text=text) for c in self.cells]
        if plotF: [f.plotGrid(ax, text=text) for f in self.faces]

        ax.set_xlim((self.x0[0], self.h[0].sum()))
        ax.set_ylim((self.x0[1], self.h[1].sum()))
        if self.dim == 3:
            ax.set_zlim((self.x0[2], self.h[2].sum()))
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
