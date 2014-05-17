from SimPEG import Utils, np
from BaseMesh import BaseRectangularMesh
from DiffOperators import DiffOperators
from InnerProducts import InnerProducts

# Some helper functions.
length2D = lambda x: (x[:, 0]**2 + x[:, 1]**2)**0.5
length3D = lambda x: (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5
normalize2D = lambda x: x/np.kron(np.ones((1, 2)), Utils.mkvc(length2D(x), 2))
normalize3D = lambda x: x/np.kron(np.ones((1, 3)), Utils.mkvc(length3D(x), 2))


class LogicallyRectMesh(BaseRectangularMesh, DiffOperators, InnerProducts):
    """
    LogicallyRectMesh is a mesh class that deals with logically rectangular meshes.

    Example of a logically rectangular mesh:

    .. plot::
            :include-source:

            from SimPEG import Mesh, Utils
            X, Y = Utils.exampleLrmGrid([3,3],'rotate')
            M = Mesh.LogicallyRectMesh([X, Y])
            M.plotGrid(showIt=True)
    """

    __metaclass__ = Utils.SimPEGMetaClass

    _meshType = 'LRM'

    def __init__(self, nodes):
        assert type(nodes) == list, "'nodes' variable must be a list of np.ndarray"
        assert len(nodes) > 1, "len(node) must be greater than 1"

        for i, nodes_i in enumerate(nodes):
            assert isinstance(nodes_i, np.ndarray), ("nodes[%i] is not a numpy array." % i)
            assert nodes_i.shape == nodes[0].shape, ("nodes[%i] is not the same shape as nodes[0]" % i)

        assert len(nodes[0].shape) == len(nodes), "Dimension mismatch"
        assert len(nodes[0].shape) > 1, "Not worth using LRM for a 1D mesh."

        BaseRectangularMesh.__init__(self, np.array(nodes[0].shape)-1, None)

        # Save nodes to private variable _gridN as vectors
        self._gridN = np.ones((nodes[0].size, self.dim))
        for i, node_i in enumerate(nodes):
            self._gridN[:, i] = Utils.mkvc(node_i.astype(float))

    def gridCC():
        doc = "Cell-centered grid."

        def fget(self):
            if self._gridCC is None:
                self._gridCC = np.concatenate([self.aveN2CC*self.gridN[:,i] for i in range(self.dim)]).reshape((-1,self.dim), order='F')
            return self._gridCC
        return locals()
    _gridCC = None  # Store grid by default
    gridCC = property(**gridCC())

    def gridN():
        doc = "Nodal grid."

        def fget(self):
            if self._gridN is None:
                raise Exception("Someone deleted this. I blame you.")
            return self._gridN
        return locals()
    _gridN = None  # Store grid by default
    gridN = property(**gridN())

    def gridFx():
        doc = "Face staggered grid in the x direction."

        def fget(self):
            if self._gridFx is None:
                N = self.r(self.gridN, 'N', 'N', 'M')
                if self.dim == 2:
                    XY = [Utils.mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                    self._gridFx = np.c_[XY[0], XY[1]]
                elif self.dim == 3:
                    XYZ = [Utils.mkvc(0.25 * (n[:, :-1, :-1] + n[:, :-1, 1:] + n[:, 1:, :-1] + n[:, 1:, 1:])) for n in N]
                    self._gridFx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
            return self._gridFx
        return locals()
    _gridFx = None  # Store grid by default
    gridFx = property(**gridFx())

    def gridFy():
        doc = "Face staggered grid in the y direction."

        def fget(self):
            if self._gridFy is None:
                N = self.r(self.gridN, 'N', 'N', 'M')
                if self.dim == 2:
                    XY = [Utils.mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                    self._gridFy = np.c_[XY[0], XY[1]]
                elif self.dim == 3:
                    XYZ = [Utils.mkvc(0.25 * (n[:-1, :, :-1] + n[:-1, :, 1:] + n[1:, :, :-1] + n[1:, :, 1:])) for n in N]
                    self._gridFy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
            return self._gridFy
        return locals()
    _gridFy = None  # Store grid by default
    gridFy = property(**gridFy())

    def gridFz():
        doc = "Face staggered grid in the z direction."

        def fget(self):
            if self._gridFz is None and self.dim == 3:
                N = self.r(self.gridN, 'N', 'N', 'M')
                XYZ = [Utils.mkvc(0.25 * (n[:-1, :-1, :] + n[:-1, 1:, :] + n[1:, :-1, :] + n[1:, 1:, :])) for n in N]
                self._gridFz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
            return self._gridFz
        return locals()
    _gridFz = None  # Store grid by default
    gridFz = property(**gridFz())

    def gridEx():
        doc = "Edge staggered grid in the x direction."

        def fget(self):
            if self._gridEx is None:
                N = self.r(self.gridN, 'N', 'N', 'M')
                if self.dim == 2:
                    XY = [Utils.mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                    self._gridEx = np.c_[XY[0], XY[1]]
                elif self.dim == 3:
                    XYZ = [Utils.mkvc(0.5 * (n[:-1, :, :] + n[1:, :, :])) for n in N]
                    self._gridEx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
            return self._gridEx
        return locals()
    _gridEx = None  # Store grid by default
    gridEx = property(**gridEx())

    def gridEy():
        doc = "Edge staggered grid in the y direction."

        def fget(self):
            if self._gridEy is None:
                N = self.r(self.gridN, 'N', 'N', 'M')
                if self.dim == 2:
                    XY = [Utils.mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                    self._gridEy = np.c_[XY[0], XY[1]]
                elif self.dim == 3:
                    XYZ = [Utils.mkvc(0.5 * (n[:, :-1, :] + n[:, 1:, :])) for n in N]
                    self._gridEy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
            return self._gridEy
        return locals()
    _gridEy = None  # Store grid by default
    gridEy = property(**gridEy())

    def gridEz():
        doc = "Edge staggered grid in the z direction."

        def fget(self):
            if self._gridEz is None and self.dim == 3:
                N = self.r(self.gridN, 'N', 'N', 'M')
                XYZ = [Utils.mkvc(0.5 * (n[:, :, :-1] + n[:, :, 1:])) for n in N]
                self._gridEz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
            return self._gridEz
        return locals()
    _gridEz = None  # Store grid by default
    gridEz = property(**gridEz())

    # --------------- Geometries ---------------------
    #
    #
    # ------------------- 2D -------------------------
    #
    #         node(i,j)          node(i,j+1)
    #              A -------------- B
    #              |                |
    #              |    cell(i,j)   |
    #              |        I       |
    #              |                |
    #             D -------------- C
    #         node(i+1,j)        node(i+1,j+1)
    #
    # ------------------- 3D -------------------------
    #
    #
    #             node(i,j,k+1)       node(i,j+1,k+1)
    #                 E --------------- F
    #                /|               / |
    #               / |              /  |
    #              /  |             /   |
    #       node(i,j,k)         node(i,j+1,k)
    #            A -------------- B     |
    #            |    H ----------|---- G
    #            |   /cell(i,j)   |   /
    #            |  /     I       |  /
    #            | /              | /
    #            D -------------- C
    #       node(i+1,j,k)      node(i+1,j+1,k)
    def vol():
        doc = "Construct cell volumes of the 3D model as 1d array."

        def fget(self):
            if(self._vol is None):
                if self.dim == 2:
                    A, B, C, D = Utils.indexCube('ABCD', self.vnC+1)
                    normal, area = Utils.faceInfo(np.c_[self.gridN, np.zeros((self.nN, 1))], A, B, C, D)
                    self._vol = area
                elif self.dim == 3:
                    # Each polyhedron can be decomposed into 5 tetrahedrons
                    # However, this presents a choice so we may as well divide in two ways and average.
                    A, B, C, D, E, F, G, H = Utils.indexCube('ABCDEFGH', self.vnC+1)

                    vol1 = (Utils.volTetra(self.gridN, A, B, D, E) +  # cutted edge top
                            Utils.volTetra(self.gridN, B, E, F, G) +  # cutted edge top
                            Utils.volTetra(self.gridN, B, D, E, G) +  # middle
                            Utils.volTetra(self.gridN, B, C, D, G) +  # cutted edge bottom
                            Utils.volTetra(self.gridN, D, E, G, H))   # cutted edge bottom

                    vol2 = (Utils.volTetra(self.gridN, A, F, B, C) +  # cutted edge top
                            Utils.volTetra(self.gridN, A, E, F, H) +  # cutted edge top
                            Utils.volTetra(self.gridN, A, H, F, C) +  # middle
                            Utils.volTetra(self.gridN, C, H, D, A) +  # cutted edge bottom
                            Utils.volTetra(self.gridN, C, G, H, F))   # cutted edge bottom

                    self._vol = (vol1 + vol2)/2
            return self._vol
        return locals()
    _vol = None
    vol = property(**vol())

    def area():
        doc = "Face areas."

        def fget(self):
            if(self._area is None or self._normals is None):
                # Compute areas of cell faces
                if(self.dim == 2):
                    xy = self.gridN
                    A, B = Utils.indexCube('AB', self.vnC+1, np.array([self.nNx, self.nCy]))
                    edge1 = xy[B, :] - xy[A, :]
                    normal1 = np.c_[edge1[:, 1], -edge1[:, 0]]
                    area1 = length2D(edge1)
                    A, D = Utils.indexCube('AD', self.vnC+1, np.array([self.nCx, self.nNy]))
                    # Note that we are doing A-D to make sure the normal points the right way.
                    # Think about it. Look at the picture. Normal points towards C iff you do this.
                    edge2 = xy[A, :] - xy[D, :]
                    normal2 = np.c_[edge2[:, 1], -edge2[:, 0]]
                    area2 = length2D(edge2)
                    self._area = np.r_[Utils.mkvc(area1), Utils.mkvc(area2)]
                    self._normals = [normalize2D(normal1), normalize2D(normal2)]
                elif(self.dim == 3):

                    A, E, F, B = Utils.indexCube('AEFB', self.vnC+1, np.array([self.nNx, self.nCy, self.nCz]))
                    normal1, area1 = Utils.faceInfo(self.gridN, A, E, F, B, average=False, normalizeNormals=False)

                    A, D, H, E = Utils.indexCube('ADHE', self.vnC+1, np.array([self.nCx, self.nNy, self.nCz]))
                    normal2, area2 = Utils.faceInfo(self.gridN, A, D, H, E, average=False, normalizeNormals=False)

                    A, B, C, D = Utils.indexCube('ABCD', self.vnC+1, np.array([self.nCx, self.nCy, self.nNz]))
                    normal3, area3 = Utils.faceInfo(self.gridN, A, B, C, D, average=False, normalizeNormals=False)

                    self._area = np.r_[Utils.mkvc(area1), Utils.mkvc(area2), Utils.mkvc(area3)]
                    self._normals = [normal1, normal2, normal3]
            return self._area
        return locals()
    _area = None
    area = property(**area())

    def normals():
        doc = """Face normals: calling this will average
        the computed normals so that there is one
        per face. This is especially relevant in
        3D, as there are up to 4 different normals
        for each face that will be different.

        To reshape the normals into a matrix and get the y component::

            NyX, NyY, NyZ = M.r(M.normals, 'F', 'Fy', 'M')
        """

        def fget(self):
            if(self._normals is None):
                self.area  # calling .area will create the face normals
            if self.dim == 2:
                return normalize2D(np.r_[self._normals[0], self._normals[1]])
            elif self.dim == 3:
                normal1 = (self._normals[0][0] + self._normals[0][1] + self._normals[0][2] + self._normals[0][3])/4
                normal2 = (self._normals[1][0] + self._normals[1][1] + self._normals[1][2] + self._normals[1][3])/4
                normal3 = (self._normals[2][0] + self._normals[2][1] + self._normals[2][2] + self._normals[2][3])/4
                return normalize3D(np.r_[normal1, normal2, normal3])
        return locals()
    _normals = None
    normals = property(**normals())

    def edge():
        doc = "Edge legnths."

        def fget(self):
            if(self._edge is None or self._tangents is None):
                if(self.dim == 2):
                    xy = self.gridN
                    A, D = Utils.indexCube('AD', self.vnC+1, np.array([self.nCx, self.nNy]))
                    edge1 = xy[D, :] - xy[A, :]
                    A, B = Utils.indexCube('AB', self.vnC+1, np.array([self.nNx, self.nCy]))
                    edge2 = xy[B, :] - xy[A, :]
                    self._edge = np.r_[Utils.mkvc(length2D(edge1)), Utils.mkvc(length2D(edge2))]
                    self._tangents = np.r_[edge1, edge2]/np.c_[self._edge, self._edge]
                elif(self.dim == 3):
                    xyz = self.gridN
                    A, D = Utils.indexCube('AD', self.vnC+1, np.array([self.nCx, self.nNy, self.nNz]))
                    edge1 = xyz[D, :] - xyz[A, :]
                    A, B = Utils.indexCube('AB', self.vnC+1, np.array([self.nNx, self.nCy, self.nNz]))
                    edge2 = xyz[B, :] - xyz[A, :]
                    A, E = Utils.indexCube('AE', self.vnC+1, np.array([self.nNx, self.nNy, self.nCz]))
                    edge3 = xyz[E, :] - xyz[A, :]
                    self._edge = np.r_[Utils.mkvc(length3D(edge1)), Utils.mkvc(length3D(edge2)), Utils.mkvc(length3D(edge3))]
                    self._tangents = np.r_[edge1, edge2, edge3]/np.c_[self._edge, self._edge, self._edge]
            return self._edge
        return locals()
    _edge = None
    edge = property(**edge())

    def tangents():
        doc = "Edge tangents."

        def fget(self):
            if(self._tangents is None):
                self.edge  # calling .edge will create the tangents
            return self._tangents
        return locals()
    _tangents = None
    tangents = property(**tangents())



    #############################################
    #            Plotting Functions             #
    #############################################

    def plotGrid(self, ax=None, nodes=False, faces=False, centers=False, edges=False, lines=True,  showIt=False):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions.


        .. plot::
            :include-source:

            from SimPEG import Mesh, Utils
            X, Y = Utils.exampleLrmGrid([3,3],'rotate')
            M = Mesh.LogicallyRectMesh([X, Y])
            M.plotGrid(showIt=True)

        """
        import matplotlib.pyplot as plt
        import matplotlib
        from mpl_toolkits.mplot3d import Axes3D
        mkvc = Utils.mkvc

        axOpts = {'projection':'3d'} if self.dim == 3 else {}
        if ax is None: ax = plt.subplot(111, **axOpts)

        NN = self.r(self.gridN, 'N', 'N', 'M')
        if self.dim == 2:

            if lines:
                X1 = np.c_[mkvc(NN[0][:-1, :]), mkvc(NN[0][1:, :]), mkvc(NN[0][:-1, :])*np.nan].flatten()
                Y1 = np.c_[mkvc(NN[1][:-1, :]), mkvc(NN[1][1:, :]), mkvc(NN[1][:-1, :])*np.nan].flatten()

                X2 = np.c_[mkvc(NN[0][:, :-1]), mkvc(NN[0][:, 1:]), mkvc(NN[0][:, :-1])*np.nan].flatten()
                Y2 = np.c_[mkvc(NN[1][:, :-1]), mkvc(NN[1][:, 1:]), mkvc(NN[1][:, :-1])*np.nan].flatten()

                X = np.r_[X1, X2]
                Y = np.r_[Y1, Y2]

                ax.plot(X, Y, 'b-')
            if centers:
                ax.plot(self.gridCC[:,0],self.gridCC[:,1],'ro')

            # Nx = self.r(self.normals, 'F', 'Fx', 'V')
            # Ny = self.r(self.normals, 'F', 'Fy', 'V')
            # Tx = self.r(self.tangents, 'E', 'Ex', 'V')
            # Ty = self.r(self.tangents, 'E', 'Ey', 'V')

            # ax.plot(self.gridN[:, 0], self.gridN[:, 1], 'bo')

            # nX = np.c_[self.gridFx[:, 0], self.gridFx[:, 0] + Nx[0]*length, self.gridFx[:, 0]*np.nan].flatten()
            # nY = np.c_[self.gridFx[:, 1], self.gridFx[:, 1] + Nx[1]*length, self.gridFx[:, 1]*np.nan].flatten()
            # ax.plot(self.gridFx[:, 0], self.gridFx[:, 1], 'rs')
            # ax.plot(nX, nY, 'r-')

            # nX = np.c_[self.gridFy[:, 0], self.gridFy[:, 0] + Ny[0]*length, self.gridFy[:, 0]*np.nan].flatten()
            # nY = np.c_[self.gridFy[:, 1], self.gridFy[:, 1] + Ny[1]*length, self.gridFy[:, 1]*np.nan].flatten()
            # #ax.plot(self.gridFy[:, 0], self.gridFy[:, 1], 'gs')
            # ax.plot(nX, nY, 'g-')

            # tX = np.c_[self.gridEx[:, 0], self.gridEx[:, 0] + Tx[0]*length, self.gridEx[:, 0]*np.nan].flatten()
            # tY = np.c_[self.gridEx[:, 1], self.gridEx[:, 1] + Tx[1]*length, self.gridEx[:, 1]*np.nan].flatten()
            # ax.plot(self.gridEx[:, 0], self.gridEx[:, 1], 'r^')
            # ax.plot(tX, tY, 'r-')

            # nX = np.c_[self.gridEy[:, 0], self.gridEy[:, 0] + Ty[0]*length, self.gridEy[:, 0]*np.nan].flatten()
            # nY = np.c_[self.gridEy[:, 1], self.gridEy[:, 1] + Ty[1]*length, self.gridEy[:, 1]*np.nan].flatten()
            # #ax.plot(self.gridEy[:, 0], self.gridEy[:, 1], 'g^')
            # ax.plot(nX, nY, 'g-')

        elif self.dim == 3:
            X1 = np.c_[mkvc(NN[0][:-1, :, :]), mkvc(NN[0][1:, :, :]), mkvc(NN[0][:-1, :, :])*np.nan].flatten()
            Y1 = np.c_[mkvc(NN[1][:-1, :, :]), mkvc(NN[1][1:, :, :]), mkvc(NN[1][:-1, :, :])*np.nan].flatten()
            Z1 = np.c_[mkvc(NN[2][:-1, :, :]), mkvc(NN[2][1:, :, :]), mkvc(NN[2][:-1, :, :])*np.nan].flatten()

            X2 = np.c_[mkvc(NN[0][:, :-1, :]), mkvc(NN[0][:, 1:, :]), mkvc(NN[0][:, :-1, :])*np.nan].flatten()
            Y2 = np.c_[mkvc(NN[1][:, :-1, :]), mkvc(NN[1][:, 1:, :]), mkvc(NN[1][:, :-1, :])*np.nan].flatten()
            Z2 = np.c_[mkvc(NN[2][:, :-1, :]), mkvc(NN[2][:, 1:, :]), mkvc(NN[2][:, :-1, :])*np.nan].flatten()

            X3 = np.c_[mkvc(NN[0][:, :, :-1]), mkvc(NN[0][:, :, 1:]), mkvc(NN[0][:, :, :-1])*np.nan].flatten()
            Y3 = np.c_[mkvc(NN[1][:, :, :-1]), mkvc(NN[1][:, :, 1:]), mkvc(NN[1][:, :, :-1])*np.nan].flatten()
            Z3 = np.c_[mkvc(NN[2][:, :, :-1]), mkvc(NN[2][:, :, 1:]), mkvc(NN[2][:, :, :-1])*np.nan].flatten()

            X = np.r_[X1, X2, X3]
            Y = np.r_[Y1, Y2, Y3]
            Z = np.r_[Z1, Z2, Z3]

            ax.plot(X, Y, 'b', zs=Z)
            ax.set_zlabel('x3')

        ax.grid(True)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        if showIt: plt.show()


if __name__ == '__main__':
    nc = 5
    h1 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    nc = 7
    h2 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h3 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    dee3 = True
    if dee3:
        X, Y, Z = Utils.ndgrid(h1, h2, h3, vector=False)
        M = LogicallyRectMesh([X, Y, Z])
    else:
        X, Y = Utils.ndgrid(h1, h2, vector=False)
        M = LogicallyRectMesh([X, Y])

    print M.r(M.normals, 'F', 'Fx', 'V')
