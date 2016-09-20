from __future__ import print_function
from SimPEG import Utils, np
from SimPEG.Mesh.BaseMesh import BaseRectangularMesh
from SimPEG.Mesh.DiffOperators import DiffOperators
from SimPEG.Mesh.InnerProducts import InnerProducts
from SimPEG.Mesh.View import CurvView


# Some helper functions.
def length2D(x):
    return (x[:, 0]**2 + x[:, 1]**2)**0.5


def length3D(x):
    return (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5


def normalize2D(x):
    return x/np.kron(np.ones((1, 2)), Utils.mkvc(length2D(x), 2))


def normalize3D(x):
    return x/np.kron(np.ones((1, 3)), Utils.mkvc(length3D(x), 2))


# Curvi Mesh

class CurvilinearMesh(BaseRectangularMesh, DiffOperators, InnerProducts,CurvView):
    """
    CurvilinearMesh is a mesh class that deals with curvilinear meshes.

    Example of a curvilinear mesh:

    .. plot::
            :include-source:

            from SimPEG import Mesh, Utils
            X, Y = Utils.exampleLrmGrid([3,3],'rotate')
            M = Mesh.CurvilinearMesh([X, Y])
            M.plotGrid(showIt=True)
    """

    _meshType = 'Curv'

    def __init__(self, nodes):
        assert type(nodes) == list, ("'nodes' variable must be a list of "
                                     "np.ndarray")
        assert len(nodes) > 1, "len(node) must be greater than 1"

        for i, nodes_i in enumerate(nodes):
            assert isinstance(nodes_i, np.ndarray), ("nodes[{0:d}] is not a"
                                                     "numpy array.".format(i))
            assert nodes_i.shape == nodes[0].shape, ("nodes[{0:d}] is not the "
                                                     "same shape as nodes[0]"
                                                     .format(i))

        assert len(nodes[0].shape) == len(nodes), "Dimension mismatch"
        assert len(nodes[0].shape) > 1, "Not worth using Curv for a 1D mesh."

        BaseRectangularMesh.__init__(self, np.array(nodes[0].shape)-1, None)

        # Save nodes to private variable _gridN as vectors
        self._gridN = np.ones((nodes[0].size, self.dim))
        for i, node_i in enumerate(nodes):
            self._gridN[:, i] = Utils.mkvc(node_i.astype(float))

    @property
    def gridCC(self):
        """
        Cell-centered grid
        """
        if getattr(self, '_gridCC', None) is None:
            self._gridCC = np.concatenate([self.aveN2CC*self.gridN[:, i]
                                           for i in range(self.dim)]).reshape(
                                           (-1, self.dim), order='F')
        return self._gridCC

    @property
    def gridN(self):
        """
        Nodal grid.
        """
        if getattr(self, '_gridN', None) is None:
            raise Exception("Someone deleted this. I blame you.")
        return self._gridN

    @property
    def gridFx(self):
        """
        Face staggered grid in the x direction.
        """

        if getattr(self, '_gridFx', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [Utils.mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._gridFx = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [Utils.mkvc(0.25 * (n[:, :-1, :-1] + n[:, :-1, 1:] +
                       n[:, 1:, :-1] + n[:, 1:, 1:])) for n in N]
                self._gridFx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFx

    @property
    def gridFy(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, '_gridFy', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [Utils.mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._gridFy = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [Utils.mkvc(0.25 * (n[:-1, :, :-1] + n[:-1, :, 1:] +
                       n[1:, :, :-1] + n[1:, :, 1:])) for n in N]
                self._gridFy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFy

    @property
    def gridFz(self):
        """
        Face staggered grid in the y direction.
        """

        if getattr(self, '_gridFz', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            XYZ = [Utils.mkvc(0.25 * (n[:-1, :-1, :] + n[:-1, 1:, :] +
                   n[1:, :-1, :] + n[1:, 1:, :])) for n in N]
            self._gridFz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridFz

    @property
    def gridEx(self):
        """
        Edge staggered grid in the x direction.
        """
        if getattr(self, '_gridEx', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [Utils.mkvc(0.5 * (n[:-1, :] + n[1:, :])) for n in N]
                self._gridEx = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [Utils.mkvc(0.5 * (n[:-1, :, :] + n[1:, :, :])) for n in N]
                self._gridEx = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEx

    @property
    def gridEy(self):
        """
        Edge staggered grid in the y direction.
        """
        if getattr(self, '_gridEy', None) is None:
            N = self.r(self.gridN, 'N', 'N', 'M')
            if self.dim == 2:
                XY = [Utils.mkvc(0.5 * (n[:, :-1] + n[:, 1:])) for n in N]
                self._gridEy = np.c_[XY[0], XY[1]]
            elif self.dim == 3:
                XYZ = [Utils.mkvc(0.5 * (n[:, :-1, :] + n[:, 1:, :])) for n in N]
                self._gridEy = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEy

    @property
    def gridEz(self):
        """
        Edge staggered grid in the z direction.
        """
        if getattr(self, '_gridEz', None) is None and self.dim == 3:
            N = self.r(self.gridN, 'N', 'N', 'M')
            XYZ = [Utils.mkvc(0.5 * (n[:, :, :-1] + n[:, :, 1:])) for n in N]
            self._gridEz = np.c_[XYZ[0], XYZ[1], XYZ[2]]
        return self._gridEz

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

    @property
    def vol(self):
        """
        Construct cell volumes of the 3D model as 1d array
        """

        if getattr(self, '_vol', None) is None:
            if self.dim == 2:
                A, B, C, D = Utils.indexCube('ABCD', self.vnC+1)
                normal, area = Utils.faceInfo(np.c_[self.gridN, np.zeros(
                                              (self.nN, 1))], A, B, C, D)
                self._vol = area
            elif self.dim == 3:
                # Each polyhedron can be decomposed into 5 tetrahedrons
                # However, this presents a choice so we may as well divide in
                # two ways and average.
                A, B, C, D, E, F, G, H = Utils.indexCube('ABCDEFGH', self.vnC +
                                                         1)

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

    @property
    def area(self):
        if (getattr(self, '_area', None) is None or
            getattr(self, '_normals', None) is None):
            # Compute areas of cell faces
            if(self.dim == 2):
                xy = self.gridN
                A, B = Utils.indexCube('AB', self.vnC+1, np.array([self.nNx,
                                       self.nCy]))
                edge1 = xy[B, :] - xy[A, :]
                normal1 = np.c_[edge1[:, 1], -edge1[:, 0]]
                area1 = length2D(edge1)
                A, D = Utils.indexCube('AD', self.vnC+1, np.array([self.nCx,
                                       self.nNy]))
                # Note that we are doing A-D to make sure the normal points the
                # right way.
                # Think about it. Look at the picture. Normal points towards C
                # iff you do this.
                edge2 = xy[A, :] - xy[D, :]
                normal2 = np.c_[edge2[:, 1], -edge2[:, 0]]
                area2 = length2D(edge2)
                self._area = np.r_[Utils.mkvc(area1), Utils.mkvc(area2)]
                self._normals = [normalize2D(normal1), normalize2D(normal2)]

            elif(self.dim == 3):

                A, E, F, B = Utils.indexCube('AEFB', self.vnC+1, np.array(
                                             [self.nNx, self.nCy, self.nCz]))
                normal1, area1 = Utils.faceInfo(self.gridN, A, E, F, B,
                                                average=False,
                                                normalizeNormals=False)

                A, D, H, E = Utils.indexCube('ADHE', self.vnC+1, np.array(
                                             [self.nCx, self.nNy, self.nCz]))
                normal2, area2 = Utils.faceInfo(self.gridN, A, D, H, E,
                                                average=False,
                                                normalizeNormals=False)

                A, B, C, D = Utils.indexCube('ABCD', self.vnC+1, np.array(
                                             [self.nCx, self.nCy, self.nNz]))
                normal3, area3 = Utils.faceInfo(self.gridN, A, B, C, D,
                                                average=False,
                                                normalizeNormals=False)

                self._area = np.r_[Utils.mkvc(area1), Utils.mkvc(area2),
                                   Utils.mkvc(area3)]
                self._normals = [normal1, normal2, normal3]
        return self._area

    @property
    def normals(self):
        """
        Face normals: calling this will average
        the computed normals so that there is one
        per face. This is especially relevant in
        3D, as there are up to 4 different normals
        for each face that will be different.

        To reshape the normals into a matrix and get the y component::

            NyX, NyY, NyZ = M.r(M.normals, 'F', 'Fy', 'M')
        """

        if getattr(self, '_normals', None) is None:
            self.area  # calling .area will create the face normals
        if self.dim == 2:
            return normalize2D(np.r_[self._normals[0], self._normals[1]])
        elif self.dim == 3:
            normal1 = (self._normals[0][0] + self._normals[0][1] + self._normals[0][2] + self._normals[0][3])/4
            normal2 = (self._normals[1][0] + self._normals[1][1] + self._normals[1][2] + self._normals[1][3])/4
            normal3 = (self._normals[2][0] + self._normals[2][1] + self._normals[2][2] + self._normals[2][3])/4
            return normalize3D(np.r_[normal1, normal2, normal3])

    @property
    def edge(self):
        """
        Edge lengths
        """
        if getattr(self, '_edge', None) is None:
            if(self.dim == 2):
                xy = self.gridN
                A, D = Utils.indexCube('AD', self.vnC+1, np.array([self.nCx,
                                                                  self.nNy]))
                edge1 = xy[D, :] - xy[A, :]
                A, B = Utils.indexCube('AB', self.vnC+1, np.array([self.nNx,
                                                                   self.nCy]))
                edge2 = xy[B, :] - xy[A, :]
                self._edge = np.r_[Utils.mkvc(length2D(edge1)),
                                   Utils.mkvc(length2D(edge2))]
                self._tangents = np.r_[edge1, edge2]/np.c_[self._edge,
                                                           self._edge]
            elif(self.dim == 3):
                xyz = self.gridN
                A, D = Utils.indexCube('AD', self.vnC+1, np.array([self.nCx,
                                                                   self.nNy,
                                                                   self.nNz]))
                edge1 = xyz[D, :] - xyz[A, :]
                A, B = Utils.indexCube('AB', self.vnC+1, np.array([self.nNx,
                                                                   self.nCy,
                                                                   self.nNz]))
                edge2 = xyz[B, :] - xyz[A, :]
                A, E = Utils.indexCube('AE', self.vnC+1, np.array([self.nNx,
                                                                   self.nNy,
                                                                   self.nCz]))
                edge3 = xyz[E, :] - xyz[A, :]
                self._edge = np.r_[Utils.mkvc(length3D(edge1)),
                                   Utils.mkvc(length3D(edge2)),
                                   Utils.mkvc(length3D(edge3))]
                self._tangents = (np.r_[edge1, edge2, edge3] /
                                  np.c_[self._edge, self._edge, self._edge])
            return self._edge
        return self._edge

    @property
    def tangents(self):
        """
        Edge tangents
        """
        if getattr(self, '_tangents', None) is None:
            self.edge  # calling .edge will create the tangents
        return self._tangents



if __name__ == '__main__':
    nc = 5
    h1 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    nc = 7
    h2 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h3 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    dee3 = True
    if dee3:
        X, Y, Z = Utils.ndgrid(h1, h2, h3, vector=False)
        M = CurvilinearMesh([X, Y, Z])
    else:
        X, Y = Utils.ndgrid(h1, h2, vector=False)
        M = CurvilinearMesh([X, Y])

    print(M.r(M.normals, 'F', 'Fx', 'V'))
