from scipy import sparse as sp
from SimPEG.Utils import sub2ind, ndgrid, mkvc, getSubArray, sdiag, inv3X3BlockDiagonal, inv2X2BlockDiagonal
import numpy as np


class InnerProducts(object):
    """
        Class creates the inner product matrices that you need!

        InnerProducts is a base class providing inner product matrices for meshes and cannot run on its own. Inherit to your favorite Mesh class.


        **Example problem for DC resistivity**

        .. math::

            \sigma^{-1}\mathbf{J} = \\nabla \phi

        We can define in weak form by integrating with a general face function F:

        .. math::

            \int_{\\text{cell}}{\sigma^{-1}\mathbf{J} \cdot \mathbf{F}} = \int_{\\text{cell}}{\\nabla \phi  \cdot \mathbf{F}}

            \int_{\\text{cell}}{\sigma^{-1}\mathbf{J} \cdot \mathbf{F}} = \int_{\\text{cell}}{(\\nabla \cdot \mathbf{F}) \phi   } + \int_{\partial \\text{cell}}{ \phi  \mathbf{F} \cdot \mathbf{n}}

        We can then discretize for every cell:

        .. math::

            v_{\\text{cell}} \sigma^{-1} (\mathbf{J}_x \mathbf{F}_x +\mathbf{J}_y \mathbf{F}_y  + \mathbf{J}_z \mathbf{F}_z ) = -\phi^{\\top} v_{\\text{cell}} (\mathbf{D}_{\\text{cell}} \mathbf{F})  + \\text{BC}

        We can represent this in vector form (again this is for every cell), and will generalize for the case of anisotropic (tensor) sigma.

        .. math::

            \mathbf{F}_c^{\\top} (\sqrt{v_{\\text{cell}}} \Sigma^{-1} \sqrt{v_{\\text{cell}}})  \mathbf{J}_c = -\phi^{\\top} v_{\\text{cell}}( v_\\text{cell}^{-1} \mathbf{D}_{\\text{cell}} \mathbf{A} \mathbf{F})  + \\text{BC}

        We multiply by volume on each side of the tensor conductivity to keep symmetry in the system. Here J_c is the Cartesian J (on the faces) and must be calculated differently depending on the mesh:

        .. math::
            \mathbf{J}_c = \mathbf{Q}_{(i)}\mathbf{J}_\\text{TENSOR} = \mathbf{N}_{(i)}^{-1}\mathbf{Q}_{(i)}\mathbf{J}_\\text{LOM}

        Here the i index refers to where we choose to approximate this integral.
        We will approximate this relation at every node of the cell, there are 8 in 3D, using a projection matrix Q_i to pick the appropriate fluxes.
        We will then average to the cell center. For the TENSOR mesh, this looks like:

        .. math::

            \mathbf{F}^{\\top}
                {1\over 8}
                \left(\sum_{i=1}^8
                \mathbf{Q}_{(i)}^{-\\top} \sqrt{v_{\\text{cell}}} \Sigma^{-1} \sqrt{v_{\\text{cell}}}  \mathbf{Q}_{(i)}
                \\right)
                \mathbf{J}
                =
                -\mathbf{F}^{\\top} \mathbf{A} \mathbf{D}_{\\text{cell}}^{\\top} \phi   + \\text{BC}

            \mathbf{M}(\Sigma^{-1}) \mathbf{J}
                =
                -\mathbf{A} \mathbf{D}_{\\text{cell}}^{\\top} \phi   + \\text{BC}

            \mathbf{M}(\Sigma^{-1}) = {1\over 8}
                \left(\sum_{i=1}^8
                \mathbf{Q}_{(i)}^{-\\top} \sqrt{v_{\\text{cell}}} \Sigma^{-1} \sqrt{v_{\\text{cell}}}  \mathbf{Q}_{(i)}
                \\right)

        The M is returned if mu is set equal  to \Sigma^{-1}.

        If requested (returnP=True) the projection matricies are returned as well (ordered by nodes).
        Here each P (3*nC, sum(nF)) is a combination of the projection, volume, and any normalization to Cartesian coordinates:

        .. math::
            \mathbf{P}_{(i)} =  \sqrt{ {1\over 8} v_{\\text{cell}}} \overbrace{\mathbf{N}_{(i)}^{-1}}^{\\text{LOM only}} \mathbf{Q}_{(i)}

        Note that this is completed for each cell in the mesh at the same time.
    """
    def __init__(self):
        raise Exception('InnerProducts is a base class providing inner product matrices for meshes and cannot run on its own. Inherit to your favorite Mesh class.')

    def getFaceInnerProduct(M, mu=None, returnP=False):
        """
            :param numpy.array mu: material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
            :param bool returnP: returns the projection matrices
            :rtype: scipy.csr_matrix
            :return: M, the inner product matrix (sum(nF), sum(nF))

            Depending on the number of columns (either 1, 3, or 6) of mu, the material property is interpreted as follows:

            .. math::
                \\vec{\mu} = \left[\\begin{matrix} \mu_{1} & 0 & 0 \\\\ 0 & \mu_{1} & 0 \\\\ 0 & 0 & \mu_{1}  \end{matrix}\\right]

                \\vec{\mu} = \left[\\begin{matrix} \mu_{1} & 0 & 0 \\\\ 0 & \mu_{2} & 0 \\\\ 0 & 0 & \mu_{3}  \end{matrix}\\right]

                \\vec{\mu} = \left[\\begin{matrix} \mu_{1} & \mu_{4} & \mu_{5} \\\\ \mu_{4} & \mu_{2} & \mu_{6} \\\\ \mu_{5} & \mu_{6} & \mu_{3}  \end{matrix}\\right]

                \mathbf{M}(\\vec{\mu}) = {1\over 8}
                    \left(\sum_{i=1}^8
                    \mathbf{J}_c^{-\\top} \sqrt{v_{\\text{cell}}} \\vec{\mu} \sqrt{v_{\\text{cell}}}  \mathbf{J}_c
                    \\right)

            If requested (returnP=True) the projection matricies are returned as well (ordered by nodes)::

                P = [P000, P100, P010, P110, P001, P101, P011, P111]

            Here each P (3*nC, sum(nF)) is a combination of the projection, volume, and any normalization to Cartesian coordinates:

            .. math::
                \mathbf{P}_{(i)} =  \sqrt{ {1\over 8} v_{\\text{cell}}} \overbrace{\mathbf{N}_{(i)}^{-1}}^{\\text{LOM only}} \mathbf{Q}_{(i)}

            Note that this is completed for each cell in the mesh at the same time.

            **For 2D:**

             Depending on the number of columns (either 1, 2, or 3) of mu, the material property is interpreted as follows:

            .. math::
                \\vec{\mu} = \left[\\begin{matrix} \mu_{1} & 0 \\\\ 0 & \mu_{1} \end{matrix}\\right]

                \\vec{\mu} = \left[\\begin{matrix} \mu_{1} & 0 \\\\ 0 & \mu_{2} \end{matrix}\\right]

                \\vec{\mu} = \left[\\begin{matrix} \mu_{1} & \mu_{3} \\\\ \mu_{3} & \mu_{2} \end{matrix}\\right]


            .. math::

                \mathbf{M}(\\vec{\mu}) = {1\over 4}
                    \left(\sum_{i=1}^4
                    \mathbf{J}_c^{-\\top} \sqrt{v_{\\text{cell}}} \\vec{\mu} \sqrt{v_{\\text{cell}}}  \mathbf{J}_c
                    \\right)


            If requested (returnP=True) the projection matricies are returned as well (ordered by nodes)::

                P = [P00, P10, P01, P11]

            Here each P (2*nC, sum(nF)) is a combination of the projection, volume, and any normalization to Cartesian coordinates:

            .. math::
                \mathbf{P}_{(i)} =  \sqrt{ {1\over 4} v_{\\text{cell}}} \overbrace{\mathbf{N}_{(i)}^{-1}}^{\\text{LOM only}} \mathbf{Q}_{(i)}

            Note that this is completed for each cell in the mesh at the same time.

        """
        if M.dim == 1:
            v = np.sqrt(0.5*M.vol)
            V1 = sdiag(v)  # We will multiply on each side to keep symmetry

            Px = _getFacePx(M)
            P000 = V1*Px('fXm')
            P100 = V1*Px('fXp')
        elif M.dim == 2:
            # Square root of cell volume multiplied by 1/4
            v = np.sqrt(0.25*M.vol)
            V2 = sdiag(np.r_[v, v])  # We will multiply on each side to keep symmetry

            Pxx = _getFacePxx(M)
            P000 = V2*Pxx('fXm', 'fYm')
            P100 = V2*Pxx('fXp', 'fYm')
            P010 = V2*Pxx('fXm', 'fYp')
            P110 = V2*Pxx('fXp', 'fYp')
        elif M.dim == 3:
            # Square root of cell volume multiplied by 1/8
            v = np.sqrt(0.125*M.vol)
            V3 = sdiag(np.r_[v, v, v])  # We will multiply on each side to keep symmetry

            Pxxx = _getFacePxxx(M)
            P000 = V3*Pxxx('fXm', 'fYm', 'fZm')
            P100 = V3*Pxxx('fXp', 'fYm', 'fZm')
            P010 = V3*Pxxx('fXm', 'fYp', 'fZm')
            P110 = V3*Pxxx('fXp', 'fYp', 'fZm')
            P001 = V3*Pxxx('fXm', 'fYm', 'fZp')
            P101 = V3*Pxxx('fXp', 'fYm', 'fZp')
            P011 = V3*Pxxx('fXm', 'fYp', 'fZp')
            P111 = V3*Pxxx('fXp', 'fYp', 'fZp')

        Mu = _makeTensor(M, mu)
        A = P000.T*Mu*P000 + P100.T*Mu*P100
        P = [P000, P100]

        if M.dim > 1:
            A = A + P010.T*Mu*P010 + P110.T*Mu*P110
            P += [P010, P110]
        if M.dim > 2:
            A = A + P001.T*Mu*P001 + P101.T*Mu*P101 + P011.T*Mu*P011 + P111.T*Mu*P111
            P += [P001, P101, P011,  P111]
        if returnP:
            return A, P
        else:
            return A

    def getEdgeInnerProduct(M, sigma=None, returnP=False):
        """
            :param numpy.array sigma: material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))
            :param bool returnP: returns the projection matrices
            :rtype: scipy.csr_matrix
            :return: M, the inner product matrix (sum(nE), sum(nE))


            Depending on the number of columns (either 1, 3, or 6) of sigma, the material property is interpreted as follows:

            .. math::
                \Sigma = \left[\\begin{matrix} \sigma_{1} & 0 & 0 \\\\ 0 & \sigma_{1} & 0 \\\\ 0 & 0 & \sigma_{1}  \end{matrix}\\right]

                \Sigma = \left[\\begin{matrix} \sigma_{1} & 0 & 0 \\\\ 0 & \sigma_{2} & 0 \\\\ 0 & 0 & \sigma_{3}  \end{matrix}\\right]

                \Sigma = \left[\\begin{matrix} \sigma_{1} & \sigma_{4} & \sigma_{5} \\\\ \sigma_{4} & \sigma_{2} & \sigma_{6} \\\\ \sigma_{5} & \sigma_{6} & \sigma_{3}  \end{matrix}\\right]

            What is returned:

            .. math::
                \mathbf{M}(\Sigma) = {1\over 8}
                    \left(\sum_{i=1}^8
                    \mathbf{J}_c^{-\\top} \sqrt{v_{\\text{cell}}} \Sigma \sqrt{v_{\\text{cell}}}  \mathbf{J}_c
                    \\right)

            If requested (returnP=True) the projection matricies are returned as well (ordered by nodes)::

                P = [P000, P100, P010, P110, P001, P101, P011, P111]

            Here each P (3*nC, sum(nE)) is a combination of the projection, volume, and any normalization to Cartesian coordinates:

            .. math::
                \mathbf{P}_{(i)} =  \sqrt{ {1\over 8} v_{\\text{cell}}} \overbrace{\mathbf{N}_{(i)}^{-1}}^{\\text{LOM only}} \mathbf{Q}_{(i)}

            Note that this is completed for each cell in the mesh at the same time.

            **For 2D:**

            Depending on the number of columns (either 1, 2, or 3) of sigma, the material property is interpreted as follows:

            .. math::
                \Sigma = \left[\\begin{matrix} \sigma_{1} & 0 \\\\ 0 & \sigma_{1} \end{matrix}\\right]

                \Sigma = \left[\\begin{matrix} \sigma_{1} & 0 \\\\ 0 & \sigma_{2} \end{matrix}\\right]

                \Sigma = \left[\\begin{matrix} \sigma_{1} & \sigma_{3} \\\\ \sigma_{3} & \sigma_{2} \end{matrix}\\right]


            .. math::

                \mathbf{M}(\Sigma) = {1\over 4}
                    \left(\sum_{i=1}^4
                    \mathbf{J}_c^{-\\top} \sqrt{v_{\\text{cell}}} \Sigma \sqrt{v_{\\text{cell}}}  \mathbf{J}_c
                    \\right)


            If requested (returnP=True) the projection matricies are returned as well (ordered by nodes)::

                P = [P00, P10, P01, P11]

            Here each P (2*nC, sum(nE)) is a combination of the projection, volume, and any normalization to Cartesian coordinates:

            .. math::
                \mathbf{P}_{(i)} =  \sqrt{ {1\over 4} v_{\\text{cell}}} \overbrace{\mathbf{N}_{(i)}^{-1}}^{\\text{LOM only}} \mathbf{Q}_{(i)}

            Note that this is completed for each cell in the mesh at the same time.

        """
        if M.dim == 1:
            raise NotImplementedError('getEdgeInnerProduct not implemented for 1D')
        # We will multiply by V on each side to keep symmetry
        elif M.dim == 2:
            # Square root of cell volume multiplied by 1/4
            v = np.sqrt(0.25*M.vol)
            V = sdiag(np.r_[v, v])
            eP = _getEdgePxx(M)
            P000 = V*eP('eX0', 'eY0')
            P100 = V*eP('eX0', 'eY1')
            P010 = V*eP('eX1', 'eY0')
            P110 = V*eP('eX1', 'eY1')
        elif M.dim == 3:
            # Square root of cell volume multiplied by 1/8
            v = np.sqrt(0.125*M.vol)
            V = sdiag(np.r_[v, v, v])
            eP = _getEdgePxxx(M)
            P000 = V*eP('eX0', 'eY0', 'eZ0')
            P100 = V*eP('eX0', 'eY1', 'eZ1')
            P010 = V*eP('eX1', 'eY0', 'eZ2')
            P110 = V*eP('eX1', 'eY1', 'eZ3')
            P001 = V*eP('eX2', 'eY2', 'eZ0')
            P101 = V*eP('eX2', 'eY3', 'eZ1')
            P011 = V*eP('eX3', 'eY2', 'eZ2')
            P111 = V*eP('eX3', 'eY3', 'eZ3')

        Sigma = _makeTensor(M, sigma)
        A = P000.T*Sigma*P000 + P100.T*Sigma*P100 + P010.T*Sigma*P010 + P110.T*Sigma*P110
        P = [P000, P100, P010, P110]
        if M.dim == 3:
            A = A + P001.T*Sigma*P001 + P101.T*Sigma*P101 + P011.T*Sigma*P011 + P111.T*Sigma*P111
            P += [P001, P101, P011,  P111]
        if returnP:
            return A, P
        else:
            return A

# ------------------------ Geometries ------------------------------
#
#
#         node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
#              /                                    /
#             /                                    / |
#         edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
#           /                                    /   |
#          /                                    /    |
#    node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
#         |                                     |    |
#         |                                     |   node(i+1,j+1,k+1)
#         |                                     |    /
#    edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1,k)
#         |                                     |  /
#         |                                     | /
#         |                                     |/
#    node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)

def _makeTensor(M, sigma):
    if sigma is None:  # default is ones
        sigma = np.ones((M.nC, 1))
    elif type(sigma) is float:
       sigma = np.ones(self.nC)*sigma

    if M.dim == 1:
        if sigma.size == M.nC:  # Isotropic!
            sigma = mkvc(sigma)  # ensure it is a vector.
            Sigma = sdiag(sigma)
        else:
            raise Exception('Unexpected shape of sigma')
    elif M.dim == 2:
        if sigma.size == M.nC:  # Isotropic!
            sigma = mkvc(sigma)  # ensure it is a vector.
            Sigma = sdiag(np.r_[sigma, sigma])
        elif sigma.shape[1] == 2:  # Diagonal tensor
            Sigma = sdiag(np.r_[sigma[:, 0], sigma[:, 1]])
        elif sigma.shape[1] == 3:  # Fully anisotropic
            row1 = sp.hstack((sdiag(sigma[:, 0]), sdiag(sigma[:, 2])))
            row2 = sp.hstack((sdiag(sigma[:, 2]), sdiag(sigma[:, 1])))
            Sigma = sp.vstack((row1, row2))
        else:
            raise Exception('Unexpected shape of sigma')
    elif M.dim == 3:
        if sigma.size == M.nC:  # Isotropic!
            sigma = mkvc(sigma)  # ensure it is a vector.
            Sigma = sdiag(np.r_[sigma, sigma, sigma])
        elif sigma.shape[1] == 3:  # Diagonal tensor
            Sigma = sdiag(np.r_[sigma[:, 0], sigma[:, 1], sigma[:, 2]])
        elif sigma.shape[1] == 6:  # Fully anisotropic
            row1 = sp.hstack((sdiag(sigma[:, 0]), sdiag(sigma[:, 3]), sdiag(sigma[:, 4])))
            row2 = sp.hstack((sdiag(sigma[:, 3]), sdiag(sigma[:, 1]), sdiag(sigma[:, 5])))
            row3 = sp.hstack((sdiag(sigma[:, 4]), sdiag(sigma[:, 5]), sdiag(sigma[:, 2])))
            Sigma = sp.vstack((row1, row2, row3))
        else:
            raise Exception('Unexpected shape of sigma')
    return Sigma


def _getFacePx(M):
    assert M._meshType == 'TENSOR', 'Only supported for a tensor mesh'
    return _getFacePx_Rectangular(M)

def _getFacePxx(M):
    if M._meshType == 'TREE':
        return M._getFacePxx

    return _getFacePxx_Rectangular(M)

def _getFacePxxx(M):
    if M._meshType == 'TREE':
        return M._getFacePxxx

    return _getFacePxxx_Rectangular(M)

def _getEdgePxx(M):
    if M._meshType == 'TREE':
        return M._getEdgePxx

    return _getEdgePxx_Rectangular(M)

def _getEdgePxxx(M):
    if M._meshType == 'TREE':
        return M._getEdgePxxx

    return _getEdgePxxx_Rectangular(M)

def _getFacePx_Rectangular(M):
    """Returns a function for creating projection matrices

    """
    ii = np.int64(range(M.nCx))

    def Px(xFace):
        """
            xFace is 'fXp' or 'fXm'
        """
        posFx = 0 if xFace == 'fXm' else 1
        IND = ii + posFx
        PX = sp.csr_matrix((np.ones(M.nC), (range(M.nC), IND)), shape=(M.nC, M.nF))
        return PX

    return Px

def _getFacePxx_Rectangular(M):
    """returns a function for creating projection matrices

        Mats takes you from faces a subset of all faces on only the
        faces that you ask for.

        These are centered around a single nodes.

        For example, if this was your entire mesh:

                        f3(Yp)
                  2_______________3
                  |               |
                  |               |
                  |               |
          f0(Xm)  |       x       |  f1(Xp)
                  |               |
                  |               |
                  |_______________|
                  0               1
                        f2(Ym)

        Pxx('m','m') = | 1, 0, 0, 0 |
                       | 0, 0, 1, 0 |

        Pxx('p','m') = | 0, 1, 0, 0 |
                       | 0, 0, 1, 0 |

        """
    i, j = np.int64(range(M.nCx)), np.int64(range(M.nCy))

    iijj = ndgrid(i, j)
    ii, jj = iijj[:, 0], iijj[:, 1]

    if M._meshType == 'LOM':
        fN1 = M.r(M.normals, 'F', 'Fx', 'M')
        fN2 = M.r(M.normals, 'F', 'Fy', 'M')

    def Pxx(xFace, yFace):
        """
            xFace is 'fXp' or 'fXm'
            yFace is 'fYp' or 'fYm'
        """
        # no | node      | f1     | f2
        # 00 | i  ,j     | i  , j | i, j
        # 10 | i+1,j     | i+1, j | i, j
        # 01 | i  ,j+1   | i  , j | i, j+1
        # 11 | i+1,j+1   | i+1, j | i, j+1

        posFx = 0 if xFace == 'fXm' else 1
        posFy = 0 if yFace == 'fYm' else 1

        ind1 = sub2ind(M.vnFx, np.c_[ii + posFx, jj])
        ind2 = sub2ind(M.vnFy, np.c_[ii, jj + posFy]) + M.nFx

        IND = np.r_[ind1, ind2].flatten()

        PXX = sp.csr_matrix((np.ones(2*M.nC), (range(2*M.nC), IND)), shape=(2*M.nC, M.nF))

        if M._meshType == 'LOM':
            I2x2 = inv2X2BlockDiagonal(getSubArray(fN1[0], [i + posFx, j]), getSubArray(fN1[1], [i + posFx, j]),
                                       getSubArray(fN2[0], [i, j + posFy]), getSubArray(fN2[1], [i, j + posFy]))
            PXX = I2x2 * PXX

        return PXX

    return Pxx

def _getFacePxxx_Rectangular(M):
    """returns a function for creating projection matrices

        Mats takes you from faces a subset of all faces on only the
        faces that you ask for.

        These are centered around a single nodes.
    """

    i, j, k = np.int64(range(M.nCx)), np.int64(range(M.nCy)), np.int64(range(M.nCz))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    if M._meshType == 'LOM':
        fN1 = M.r(M.normals, 'F', 'Fx', 'M')
        fN2 = M.r(M.normals, 'F', 'Fy', 'M')
        fN3 = M.r(M.normals, 'F', 'Fz', 'M')

    def Pxxx(xFace, yFace, zFace):
        """
            xFace is 'fXp' or 'fXm'
            yFace is 'fYp' or 'fYm'
            zFace is 'fZp' or 'fZm'
        """

        # no  | node        | f1        | f2        | f3
        # 000 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k
        # 100 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k
        # 010 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k
        # 110 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k
        # 001 | i  ,j  ,k+1 | i  , j, k | i, j  , k | i, j, k+1
        # 101 | i+1,j  ,k+1 | i+1, j, k | i, j  , k | i, j, k+1
        # 011 | i  ,j+1,k+1 | i  , j, k | i, j+1, k | i, j, k+1
        # 111 | i+1,j+1,k+1 | i+1, j, k | i, j+1, k | i, j, k+1

        posX = 0 if xFace == 'fXm' else 1
        posY = 0 if yFace == 'fYm' else 1
        posZ = 0 if zFace == 'fZm' else 1

        ind1 = sub2ind(M.vnFx, np.c_[ii + posX, jj, kk])
        ind2 = sub2ind(M.vnFy, np.c_[ii, jj + posY, kk]) + M.nFx
        ind3 = sub2ind(M.vnFz, np.c_[ii, jj, kk + posZ]) + M.nFx + M.nFy

        IND = np.r_[ind1, ind2, ind3].flatten()

        PXXX = sp.coo_matrix((np.ones(3*M.nC), (range(3*M.nC), IND)), shape=(3*M.nC, M.nF)).tocsr()

        if M._meshType == 'LOM':
            I3x3 = inv3X3BlockDiagonal(getSubArray(fN1[0], [i + posX, j, k]), getSubArray(fN1[1], [i + posX, j, k]), getSubArray(fN1[2], [i + posX, j, k]),
                                       getSubArray(fN2[0], [i, j + posY, k]), getSubArray(fN2[1], [i, j + posY, k]), getSubArray(fN2[2], [i, j + posY, k]),
                                       getSubArray(fN3[0], [i, j, k + posZ]), getSubArray(fN3[1], [i, j, k + posZ]), getSubArray(fN3[2], [i, j, k + posZ]))
            PXXX = I3x3 * PXXX

        return PXXX
    return Pxxx

def _getEdgePxx_Rectangular(M):
    i, j = np.int64(range(M.nCx)), np.int64(range(M.nCy))

    iijj = ndgrid(i, j)
    ii, jj = iijj[:, 0], iijj[:, 1]

    if M._meshType == 'LOM':
        eT1 = M.r(M.tangents, 'E', 'Ex', 'M')
        eT2 = M.r(M.tangents, 'E', 'Ey', 'M')

    def Pxx(xEdge, yEdge):
        # no | node      | e1      | e2
        # 00 | i  ,j     | i  ,j   | i  ,j
        # 10 | i+1,j     | i  ,j   | i+1,j
        # 01 | i  ,j+1   | i  ,j+1 | i  ,j
        # 11 | i+1,j+1   | i  ,j+1 | i+1,j
        posX = 0 if xEdge == 'eX0' else 1
        posY = 0 if yEdge == 'eY0' else 1

        ind1 = sub2ind(M.vnEx, np.c_[ii, jj + posX])
        ind2 = sub2ind(M.vnEy, np.c_[ii + posY, jj]) + M.nEx

        IND = np.r_[ind1, ind2].flatten()

        PXX = sp.coo_matrix((np.ones(2*M.nC), (range(2*M.nC), IND)), shape=(2*M.nC, M.nE)).tocsr()

        if M._meshType == 'LOM':
            I2x2 = inv2X2BlockDiagonal(getSubArray(eT1[0], [i, j + posX]), getSubArray(eT1[1], [i, j + posX]),
                                       getSubArray(eT2[0], [i + posY, j]), getSubArray(eT2[1], [i + posY, j]))
            PXX = I2x2 * PXX

        return PXX
    return Pxx

def _getEdgePxxx_Rectangular(M):
    i, j, k = np.int64(range(M.nCx)), np.int64(range(M.nCy)), np.int64(range(M.nCz))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    if M._meshType == 'LOM':
        eT1 = M.r(M.tangents, 'E', 'Ex', 'M')
        eT2 = M.r(M.tangents, 'E', 'Ey', 'M')
        eT3 = M.r(M.tangents, 'E', 'Ez', 'M')

    def Pxxx(xEdge, yEdge, zEdge):

        # no  | node        | e1          | e2          | e3
        # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
        # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
        # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
        # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
        # 001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
        # 101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
        # 011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
        # 111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k

        posX = [0,0] if xEdge == 'eX0' else [1, 0] if xEdge == 'eX1' else [0,1] if xEdge == 'eX2' else [1,1]
        posY = [0,0] if yEdge == 'eY0' else [1, 0] if yEdge == 'eY1' else [0,1] if yEdge == 'eY2' else [1,1]
        posZ = [0,0] if zEdge == 'eZ0' else [1, 0] if zEdge == 'eZ1' else [0,1] if zEdge == 'eZ2' else [1,1]

        ind1 = sub2ind(M.vnEx, np.c_[ii, jj + posX[0], kk + posX[1]])
        ind2 = sub2ind(M.vnEy, np.c_[ii + posY[0], jj, kk + posY[1]]) + M.nEx
        ind3 = sub2ind(M.vnEz, np.c_[ii + posZ[0], jj + posZ[1], kk]) + M.nEx + M.nEy

        IND = np.r_[ind1, ind2, ind3].flatten()

        PXXX = sp.coo_matrix((np.ones(3*M.nC), (range(3*M.nC), IND)), shape=(3*M.nC, M.nE)).tocsr()

        if M._meshType == 'LOM':
            I3x3 = inv3X3BlockDiagonal(getSubArray(eT1[0], [i, j + posX[0], k + posX[1]]), getSubArray(eT1[1], [i, j + posX[0], k + posX[1]]), getSubArray(eT1[2], [i, j + posX[0], k + posX[1]]),
                                       getSubArray(eT2[0], [i + posY[0], j, k + posY[1]]), getSubArray(eT2[1], [i + posY[0], j, k + posY[1]]), getSubArray(eT2[2], [i + posY[0], j, k + posY[1]]),
                                       getSubArray(eT3[0], [i + posZ[0], j + posZ[1], k]), getSubArray(eT3[1], [i + posZ[0], j + posZ[1], k]), getSubArray(eT3[2], [i + posZ[0], j + posZ[1], k]))
            PXXX = I3x3 * PXXX

        return PXXX
    return Pxxx

if __name__ == '__main__':
    from TensorMesh import TensorMesh
    h = [np.array([1, 2, 3, 4]), np.array([1, 2, 1, 4, 2]), np.array([1, 1, 4, 1])]
    M = TensorMesh(h)
    mu = np.ones((M.nC, 6))
    A, P = M.getFaceInnerProduct(mu, returnP=True)
    B, P = M.getEdgeInnerProduct(mu, returnP=True)
