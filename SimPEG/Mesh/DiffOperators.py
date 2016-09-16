from __future__ import print_function
import numpy as np
from scipy import sparse as sp
from SimPEG.Utils import sdiag, speye, kron3, spzeros, ddx, av, avExtrap


def checkBC(bc):
    """

        Checks if boundary condition 'bc' is valid.

        Each bc must be either 'dirichlet' or 'neumann'

    """
    if(type(bc) is str):
        bc = [bc, bc]
    assert type(bc) is list, 'bc must be a list'
    assert len(bc) == 2, 'bc must have two elements'

    for bc_i in bc:
        assert type(bc_i) is str, "each bc must be a string"
        assert bc_i in ['dirichlet', 'neumann'], ("each bc must be either,"
                                                  "'dirichlet' or 'neumann'")
    return bc


def ddxCellGrad(n, bc):
    """
        Create 1D derivative operator from cell-centers to nodes this means we
        go from n to n+1

        For Cell-Centered **Dirichlet**, use a ghost point::

            (u_1 - u_g)/hf = grad

                u_g       u_1      u_2
                 *    |    *   |    *     ...
                      ^
                      0

            u_g = - u_1
            grad = 2*u1/dx
            negitive on the other side.

        For Cell-Centered **Neumann**, use a ghost point::

            (u_1 - u_g)/hf = 0

                u_g       u_1      u_2
                 *    |    *   |    *     ...

            u_g = u_1
            grad = 0;  put a zero in.

    """
    bc = checkBC(bc)

    D = sp.spdiags((np.ones((n+1, 1))*[-1, 1]).T, [-1, 0], n+1, n,
                   format="csr")
    # Set the first side
    if(bc[0] == 'dirichlet'):
        D[0, 0] = 2
    elif(bc[0] == 'neumann'):
        D[0, 0] = 0
    # Set the second side
    if(bc[1] == 'dirichlet'):
        D[-1, -1] = -2
    elif(bc[1] == 'neumann'):
        D[-1, -1] = 0
    return D


def ddxCellGradBC(n, bc):
    """
        Create 1D derivative operator from cell-centers to nodes this means we
        go from n to n+1

        For Cell-Centered **Dirichlet**, use a ghost point::

            (u_1 - u_g)/hf = grad

             u_g       u_1      u_2
              *    |    *   |    *     ...
                   ^
                  u_b

        We know the value at the boundary (u_b)::

            (u_g+u_1)/2 = u_b               (the average)
            u_g = 2*u_b - u_1

            So plug in to gradient:

            (u_1 - (2*u_b - u_1))/hf = grad
            2*(u_1-u_b)/hf = grad

        Separate, because BC are known (and can move to RHS later)::

            ( 2/hf )*u_1 + ( -2/hf )*u_b = grad

                           (   ^   ) JUST RETURN THIS


    """
    bc = checkBC(bc)

    ij   = (np.array([0, n]), np.array([0, 1]))
    vals = np.zeros(2)

    # Set the first side
    if(bc[0] == 'dirichlet'):
        vals[0] = -2
    elif(bc[0] == 'neumann'):
        vals[0] = 0
    # Set the second side
    if(bc[1] == 'dirichlet'):
        vals[1] = 2
    elif(bc[1] == 'neumann'):
        vals[1] = 0
    D = sp.csr_matrix((vals, ij), shape=(n+1, 2))
    return D


class DiffOperators(object):
    """
        Class creates the differential operators that you need!
    """
    def __init__(self):
        raise Exception('DiffOperators is a base class providing differential'
                        'operators on meshes and cannot run on its own.'
                        'Inherit to your favorite Mesh class.')

    @property
    def faceDiv(self):
        """
        Construct divergence operator (face-stg to cell-centres).
        """
        if getattr(self, '_faceDiv', None) is None:
            n = self.vnC
            # Compute faceDivergence operator on faces
            if(self.dim == 1):
                D = ddx(n[0])
            elif(self.dim == 2):
                D1 = sp.kron(speye(n[1]), ddx(n[0]))
                D2 = sp.kron(ddx(n[1]), speye(n[0]))
                D = sp.hstack((D1, D2), format="csr")
            elif(self.dim == 3):
                D1 = kron3(speye(n[2]), speye(n[1]), ddx(n[0]))
                D2 = kron3(speye(n[2]), ddx(n[1]), speye(n[0]))
                D3 = kron3(ddx(n[2]), speye(n[1]), speye(n[0]))
                D = sp.hstack((D1, D2, D3), format="csr")
            # Compute areas of cell faces & volumes
            S = self.area
            V = self.vol
            self._faceDiv = sdiag(1/V)*D*sdiag(S)
        return self._faceDiv

    @property
    def faceDivx(self):
        """
        Construct divergence operator in the x component (face-stg to
        cell-centres).
        """
        if getattr(self, '_faceDivx', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            # Compute faceDivergence operator on faces
            if(self.dim == 1):
                D1 = ddx(n[0])
            elif(self.dim == 2):
                D1 = sp.kron(speye(n[1]), ddx(n[0]))
            elif(self.dim == 3):
                D1 = kron3(speye(n[2]), speye(n[1]), ddx(n[0]))
            # Compute areas of cell faces & volumes
            S = self.r(self.area, 'F', 'Fx', 'V')
            V = self.vol
            self._faceDivx = sdiag(1/V)*D1*sdiag(S)

        return self._faceDivx

    @property
    def faceDivy(self):
        if(self.dim < 2):
            return None
        if getattr(self, '_faceDivy', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            # Compute faceDivergence operator on faces
            if(self.dim == 2):
                D2 = sp.kron(ddx(n[1]), speye(n[0]))
            elif(self.dim == 3):
                D2 = kron3(speye(n[2]), ddx(n[1]), speye(n[0]))
            # Compute areas of cell faces & volumes
            S = self.r(self.area, 'F', 'Fy', 'V')
            V = self.vol
            self._faceDivy = sdiag(1/V)*D2*sdiag(S)
        return self._faceDivy

    @property
    def faceDivz(self):
        """
        Construct divergence operator in the z component (face-stg to
        cell-centres).
        """
        if(self.dim < 3):
            return None
        if getattr(self, '_faceDivz', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            # Compute faceDivergence operator on faces
            D3 = kron3(ddx(n[2]), speye(n[1]), speye(n[0]))
            # Compute areas of cell faces & volumes
            S = self.r(self.area, 'F', 'Fz', 'V')
            V = self.vol
            self._faceDivz = sdiag(1/V)*D3*sdiag(S)
        return self._faceDivz

    @property
    def nodalGrad(self):
        """
        Construct gradient operator (nodes to edges).
        """
        if getattr(self, '_nodalGrad', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            # Compute divergence operator on faces
            if(self.dim == 1):
                G = ddx(n[0])
            elif(self.dim == 2):
                D1 = sp.kron(speye(n[1]+1), ddx(n[0]))
                D2 = sp.kron(ddx(n[1]), speye(n[0]+1))
                G = sp.vstack((D1, D2), format="csr")
            elif(self.dim == 3):
                D1 = kron3(speye(n[2]+1), speye(n[1]+1), ddx(n[0]))
                D2 = kron3(speye(n[2]+1), ddx(n[1]), speye(n[0]+1))
                D3 = kron3(ddx(n[2]), speye(n[1]+1), speye(n[0]+1))
                G = sp.vstack((D1, D2, D3), format="csr")
            # Compute lengths of cell edges
            L = self.edge
            self._nodalGrad = sdiag(1/L)*G
        return self._nodalGrad

    @property
    def nodalLaplacian(self):
        """
        Construct laplacian operator (nodes to edges).
        """
        if getattr(self, '_nodalLaplacian', None) is None:
            print('Warning: Laplacian has not been tested rigorously.')
            # The number of cell centers in each direction
            n = self.vnC
            # Compute divergence operator on faces
            if(self.dim == 1):
                D1 = sdiag(1./self.hx) * ddx(mesh.nCx)
                L = - D1.T*D1
            elif(self.dim == 2):
                D1 = sdiag(1./self.hx) * ddx(n[0])
                D2 = sdiag(1./self.hy) * ddx(n[1])
                L1 = sp.kron(speye(n[1]+1), - D1.T * D1)
                L2 = sp.kron(- D2.T * D2, speye(n[0]+1))
                L = L1 + L2
            elif(self.dim == 3):
                D1 = sdiag(1./self.hx) * ddx(n[0])
                D2 = sdiag(1./self.hy) * ddx(n[1])
                D3 = sdiag(1./self.hz) * ddx(n[2])
                L1 = kron3(speye(n[2]+1), speye(n[1]+1), - D1.T * D1)
                L2 = kron3(speye(n[2]+1), - D2.T * D2, speye(n[0]+1))
                L3 = kron3(- D3.T * D3, speye(n[1]+1), speye(n[0]+1))
                L = L1 + L2 + L3
            self._nodalLaplacian = L
        return self._nodalLaplacian

    def setCellGradBC(self, BC):
        """
        Function that sets the boundary conditions for cell-centred derivative
        operators.

        Examples::
            # Neumann in all directions
            BC = 'neumann'

            # 3D, Dirichlet in y Neumann else
            BC = ['neumann', 'dirichlet', 'neumann']

            # 3D, Neumann in x on bottom of domain,  Dirichlet else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        """

        if(type(BC) is str):
            BC = [BC]*self.dim
        if(type(BC) is list):
            assert len(BC) == self.dim, 'BC list must be the size of your mesh'
        else:
            raise Exception("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = checkBC(bc_i)

        # ensure we create a new gradient next time we call it
        self._cellGrad        = None
        self._cellGradBC      = None
        self._cellGradBC_list = BC
        return BC
    _cellGradBC_list = 'neumann'

    def _cellGradStencil(self):
        BC = self.setCellGradBC(self._cellGradBC_list)
        n = self.vnC
        if(self.dim == 1):
            G = ddxCellGrad(n[0], BC[0])
        elif(self.dim == 2):
            G1 = sp.kron(speye(n[1]), ddxCellGrad(n[0], BC[0]))
            G2 = sp.kron(ddxCellGrad(n[1], BC[1]), speye(n[0]))
            G = sp.vstack((G1, G2), format="csr")
        elif(self.dim == 3):
            G1 = kron3(speye(n[2]), speye(n[1]), ddxCellGrad(n[0], BC[0]))
            G2 = kron3(speye(n[2]), ddxCellGrad(n[1], BC[1]), speye(n[0]))
            G3 = kron3(ddxCellGrad(n[2], BC[2]), speye(n[1]), speye(n[0]))
            G = sp.vstack((G1, G2, G3), format="csr")
        return G

    @property
    def cellGrad(self):
        """
        The cell centered Gradient, takes you to cell faces.
        """
        if getattr(self, '_cellGrad', None) is None:
            G = self._cellGradStencil()
            S = self.area  # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol  # Average volume between adjacent cells
            self._cellGrad = sdiag(S/V)*G
        return self._cellGrad

    @property
    def cellGradBC(self):
        """
        The cell centered Gradient boundary condition matrix
        """
        if getattr(self, '_cellGradBC', None) is None:
            BC = self.setCellGradBC(self._cellGradBC_list)
            n = self.vnC
            if(self.dim == 1):
                G = ddxCellGradBC(n[0], BC[0])
            elif(self.dim == 2):
                G1 = sp.kron(speye(n[1]), ddxCellGradBC(n[0], BC[0]))
                G2 = sp.kron(ddxCellGradBC(n[1], BC[1]), speye(n[0]))
                G = sp.block_diag((G1, G2), format="csr")
            elif(self.dim == 3):
                G1 = kron3(speye(n[2]), speye(n[1]), ddxCellGradBC(n[0], BC[0]))
                G2 = kron3(speye(n[2]), ddxCellGradBC(n[1], BC[1]), speye(n[0]))
                G3 = kron3(ddxCellGradBC(n[2], BC[2]), speye(n[1]), speye(n[0]))
                G = sp.block_diag((G1, G2, G3), format="csr")
            # Compute areas of cell faces & volumes
            S = self.area
            V = self.aveCC2F*self.vol  # Average volume between adjacent cells
            self._cellGradBC = sdiag(S/V)*G
        return self._cellGradBC

    # def cellGradBC():
    #     doc = "The cell centered Gradient boundary condition matrix"

    #     def fget(self):
    #         if(self._cellGradBC is None):
    #             BC = self.setCellGradBC(self._cellGradBC_list)
    #             n = self.vnC
    #             if(self.dim == 1):
    #                 G = ddxCellGradBC(n[0], BC[0])
    #             elif(self.dim == 2):
    #                 G1 = sp.kron(speye(n[1]), ddxCellGradBC(n[0], BC[0]))
    #                 G2 = sp.kron(ddxCellGradBC(n[1], BC[1]), speye(n[0]))
    #                 G = sp.block_diag((G1, G2), format="csr")
    #             elif(self.dim == 3):
    #                 G1 = kron3(speye(n[2]), speye(n[1]), ddxCellGradBC(n[0], BC[0]))
    #                 G2 = kron3(speye(n[2]), ddxCellGradBC(n[1], BC[1]), speye(n[0]))
    #                 G3 = kron3(ddxCellGradBC(n[2], BC[2]), speye(n[1]), speye(n[0]))
    #                 G = sp.block_diag((G1, G2, G3), format="csr")
    #             # Compute areas of cell faces & volumes
    #             S = self.area
    #             V = self.aveCC2F*self.vol  # Average volume between adjacent cells
    #             self._cellGradBC = sdiag(S/V)*G
    #         return self._cellGradBC
    #     return locals()
    # _cellGradBC = None
    # cellGradBC = property(**cellGradBC())

    def _cellGradxStencil(self):
        BC = ['neumann', 'neumann']
        n = self.vnC
        if(self.dim == 1):
            G1 = ddxCellGrad(n[0], BC)
        elif(self.dim == 2):
            G1 = sp.kron(speye(n[1]), ddxCellGrad(n[0], BC))
        elif(self.dim == 3):
            G1 = kron3(speye(n[2]), speye(n[1]), ddxCellGrad(n[0], BC))
        return G1

    @property
    def cellGradx(self):
        """
        Cell centered Gradient in the x dimension. Has neumann boundary
        conditions.
        """
        if getattr(self, '_cellGradx', None) is None:
            G1 = self._cellGradxStencil()
            # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol
            L = self.r(self.area/V, 'F','Fx', 'V')
            self._cellGradx = sdiag(L)*G1
        return self._cellGradx

    def _cellGradyStencil(self):
        if self.dim < 2: return None
        BC = ['neumann', 'neumann']
        n = self.vnC
        if(self.dim == 2):
            G2 = sp.kron(ddxCellGrad(n[1], BC), speye(n[0]))
        elif(self.dim == 3):
            G2 = kron3(speye(n[2]), ddxCellGrad(n[1], BC), speye(n[0]))
        return G2

    @property
    def cellGrady(self):
        if self.dim < 2:
            return None
        if getattr(self, '_cellGrady', None) is None:
            G2 = self._cellGradyStencil()
            # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol
            L = self.r(self.area/V, 'F', 'Fy', 'V')
            self._cellGrady = sdiag(L)*G2
        return self._cellGrady

    def _cellGradzStencil(self):
        if self.dim < 3: return None
        BC = ['neumann', 'neumann']
        n = self.vnC
        G3 = kron3(ddxCellGrad(n[2], BC), speye(n[1]), speye(n[0]))
        return G3

    @property
    def cellGradz(self):
        """
        Cell centered Gradient in the x dimension. Has neumann boundary
        conditions.
        """
        if self.dim < 3:
            return None
        if getattr(self, '_cellGradz', None) is None:
            G3 = self._cellGradzStencil()
            # Compute areas of cell faces & volumes
            V = self.aveCC2F*self.vol
            L = self.r(self.area/V, 'F', 'Fz', 'V')
            self._cellGradz = sdiag(L)*G3
        return self._cellGradz

    @property
    def edgeCurl(self):
        """
        Construct the 3D curl operator.
        """
        if getattr(self, '_edgeCurl', None) is None:
            assert self.dim > 1, "Edge Curl only programed for 2 or 3D."

            n = self.vnC  # The number of cell centers in each direction
            L = self.edge  # Compute lengths of cell edges
            S = self.area # Compute areas of cell faces

            # Compute divergence operator on faces
            if self.dim == 2:

                D21 = sp.kron(ddx(n[1]), speye(n[0]))
                D12 = sp.kron(speye(n[1]), ddx(n[0]))
                C = sp.hstack((-D21, D12), format="csr")
                self._edgeCurl = C*sdiag(1/S)

            elif self.dim == 3:

                D32 = kron3(ddx(n[2]), speye(n[1]), speye(n[0]+1))
                D23 = kron3(speye(n[2]), ddx(n[1]), speye(n[0]+1))
                D31 = kron3(ddx(n[2]), speye(n[1]+1), speye(n[0]))
                D13 = kron3(speye(n[2]), speye(n[1]+1), ddx(n[0]))
                D21 = kron3(speye(n[2]+1), ddx(n[1]), speye(n[0]))
                D12 = kron3(speye(n[2]+1), speye(n[1]), ddx(n[0]))

                O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
                O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
                O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])

                C = sp.vstack((sp.hstack((O1, -D32, D23)),
                               sp.hstack((D31, O2, -D13)),
                               sp.hstack((-D21, D12, O3))), format="csr")

                self._edgeCurl = sdiag(1/S)*(C*sdiag(L))
        return self._edgeCurl

    def getBCProjWF(self, BC, discretization='CC'):
        """

        The weak form boundary condition projection matrices.

        Examples::
            # Neumann in all directions
            BC = 'neumann'

            # 3D, Dirichlet in y Neumann else
            BC = ['neumann', 'dirichlet', 'neumann']

            # 3D, Neumann in x on bottom of domain, Dirichlet else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        """

        if discretization is not 'CC':
            raise NotImplementedError('Boundary conditions only implemented'
                                      'for CC discretization.')

        if(type(BC) is str):
            BC = [BC for _ in self.vnC]  # Repeat the str self.dim times
        elif(type(BC) is list):
            assert len(BC) == self.dim, 'BC list must be the size of your mesh'
        else:
            raise Exception("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = checkBC(bc_i)

        def projDirichlet(n, bc):
            bc = checkBC(bc)
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            if(bc[0] == 'dirichlet'):
                vals[0] = -1
            if(bc[1] == 'dirichlet'):
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n+1, 2))

        def projNeumannIn(n, bc):
            bc = checkBC(bc)
            P = sp.identity(n+1).tocsr()
            if(bc[0] == 'neumann'):
                P = P[1:, :]
            if(bc[1] == 'neumann'):
                P = P[:-1, :]
            return P

        def projNeumannOut(n, bc):
            bc = checkBC(bc)
            ij   = ([0, 1], [0, n])
            vals = [0,0]
            if(bc[0] == 'neumann'):
                vals[0] = 1
            if(bc[1] == 'neumann'):
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(2, n+1))

        n = self.vnC
        indF = self.faceBoundaryInd
        if(self.dim == 1):
            Pbc = projDirichlet(n[0], BC[0])
            indF = indF[0] | indF[1]
            Pbc = Pbc*sdiag(self.area[indF])

            Pin = projNeumannIn(n[0], BC[0])

            Pout = projNeumannOut(n[0], BC[0])

        elif(self.dim == 2):
            Pbc1 = sp.kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = sp.kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3])]
            Pbc = Pbc*sdiag(self.area[indF])

            P1 = sp.kron(speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = sp.kron(projNeumannIn(n[1], BC[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2), format="csr")

            P1 = sp.kron(speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = sp.kron(projNeumannOut(n[1], BC[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2), format="csr")

        elif(self.dim == 3):
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2, Pbc3), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3]), (indF[4] |
                          indF[5])]
            Pbc = Pbc*sdiag(self.area[indF])

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannIn(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannIn(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2, P3), format="csr")

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannOut(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannOut(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2, P3), format="csr")

        return Pbc, Pin, Pout

    def getBCProjWF_simple(self, discretization='CC'):
        """
        The weak form boundary condition projection matrices
        when mixed boundary condition is used
        """

        if discretization is not 'CC':
            raise NotImplementedError('Boundary conditions only implemented'
                                      'for CC discretization.')

        def projBC(n):
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            vals[0] = 1
            vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n+1, 2))

        def projDirichlet(n, bc):
            bc = checkBC(bc)
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            if(bc[0] == 'dirichlet'):
                vals[0] = -1
            if(bc[1] == 'dirichlet'):
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n+1, 2))

        BC = [['dirichlet', 'dirichlet'], ['dirichlet', 'dirichlet'],
              ['dirichlet', 'dirichlet']]
        n = self.vnC
        indF = self.faceBoundaryInd

        if(self.dim == 1):
            Pbc = projDirichlet(n[0], BC[0])
            B = projBC(n[0])
            indF = indF[0] | indF[1]
            Pbc = Pbc*sdiag(self.area[indF])

        elif(self.dim == 2):
            Pbc1 = sp.kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = sp.kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2), format="csr")
            B1 = sp.kron(speye(n[1]), projBC(n[0]))
            B2 = sp.kron(projBC(n[1]), speye(n[0]))
            B = sp.block_diag((B1, B2), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3])]
            Pbc = Pbc*sdiag(self.area[indF])

        elif(self.dim == 3):
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2, Pbc3), format="csr")
            B1 = kron3(speye(n[2]), speye(n[1]), projBC(n[0]))
            B2 = kron3(speye(n[2]), projBC(n[1]), speye(n[0]))
            B3 = kron3(projBC(n[2]), speye(n[1]), speye(n[0]))
            B = sp.block_diag((B1, B2, B3), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3]), (indF[4] | indF[5])]
            Pbc = Pbc*sdiag(self.area[indF])

        return Pbc, B.T
    # --------------- Averaging ---------------------

    @property
    def aveF2CC(self):
        "Construct the averaging operator on cell faces to cell centers."
        if(self.dim == 1):
            return self.aveFx2CC
        elif(self.dim == 2):
            return (0.5)*sp.hstack((self.aveFx2CC, self.aveFy2CC),
                                    format="csr")
        elif(self.dim == 3):
            return (1./3.)*sp.hstack((self.aveFx2CC, self.aveFy2CC,
                                      self.aveFz2CC), format="csr")

    @property
    def aveF2CCV(self):
        "Construct the averaging operator on cell faces to cell centers."
        if(self.dim == 1):
            return self.aveFx2CC
        elif(self.dim == 2):
            return sp.block_diag((self.aveFx2CC, self.aveFy2CC), format="csr")
        elif(self.dim == 3):
            return sp.block_diag((self.aveFx2CC, self.aveFy2CC, self.aveFz2CC),
                                 format="csr")

    @property
    def aveFx2CC(self):
        """
        Construct the averaging operator on cell faces in the x direction to
        cell centers.
        """

        if getattr(self, '_aveFx2CC', None) is None:
            n = self.vnC
            if(self.dim == 1):
                self._aveFx2CC = av(n[0])
            elif(self.dim == 2):
                self._aveFx2CC = sp.kron(speye(n[1]), av(n[0]))
            elif(self.dim == 3):
                self._aveFx2CC = kron3(speye(n[2]), speye(n[1]), av(n[0]))
        return self._aveFx2CC

    @property
    def aveFy2CC(self):
        """
        Construct the averaging operator on cell faces in the y direction to
        cell centers.
        """
        if self.dim < 2:
            return None
        if getattr(self, '_aveFy2CC', None) is None:
            n = self.vnC
            if(self.dim == 2):
                self._aveFy2CC = sp.kron(av(n[1]), speye(n[0]))
            elif(self.dim == 3):
                self._aveFy2CC = kron3(speye(n[2]), av(n[1]), speye(n[0]))
        return self._aveFy2CC

    @property
    def aveFz2CC(self):
        """
        Construct the averaging operator on cell faces in the z direction to
        cell centers.
        """
        if self.dim < 3: return None
        if getattr(self, '_aveFz2CC', None) is None:
            n = self.vnC
            if(self.dim == 3):
                self._aveFz2CC = kron3(av(n[2]), speye(n[1]), speye(n[0]))
        return self._aveFz2CC


    @property
    def aveCC2F(self):
        "Construct the averaging operator on cell cell centers to faces."
        if getattr(self, '_aveCC2F', None) is None:
            n = self.vnC
            if(self.dim == 1):
                self._aveCC2F = avExtrap(n[0])
            elif(self.dim == 2):
                self._aveCC2F = sp.vstack((sp.kron(speye(n[1]),
                                           avExtrap(n[0])),
                                           sp.kron(avExtrap(n[1]),
                                           speye(n[0]))), format="csr")
            elif(self.dim == 3):
                self._aveCC2F = sp.vstack((kron3(speye(n[2]), speye(n[1]),
                                                 avExtrap(n[0])),
                                           kron3(speye(n[2]), avExtrap(n[1]),
                                                 speye(n[0])),
                                           kron3(avExtrap(n[2]), speye(n[1]),
                                                 speye(n[0]))),
                                          format="csr")
        return self._aveCC2F

    @property
    def aveE2CC(self):
        "Construct the averaging operator on cell edges to cell centers."
        if(self.dim == 1):
            return self.aveEx2CC
        elif(self.dim == 2):
            return 0.5*sp.hstack((self.aveEx2CC, self.aveEy2CC), format="csr")
        elif(self.dim == 3):
            return (1./3)*sp.hstack((self.aveEx2CC, self.aveEy2CC,
                                     self.aveEz2CC), format="csr")

    @property
    def aveE2CCV(self):
        "Construct the averaging operator on cell edges to cell centers."
        if(self.dim == 1):
            return self.aveEx2CC
        elif(self.dim == 2):
            return sp.block_diag((self.aveEx2CC, self.aveEy2CC), format="csr")
        elif(self.dim == 3):
            return sp.block_diag((self.aveEx2CC, self.aveEy2CC, self.aveEz2CC),
                                 format="csr")

    @property
    def aveEx2CC(self):
        """
        Construct the averaging operator on cell edges in the x direction to
        cell centers.
        """
        if getattr(self, '_aveEx2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 1):
                self._aveEx2CC = speye(n[0])
            elif(self.dim == 2):
                self._aveEx2CC = sp.kron(av(n[1]), speye(n[0]))
            elif(self.dim == 3):
                self._aveEx2CC = kron3(av(n[2]), av(n[1]), speye(n[0]))
        return self._aveEx2CC

    @property
    def aveEy2CC(self):
        """
        Construct the averaging operator on cell edges in the y direction to
        cell centers.
        """
        if self.dim < 2:
            return None
        if getattr(self, '_aveEy2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 2):
                self._aveEy2CC = sp.kron(speye(n[1]), av(n[0]))
            elif(self.dim == 3):
                self._aveEy2CC = kron3(av(n[2]), speye(n[1]), av(n[0]))
        return self._aveEy2CC

    @property
    def aveEz2CC(self):
        """
        Construct the averaging operator on cell edges in the z direction to
        cell centers.
        """
        if self.dim < 3:
            return None
        if getattr(self, '_aveEz2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 3):
                self._aveEz2CC = kron3(speye(n[2]), av(n[1]), av(n[0]))
        return self._aveEz2CC

    @property
    def aveN2CC(self):
        "Construct the averaging operator on cell nodes to cell centers."
        if getattr(self, '_aveN2CC', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 1):
                self._aveN2CC = av(n[0])
            elif(self.dim == 2):
                self._aveN2CC = sp.kron(av(n[1]), av(n[0])).tocsr()
            elif(self.dim == 3):
                self._aveN2CC = kron3(av(n[2]), av(n[1]), av(n[0])).tocsr()
        return self._aveN2CC

    @property
    def aveN2E(self):
        """
        Construct the averaging operator on cell nodes to cell edges, keeping
        each dimension separate.
        """

        if getattr(self, '_aveN2E', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 1):
                self._aveN2E = av(n[0])
            elif(self.dim == 2):
                self._aveN2E = sp.vstack((sp.kron(speye(n[1]+1), av(n[0])),
                                          sp.kron(av(n[1]), speye(n[0]+1))),
                                         format="csr")
            elif(self.dim == 3):
                self._aveN2E = sp.vstack((kron3(speye(n[2]+1), speye(n[1]+1),
                                                av(n[0])),
                                          kron3(speye(n[2]+1), av(n[1]),
                                                speye(n[0]+1)),
                                          kron3(av(n[2]), speye(n[1]+1),
                                                speye(n[0]+1))),
                                         format="csr")
        return self._aveN2E

    @property
    def aveN2F(self):
        """
        Construct the averaging operator on cell nodes to cell faces, keeping
        each dimension separate.
        """
        if getattr(self, '_aveN2F', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if(self.dim == 1):
                self._aveN2F = av(n[0])
            elif(self.dim == 2):
                self._aveN2F = sp.vstack((sp.kron(av(n[1]), speye(n[0]+1)),
                                          sp.kron(speye(n[1]+1), av(n[0]))),
                                         format="csr")
            elif(self.dim == 3):
                self._aveN2F = sp.vstack((kron3(av(n[2]), av(n[1]),
                                                speye(n[0]+1)),
                                          kron3(av(n[2]), speye(n[1]+1),
                                                av(n[0])),
                                          kron3(speye(n[2]+1), av(n[1]),
                                                av(n[0]))),
                                         format="csr")
        return self._aveN2F
