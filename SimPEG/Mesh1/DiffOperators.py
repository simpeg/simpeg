import numpy as np
from scipy import sparse as sp
from SimPEG.Utils import mkvc, sdiag, speye, kron3, spzeros, ddx, av, avExtrap


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
        assert bc_i in ['dirichlet', 'neumann'], "each bc must be either, 'dirichlet' or 'neumann'"
    return bc


def ddxCellGrad(n, bc):
    """
        Create 1D derivative operator from cell-centers to nodes this means we go from n to n+1

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

    D = sp.spdiags((np.ones((n+1, 1))*[-1, 1]).T, [-1, 0], n+1, n, format="csr")
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

        Create 1D derivative operator from cell-centers to nodes this means we go from n to n+1

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

    ij   = (np.array([0, n]),np.array([0, 1]))
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
    D = sp.csr_matrix((vals, ij), shape=(n+1,2))
    return D


class DiffOperators(object):
    """
        Class creates the differential operators that you need!
    """
    def __init__(self):
        raise Exception('DiffOperators is a base class providing differential operators on meshes and cannot run on its own. Inherit to your favorite Mesh class.')

    def faceDiv():
        doc = "Construct divergence operator (face-stg to cell-centres)."

        def fget(self):
            if(self._faceDiv is None):
                # The number of cell centers in each direction
                n = self.n
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
        return locals()
    _faceDiv = None
    faceDiv = property(**faceDiv())

    def faceDivx():
        doc = "Construct divergence operator in the x component (face-stg to cell-centres)."

        def fget(self):
            if(self._faceDivx is None):
                # The number of cell centers in each direction
                n = self.n
                # Compute faceDivergence operator on faces
                if(self.dim == 1):
                    D1 = ddx(n[0])
                elif(self.dim == 2):
                    D1 = sp.kron(speye(n[1]), ddx(n[0]))
                elif(self.dim == 3):
                    D1 = kron3(speye(n[2]), speye(n[1]), ddx(n[0]))
                # Compute areas of cell faces & volumes
                S = self.r(self.area, 'F','Fx', 'V')
                V = self.vol
                self._faceDivx = sdiag(1/V)*D1*sdiag(S)

            return self._faceDivx
        return locals()
    _faceDivx = None
    faceDivx = property(**faceDivx())

    def faceDivy():
        doc = "Construct divergence operator in the y component (face-stg to cell-centres)."

        def fget(self):
            if(self.dim < 2): return None
            if(self._faceDivy is None):
                # The number of cell centers in each direction
                n = self.n
                # Compute faceDivergence operator on faces
                if(self.dim == 2):
                    D2 = sp.kron(ddx(n[1]), speye(n[0]))
                elif(self.dim == 3):
                    D2 = kron3(speye(n[2]), ddx(n[1]), speye(n[0]))
                # Compute areas of cell faces & volumes
                S = self.r(self.area, 'F','Fy', 'V')
                V = self.vol
                self._faceDivy = sdiag(1/V)*D2*sdiag(S)

            return self._faceDivy
        return locals()
    _faceDivy = None
    faceDivy = property(**faceDivy())

    def faceDivz():
        doc = "Construct divergence operator in the z component (face-stg to cell-centres)."

        def fget(self):
            if(self.dim < 3): return None
            if(self._faceDivz is None):
                # The number of cell centers in each direction
                n = self.n
                # Compute faceDivergence operator on faces
                D3 = kron3(ddx(n[2]), speye(n[1]), speye(n[0]))
                # Compute areas of cell faces & volumes
                S = self.r(self.area, 'F','Fz', 'V')
                V = self.vol
                self._faceDivz = sdiag(1/V)*D3*sdiag(S)

            return self._faceDivz
        return locals()
    _faceDivz = None
    faceDivz = property(**faceDivz())

    def nodalGrad():
        doc = "Construct gradient operator (nodes to edges)."

        def fget(self):
            if(self._nodalGrad is None):
                # The number of cell centers in each direction
                n = self.n
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
        return locals()
    _nodalGrad = None
    nodalGrad = property(**nodalGrad())

    def nodalLaplacian():
        doc = "Construct laplacian operator (nodes to edges)."

        def fget(self):
            if(self._nodalLaplacian is None):
                print 'Warning: Laplacian has not been tested rigorously.'
                # The number of cell centers in each direction
                n = self.n
                # Compute divergence operator on faces
                if(self.dim == 1):
                    D1 = sdiag(1./self.hx) * ddx(mesh.nCx)
                    L  = - D1.T*D1
                elif(self.dim == 2):
                    D1 = sdiag(1./self.hx) * ddx(n[0])
                    D2 = sdiag(1./self.hy) * ddx(n[1])
                    L1 = sp.kron(speye(n[1]+1), - D1.T * D1)
                    L2 = sp.kron(- D2.T * D2, speye(n[0]+1))
                    L  = L1 + L2
                elif(self.dim == 3):
                    D1 = sdiag(1./self.hx) * ddx(n[0])
                    D2 = sdiag(1./self.hy) * ddx(n[1])
                    D3 = sdiag(1./self.hz) * ddx(n[2])
                    L1 = kron3(speye(n[2]+1), speye(n[1]+1), - D1.T * D1)
                    L2 = kron3(speye(n[2]+1), - D2.T * D2, speye(n[0]+1))
                    L3 = kron3(- D3.T * D3, speye(n[1]+1), speye(n[0]+1))
                    L  = L1 + L2 + L3
                self._nodalLaplacian = L
            return self._nodalLaplacian
        return locals()
    _nodalLaplacian = None
    nodalLaplacian = property(**nodalLaplacian())

    def setCellGradBC(self, BC):
        """
        Function that sets the boundary conditions for cell-centred derivative operators.

        Examples::

            BC = 'neumann'                                            # Neumann in all directions
            BC = ['neumann', 'dirichlet', 'neumann']                  # 3D, Dirichlet in y Neumann else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet'] # 3D, Neumann in x on bottom of domain,
                                                                      #     Dirichlet else

        """
        if(type(BC) is str):
            BC = [BC for _ in self.n]  # Repeat the str self.dim times
        elif(type(BC) is list):
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

    def cellGrad():
        doc = "The cell centered Gradient, takes you to cell faces."

        def fget(self):
            if(self._cellGrad is None):
                BC = self.setCellGradBC(self._cellGradBC_list)
                n = self.n
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
                # Compute areas of cell faces & volumes
                S = self.area
                V = self.aveCC2F*self.vol  # Average volume between adjacent cells
                self._cellGrad = sdiag(S/V)*G
            return self._cellGrad
        return locals()
    _cellGrad = None
    cellGrad = property(**cellGrad())

    def cellGradBC():
        doc = "The cell centered Gradient boundary condition matrix"

        def fget(self):
            if(self._cellGradBC is None):
                BC = self.setCellGradBC(self._cellGradBC_list)
                n = self.n
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
        return locals()
    _cellGradBC = None
    cellGradBC = property(**cellGradBC())

    def cellGradx():
        doc = "Cell centered Gradient in the x dimension. Has neumann boundary conditions."

        def fget(self):
            if getattr(self, '_cellGradx', None) is None:
                BC = ['neumann', 'neumann']
                n = self.n
                if(self.dim == 1):
                    G1 = ddxCellGrad(n[0], BC)
                elif(self.dim == 2):
                    G1 = sp.kron(speye(n[1]), ddxCellGrad(n[0], BC))
                elif(self.dim == 3):
                    G1 = kron3(speye(n[2]), speye(n[1]), ddxCellGrad(n[0], BC))
                # Compute areas of cell faces & volumes
                V = self.aveCC2F*self.vol
                L = self.r(self.area/V, 'F','Fx', 'V')
                self._cellGradx = sdiag(L)*G1
            return self._cellGradx
        return locals()
    cellGradx = property(**cellGradx())

    def cellGrady():
        doc = "Cell centered Gradient in the x dimension. Has neumann boundary conditions."
        def fget(self):
            if self.dim < 2: return None
            if getattr(self, '_cellGrady', None) is None:
                BC = ['neumann', 'neumann']
                n = self.n
                if(self.dim == 2):
                    G2 = sp.kron(ddxCellGrad(n[1], BC), speye(n[0]))
                elif(self.dim == 3):
                    G2 = kron3(speye(n[2]), ddxCellGrad(n[1], BC), speye(n[0]))
                # Compute areas of cell faces & volumes
                V = self.aveCC2F*self.vol
                L = self.r(self.area/V, 'F','Fy', 'V')
                self._cellGrady = sdiag(L)*G2
            return self._cellGrady
        return locals()
    cellGrady = property(**cellGrady())

    def cellGradz():
        doc = "Cell centered Gradient in the x dimension. Has neumann boundary conditions."
        def fget(self):
            if self.dim < 3: return None
            if getattr(self, '_cellGradz', None) is None:
                BC = ['neumann', 'neumann']
                n = self.n
                G3 = kron3(ddxCellGrad(n[2], BC), speye(n[1]), speye(n[0]))
                # Compute areas of cell faces & volumes
                V = self.aveCC2F*self.vol
                L = self.r(self.area/V, 'F','Fz', 'V')
                self._cellGradz = sdiag(L)*G3
            return self._cellGradz
        return locals()
    cellGradz = property(**cellGradz())

    def edgeCurl():
        doc = "Construct the 3D curl operator."

        def fget(self):
            if(self._edgeCurl is None):
                # The number of cell centers in each direction
                n1 = self.nCx
                n2 = self.nCy
                n3 = self.nCz

                # Compute lengths of cell edges
                L = self.edge

                # Compute areas of cell faces
                S = self.area

                # Compute divergence operator on faces
                d1 = ddx(n1)
                d2 = ddx(n2)
                d3 = ddx(n3)

                D32 = kron3(d3, speye(n2), speye(n1+1))
                D23 = kron3(speye(n3), d2, speye(n1+1))
                D31 = kron3(d3, speye(n2+1), speye(n1))
                D13 = kron3(speye(n3), speye(n2+1), d1)
                D21 = kron3(speye(n3+1), d2, speye(n1))
                D12 = kron3(speye(n3+1), speye(n2), d1)

                O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
                O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
                O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])

                C = sp.vstack((sp.hstack((O1, -D32, D23)),
                               sp.hstack((D31, O2, -D13)),
                               sp.hstack((-D21, D12, O3))), format="csr")

                self._edgeCurl = sdiag(1/S)*(C*sdiag(L))
            return self._edgeCurl
        return locals()
    _edgeCurl = None
    edgeCurl = property(**edgeCurl())

    # --------------- Averaging ---------------------

    def aveF2CC():
        doc = "Construct the averaging operator on cell faces to cell centers."

        def fget(self):
            if(self._aveF2CC is None):
                n = self.n
                if(self.dim == 1):
                    self._aveF2CC = av(n[0])
                elif(self.dim == 2):
                    self._aveF2CC = (0.5)*sp.hstack((sp.kron(speye(n[1]), av(n[0])),
                                                     sp.kron(av(n[1]), speye(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._aveF2CC = (1./3.)*sp.hstack((kron3(speye(n[2]), speye(n[1]), av(n[0])),
                                                       kron3(speye(n[2]), av(n[1]), speye(n[0])),
                                                       kron3(av(n[2]), speye(n[1]), speye(n[0]))), format="csr")
            return self._aveF2CC
        return locals()
    _aveF2CC = None
    aveF2CC = property(**aveF2CC())

    def aveCC2F():
        doc = "Construct the averaging operator on cell cell centers to faces."

        def fget(self):
            if(self._aveCC2F is None):
                n = self.n
                if(self.dim == 1):
                    self._aveCC2F = avExtrap(n[0])
                elif(self.dim == 2):
                    self._aveCC2F = sp.vstack((sp.kron(speye(n[1]), avExtrap(n[0])),
                                               sp.kron(avExtrap(n[1]), speye(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._aveCC2F = sp.vstack((kron3(speye(n[2]), speye(n[1]), avExtrap(n[0])),
                                               kron3(speye(n[2]), avExtrap(n[1]), speye(n[0])),
                                               kron3(avExtrap(n[2]), speye(n[1]), speye(n[0]))), format="csr")
            return self._aveCC2F
        return locals()
    _aveCC2F = None
    aveCC2F = property(**aveCC2F())

    def aveE2CC():
        doc = "Construct the averaging operator on cell edges to cell centers."

        def fget(self):
            if(self._aveE2CC is None):
                # The number of cell centers in each direction
                n = self.n
                if(self.dim == 1):
                    raise Exception('Edge Averaging does not make sense in 1D: Use Identity?')
                elif(self.dim == 2):
                    self._aveE2CC = 0.5*sp.hstack((sp.kron(av(n[1]), speye(n[0])),
                                               sp.kron(speye(n[1]), av(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._aveE2CC = (1./3)*sp.hstack((kron3(av(n[2]), av(n[1]), speye(n[0])),
                                               kron3(av(n[2]), speye(n[1]), av(n[0])),
                                               kron3(speye(n[2]), av(n[1]), av(n[0]))), format="csr")
            return self._aveE2CC
        return locals()
    _aveE2CC = None
    aveE2CC = property(**aveE2CC())

    def aveN2CC():
        doc = "Construct the averaging operator on cell nodes to cell centers."

        def fget(self):
            if(self._aveN2CC is None):
                # The number of cell centers in each direction
                n = self.n
                if(self.dim == 1):
                    self._aveN2CC = av(n[0])
                elif(self.dim == 2):
                    self._aveN2CC = sp.kron(av(n[1]), av(n[0])).tocsr()
                elif(self.dim == 3):
                    self._aveN2CC = kron3(av(n[2]), av(n[1]), av(n[0])).tocsr()
            return self._aveN2CC
        return locals()
    _aveN2CC = None
    aveN2CC = property(**aveN2CC())

    def aveN2E():
        doc = "Construct the averaging operator on cell nodes to cell edges, keeping each dimension separate."

        def fget(self):
            if(self._aveN2E is None):
                # The number of cell centers in each direction
                n = self.n
                if(self.dim == 1):
                    self._aveN2E = av(n[0])
                elif(self.dim == 2):
                    self._aveN2E = sp.vstack((sp.kron(speye(n[1]+1), av(n[0])),
                                              sp.kron(av(n[1]), speye(n[0]+1))), format="csr")
                elif(self.dim == 3):
                    self._aveN2E = sp.vstack((kron3(speye(n[2]+1), speye(n[1]+1), av(n[0])),
                                              kron3(speye(n[2]+1), av(n[1]), speye(n[0]+1)),
                                              kron3(av(n[2]), speye(n[1]+1), speye(n[0]+1))), format="csr")
            return self._aveN2E
        return locals()
    _aveN2E = None
    aveN2E = property(**aveN2E())

    def aveN2F():
        doc = "Construct the averaging operator on cell nodes to cell faces, keeping each dimension separate."

        def fget(self):
            if(self._aveN2F is None):
                # The number of cell centers in each direction
                n = self.n
                if(self.dim == 1):
                    self._aveN2F = av(n[0])
                elif(self.dim == 2):
                    self._aveN2F = sp.vstack((sp.kron(av(n[1]), speye(n[0]+1)),
                                              sp.kron(speye(n[1]+1), av(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._aveN2F = sp.vstack((kron3(av(n[2]), av(n[1]), speye(n[0]+1)),
                                              kron3(av(n[2]), speye(n[1]+1), av(n[0])),
                                              kron3(speye(n[2]+1), av(n[1]), av(n[0]))), format="csr")
            return self._aveN2F
        return locals()
    _aveN2F = None
    aveN2F = property(**aveN2F())

    # --------------- Methods ---------------------

    def getMass(self, materialProp=None, loc='e'):
        """ Produces mass matricies.

        :param str loc: Average to location: 'e'-edges, 'f'-faces
        :param None,float,numpy.ndarray materialProp: property to be averaged (see below)
        :rtype: scipy.sparse.csr.csr_matrix
        :return: M, the mass matrix

        materialProp can be::

            None            -> takes materialProp = 1 (default)
            float           -> a constant value for entire domain
            numpy.ndarray   -> if materialProp.size == self.nC
                                    3D property model
                               if materialProp.size = self.nCz
                                    1D (layered eath) property model
        """
        if materialProp is None:
            materialProp = np.ones(self.nC)
        elif type(materialProp) is float:
            materialProp = np.ones(self.nC)*materialProp
        elif materialProp.shape == (self.nCz,):
            materialProp = materialProp.repeat(self.nCx*self.nCy)
        materialProp = mkvc(materialProp)
        assert materialProp.shape == (self.nC,), "materialProp incorrect shape"

        if loc=='e':
            Av = self.aveE2CC
        elif loc=='f':
            Av = self.aveF2CC
        else:
            raise ValueError('Invalid loc')

        diag = Av.T * (self.vol * mkvc(materialProp))

        return sdiag(diag)

    def getEdgeMass(self, materialProp=None):
        """mass matrix for products of edge functions w'*M(materialProp)*e"""
        return self.getMass(loc='e', materialProp=materialProp)

    def getFaceMass(self, materialProp=None):
        """mass matrix for products of face functions w'*M(materialProp)*f"""
        return self.getMass(loc='f', materialProp=materialProp)

    def getFaceMassDeriv(self):
        Av = self.aveF2CC
        return Av.T * sdiag(self.vol)
