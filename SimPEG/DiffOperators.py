import numpy as np
from scipy import sparse as sp
from SimPEG.utils import mkvc, sdiag, speye, kron3, spzeros


def ddx(n):
    """Define 1D derivatives, inner, this means we go from n+1 to n+1"""
    return sp.spdiags((np.ones((n+1, 1))*[-1, 1]).T, [0, 1], n, n+1, format="csr")


def checkBC(bc):
    """ Checks if boundary condition 'bc' is valid. """
    if(type(bc) is str):
        bc = [bc, bc]
    assert type(bc) is list, 'bc must be a list'
    assert len(bc) == 2, 'bc must have two elements'

    for bc_i in bc:
        assert type(bc_i) is str, "each bc must be a string"
        assert bc_i in ['dirichlet', 'neumann'], "each bc must be either, 'dirichlet' or 'neumann'"
    return bc


def ddxCellGrad(n, bc):
    """Create 1D derivative operator from cell-centres to nodes this means we go from n to n+1"""
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


def av(n):
    """Define 1D averaging operator from cell-centres to nodes."""
    return sp.spdiags((0.5*np.ones((n+1, 1))*[1, 1]).T, [0, 1], n, n+1, format="csr")


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

        self._cellGrad = None  # ensure we create a new gradient next time we call it
        self._cellGradBC = BC
        return BC
    _cellGradBC = 'neumann'

    def cellGrad():
        doc = "The cell centered Gradient, takes you to cell faces."

        def fget(self):
            if(self._cellGrad is None):
                BC = self.setCellGradBC(self._cellGradBC)
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
                V = self.vol
                self._cellGrad = sdiag(S)*G*sdiag(1/V)
            return self._cellGrad
        return locals()
    _cellGrad = None
    cellGrad = property(**cellGrad())

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

    def faceAve():
        doc = "Construct the averaging operator on cell faces to cell centers."

        def fget(self):
            if(self._faceAve is None):
                n = self.n
                if(self.dim == 1):
                    self._faceAve = av(n[0])
                elif(self.dim == 2):
                    self._faceAve = sp.hstack((sp.kron(speye(n[1]), av(n[0])),
                                               sp.kron(av(n[1]), speye(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._faceAve = sp.hstack((kron3(speye(n[2]), speye(n[1]), av(n[0])),
                                               kron3(speye(n[2]), av(n[1]), speye(n[0])),
                                               kron3(av(n[2]), speye(n[1]), speye(n[0]))), format="csr")
            return self._faceAve
        return locals()
    _faceAve = None
    faceAve = property(**faceAve())

    def edgeAve():
        doc = "Construct the averaging operator on cell edges to cell centers."

        def fget(self):
            if(self._edgeAve is None):
                # The number of cell centers in each direction
                n = self.n
                if(self.dim == 1):
                    raise Exception('Edge Averaging does not make sense in 1D: Use Identity?')
                elif(self.dim == 2):
                    self._edgeAve = sp.hstack((sp.kron(av(n[1]), speye(n[0])),
                                               sp.kron(speye(n[1]), av(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._edgeAve = sp.hstack((kron3(av(n[2]), av(n[1]), speye(n[0])),
                                               kron3(av(n[2]), speye(n[1]), av(n[0])),
                                               kron3(speye(n[2]), av(n[1]), av(n[0]))), format="csr")
            return self._edgeAve
        return locals()
    _edgeAve = None
    edgeAve = property(**edgeAve())

    def nodalAve():
        doc = "Construct the averaging operator on cell nodes to cell centers."

        def fget(self):
            if(self._nodalAve is None):
                # The number of cell centers in each direction
                n = self.n
                if(self.dim == 1):
                    self._nodalAve = av(n[0])
                elif(self.dim == 2):
                    self._nodalAve = sp.hstack((sp.kron(av(n[1]), av(n[0])),
                                                sp.kron(av(n[1]), av(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._nodalAve = sp.hstack((kron3(av(n[2]), av(n[1]), av(n[0])),
                                                kron3(av(n[2]), av(n[1]), av(n[0])),
                                                kron3(av(n[2]), av(n[1]), av(n[0]))), format="csr")
            return self._nodalAve
        return locals()
    _nodalAve = None
    nodalAve = property(**nodalAve())

    def nodalVectorAve():
        doc = "Construct the averaging operator on cell nodes to cell centers, keeping each dimension separate."

        def fget(self):
            if(self._nodalVectorAve is None):
                # The number of cell centers in each direction
                n = self.n
                if(self.dim == 1):
                    self._nodalVectorAve = av(n[0])
                elif(self.dim == 2):
                    self._nodalVectorAve = sp.block_diag((sp.kron(av(n[1]), av(n[0])),
                                                          sp.kron(av(n[1]), av(n[0]))), format="csr")
                elif(self.dim == 3):
                    self._nodalVectorAve = sp.block_diag((kron3(av(n[2]), av(n[1]), av(n[0])),
                                                          kron3(av(n[2]), av(n[1]), av(n[0])),
                                                          kron3(av(n[2]), av(n[1]), av(n[0]))), format="csr")
            return self._nodalVectorAve
        return locals()
    _nodalVectorAve = None
    nodalVectorAve = property(**nodalVectorAve())

    def getMass(self, loc='e', materialProp=None, inv=False):
        """ Produces mass matricies.

        Kwargs:
            loc (str): 'e' - Average to edges
                       'f'              faces
            materialProp: property to be averaged (see below)
            inv (bool): True returns matrix inverse

        Returns:
            scipy.sparse.csr.csr_matrix

        materialProp can be:
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
            Av = self.edgeAve
        elif loc=='f':
            Av = self.faceAve
        else:
            raise ValueError('Invalid loc')

        diag = Av.T * (self.vol * mkvc(materialProp))

        if inv:
            diag = 1/diag

        return sdiag(diag)

    def getEdgeMass(self, materialProp=None):
        """mass matrix for products of edge functions w'*M(materialProp)*e"""
        if(materialProp is None):
            materialProp = np.ones(self.nC)
        Av = self.edgeAve
        return sdiag(Av.T * (self.vol * mkvc(materialProp)))

    def getFaceMass(self, materialProp=None):
        """mass matrix for products of face functions w'*M(materialProp)*f"""
        if(materialProp is None):
            materialProp = np.ones(self.nC)
        Av = self.faceAve
        return sdiag(Av.T * (self.vol * mkvc(materialProp)))

    def getFaceMassDeriv(self):
        Av = self.faceAve
        return Av.T * sdiag(self.vol)
