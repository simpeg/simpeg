import numpy as np
from scipy import sparse
from sputils import ddx, sdiag, speye, kron3, spzeros, av


class DiffOperators(object):
    """
        Class creates the differential operators that you need!
    """
    def __init__(self):
        raise Exception('You should use a Mesh class.')

    def DIV():
        doc = "Construct the 3D divergence operator on Faces."

        def fget(self):
            if(self._DIV is None):
                # The number of cell centers in each direction
                n = [x.size for x in self.h]
                # Compute divergence operator on faces
                dd = [ddx(x) for x in n]
                if(self.dim == 1):
                    D = dd[0]
                elif(self.dim == 2):
                    D1 = sparse.kron(speye(n[1]), dd[0])
                    D2 = sparse.kron(dd[1], speye(n[0]))
                    D = sparse.hstack((D1, D2), format="csr")
                elif(self.dim == 3):
                    D1 = kron3(speye(n[2]), speye(n[1]), dd[0])
                    D2 = kron3(speye(n[2]), dd[1], speye(n[0]))
                    D3 = kron3(dd[2], speye(n[1]), speye(n[0]))
                    D = sparse.hstack((D1, D2, D3), format="csr")
                # Compute areas of cell faces
                S = self.area
                # Compute cell volumes
                V = self.vol
                self._DIV = sdiag(1/V)*D*sdiag(S)

            return self._DIV
        return locals()
    _DIV = None
    DIV = property(**DIV())

    def GRAD():
        doc = "Construct the 3D nodal gradient operator."

        def fget(self):
            if(self._GRAD is None):
                # The number of cell centers in each direction
                n1 = np.size(self.hx)
                n2 = np.size(self.hy)
                n3 = np.size(self.hz)

                # Compute lengths of cell edges
                L = self.edge

                # Compute divergence operator on faces
                d1 = ddx(n1)
                d2 = ddx(n2)
                d3 = ddx(n3)
                D1 = kron3(speye(n3+1), speye(n2+1), d1)
                D2 = kron3(speye(n3+1), d2, speye(n1+1))
                D3 = kron3(d3, speye(n2+1), speye(n1+1))

                G = sparse.vstack((D1, D2, D3), format="csr")
                self._GRAD = sdiag(1/L)*G
            return self._GRAD
        return locals()
    _GRAD = None
    GRAD = property(**GRAD())

    def CURL():
        doc = "Construct the 3D curl operator."

        def fget(self):
            if(self._CURL is None):
                # The number of cell centers in each direction
                n1 = np.size(self.hx)
                n2 = np.size(self.hy)
                n3 = np.size(self.hz)

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

                C = sparse.vstack((sparse.hstack((O1, -D32, D23)),
                                   sparse.hstack((D31, O2, -D13)),
                                   sparse.hstack((-D21, D12, O3))), format="csr")

                self._CURL = sdiag(1/S)*(C*sdiag(L))
            return self._CURL
        return locals()
    _CURL = None
    CURL = property(**CURL())

    def AVE_F():
        doc = "Construct the 3D averaging operator on cell faces."

        def fget(self):
            if(self._AVE_F is None):
                # The number of cell centers in each direction
                n1 = np.size(self.hx)
                n2 = np.size(self.hy)
                n3 = np.size(self.hz)

                av1 = av(n1)
                av2 = av(n2)
                av3 = av(n3)

                self._AVE_F = sparse.hstack(kron3(speye(n3), speye(n2), av1),
                                            kron3(speye(n3), av2, speye(n3)),
                                            kron3(av3, speye(n2), speye(n3)), format="csr")
            return self._AVE_F
        return locals()
    _AVE_F = None
    AVE_F = property(**AVE_F())

    def AVE_E():
        doc = "Construct the 3D averaging operator on cell edges."

        def fget(self):
            if(self._AVE_E is None):
                # The number of cell centers in each direction
                n1 = np.size(self.hx)
                n2 = np.size(self.hy)
                n3 = np.size(self.hz)

                av1 = av(n1)
                av2 = av(n2)
                av3 = av(n3)

                self._AVE_E = sparse.hstack(kron3(av3, av2, speye(n1)),
                                            kron3(av3, speye(n2), av1),
                                            kron3(speye(n3), av2, av1), format="csr")
            return self._AVE_E
        return locals()
    _AVE_E = None
    AVE_E = property(**AVE_E())
