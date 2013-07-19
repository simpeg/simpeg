import numpy as np
from scipy import sparse as sp
from sputils import sdiag, speye, kron3, spzeros


def ddx(n):
    """Define 1D derivatives"""
    return sp.spdiags((np.ones((n+1, 1))*[-1, 1]).T, [0, 1], n, n+1, format="csr")


def av(n):
    """Define 1D averaging operator"""
    return sp.spdiags((0.5*np.ones((n+1, 1))*[1, 1]).T, [0, 1], n, n+1, format="csr")


class DiffOperators(object):
    """
        Class creates the differential operators that you need!
    """
    def __init__(self):
        raise Exception('You should use a Mesh class.')

    def faceDiv():
        doc = "Construct the 3D Divergence operator on Faces."

        def fget(self):
            if(self._faceDiv is None):
                # The number of cell centers in each direction
                n = self.n
                # Compute faceDivergence operator on faces
                dd = [ddx(k) for k in n]
                if(self.dim == 1):
                    D = dd[0]
                elif(self.dim == 2):
                    D1 = sp.kron(speye(n[1]), dd[0])
                    D2 = sp.kron(dd[1], speye(n[0]))
                    D = sp.hstack((D1, D2), format="csr")
                elif(self.dim == 3):
                    D1 = kron3(speye(n[2]), speye(n[1]), dd[0])
                    D2 = kron3(speye(n[2]), dd[1], speye(n[0]))
                    D3 = kron3(dd[2], speye(n[1]), speye(n[0]))
                    D = sp.hstack((D1, D2, D3), format="csr")
                # Compute areas of cell faces
                S = self.area
                # Compute cell volumes
                V = self.vol
                self._faceDiv = sdiag(1/V)*D*sdiag(S)

            return self._faceDiv
        return locals()
    _faceDiv = None
    faceDiv = property(**faceDiv())

    def nodalGrad():
        doc = "Construct the 3D nodal gradient operator."

        def fget(self):
            if(self._nodalGrad is None):
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

                G = sp.vstack((D1, D2, D3), format="csr")
                self._nodalGrad = sdiag(1/L)*G
            return self._nodalGrad
        return locals()
    _nodalGrad = None
    nodalGrad = property(**nodalGrad())

    def edgeCurl():
        doc = "Construct the 3D curl operator."

        def fget(self):
            if(self._edgeCurl is None):
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

                C = sp.vstack((sp.hstack((O1, -D32, D23)),
                               sp.hstack((D31, O2, -D13)),
                               sp.hstack((-D21, D12, O3))), format="csr")

                self._edgeCurl = sdiag(1/S)*(C*sdiag(L))
            return self._edgeCurl
        return locals()
    _edgeCurl = None
    edgeCurl = property(**edgeCurl())

    def faceAve():
        doc = "Construct the 3D averaging operator on cell faces to cell centers."

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
        doc = "Construct the 3D averaging operator on cell edges."

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



def getEdgeMassMatrix(h,sigma):
    # mass matix for products of edge functions w'*M(sigma)*e
         
    Av    = getEdgeToCellAverge(h)
    v     = getVol(h)
    sigma = mkvc(sigma)
    
    return sdiag(Av.T*(v*sigma))
    
def getFaceMassMatrix(h,sigma):
    # mass matix for products of edge functions w'*M(sigma)*e
         
    Av    = getFaceToCellAverge(h)
    v     = getVol(h)
    sigma = mkvc(sigma)
    
    return sdiag(Av.T*(v*sigma))    