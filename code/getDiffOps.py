from sputils import *
from utils import *
from numpy import *
from getEdgeTangent import *
from getCellVolume import getCellVolume
from getFaceNormals import getFaceNormals


def getDivMatrix(X, Y, Z):
    """Face DIV"""

    n = array(shape(X))-1
    n1 = n[0]
    n2 = n[1]
    n3 = n[2]

    n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, area1, area2, area3 = getFaceNormals(X, Y, Z)

    area = hstack((hstack((mkvc(area1), mkvc(area2))), mkvc(area3)))
    S = sdiag(area)
    V = getCellVolume(X, Y, Z)

    d1 = ddx(n1)
    d2 = ddx(n2)
    d3 = ddx(n3)
    D1 = kron3(speye(n3), speye(n2), d1)
    D2 = kron3(speye(n3), d2, speye(n1))
    D3 = kron3(d3, speye(n2), speye(n1))

    # divergence on faces
    D = appendRight3(D1, D2, D3)

    return sdiag(1/V)*D*S


def getCurlMatrix(X, Y, Z):
    """Edge CURL """

    n = array(shape(X))-1
    n1 = n[0]; n2 = n[1]; n3 = n[2]

    d1 = ddx(n1); d2 = ddx(n2); d3 = ddx(n3)
    # derivatives on x-edge variables
    D32 = kron3(d3, speye(n2), speye(n1+1))
    D23 = kron3(speye(n3), d2, speye(n1+1))
    D31 = kron3(d3, speye(n2+1), speye(n1))
    D13 = kron3(speye(n3), speye(n2+1), d1)
    D21 = kron3(speye(n3+1), d2, speye(n1))
    D12 = kron3(speye(n3+1), speye(n2), d1)

    O1 = spzeros(shape(D32)[0], shape(D31)[1])
    O2 = spzeros(shape(D31)[0], shape(D32)[1])
    O3 = spzeros(shape(D21)[0], shape(D13)[1])

    CURL = appendBottom3(
        appendRight3(O1,    -D32,  D23),
        appendRight3(D31,    O2,  -D13),
        appendRight3(-D21,  D12,   O3))

    # scale for non-uniform mesh
    e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, norme1, norme2, norme3 = getEdgeTangent(X, Y, Z)
    n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, area1, area2, area3 = getFaceNormals(X, Y, Z)

    area = hstack((hstack((mkvc(area1), mkvc(area2))), mkvc(area3)))
    S = sdiag(1/area)
    lngth = hstack((hstack((mkvc(norme1), mkvc(norme2))), mkvc(norme3)))
    L = sdiag(lngth)

    return S*(CURL*L)


def getNodalGradient(X, Y, Z):
    """Nodal Gradients"""

    n = array(shape(X))-1
    n1 = n[0]; n2 = n[1]; n3 = n[2]

    D1 = kron3(speye(n3+1), speye(n2+1), ddx(n1))
    D2 = kron3(speye(n3+1), ddx(n2), speye(n1+1))
    D3 = kron3(ddx(n3), speye(n2+1), speye(n1+1))

    # topological gradient
    GRAD = appendBottom3(D1, D2, D3)

    # scale for non-uniform mesh
    e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z, norme1, norme2, norme3 = getEdgeTangent(X, Y, Z)
    lngth = hstack((hstack((mkvc(norme1), mkvc(norme2))), mkvc(norme3)))
    L = sdiag(1/lngth)

    return L*GRAD


if __name__ == '__main__':

    X, Y, Z = ndgrid(linspace(0, 2, 3), linspace(0, 2, 3), linspace(0, 2, 3))
    Z[2, 2, 2] = 2.5
    Z[0, 0, 0] = -0.5
    X[2, 2, 2] = 2.5
    X[0, 0, 0] = -0.5
    sig = ones([2, 2, 2])
    C = getCurlMatrix(X, Y, Z)

    G = getNodalGradient(X, Y, Z)

    tt = C*G
    print(tt)
