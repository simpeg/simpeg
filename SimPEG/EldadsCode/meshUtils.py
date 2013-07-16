from sputils import *
from utils import *
from numpy import *


def getCellCenterFromNodal(X, Y, Z):
    """Cell Centers from Nodal locations"""
    XC = 1.0/8.0 * (X[:-1, :-1, :-1] + X[1:, :-1, :-1] + X[:-1, 1:, :-1] + X[1:, 1:, :-1] +
                    X[:-1, :-1, 1:] + X[1:, :-1, 1:] + X[:-1, 1:, 1:] + X[1:, 1:, 1:])

    YC = 1.0/8.0 * (Y[:-1, :-1, :-1] + Y[1:, :-1, :-1] + Y[:-1, 1:, :-1] + Y[1:, 1:, :-1] +
                    Y[:-1, :-1, 1:] + Y[1:, :-1, 1:] + Y[:-1, 1:, 1:] + Y[1:, 1:, 1:])

    ZC = 1.0/8.0 * (Z[:-1, :-1, :-1] + Z[1:, :-1, :-1] + Z[:-1, 1:, :-1] + Z[1:, 1:, :-1] +
                    Z[:-1, :-1, 1:] + Z[1:, :-1, 1:] + Z[:-1, 1:, 1:] + Z[1:, 1:, 1:])

    return (XC, YC, ZC)


def getEdgesFromNodal(X, Y, Z):
    """Edges from Nodal locations

         node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
              /                                    /
             /                                    / |
         edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
           /                                    /   |
          /                                    /    |
    node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
         |                                     |    |
         |                                     |   node(i+1,j+1,k+1)
         |                                     |    /
    edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1.k)
         |                                     |  /
         |                                     | /
         |                                     |/
    node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)
    """

    XE1 = (X[1:, :, :]+X[:-1, :, :])/2.0
    YE1 = (Y[1:, :, :]+Y[:-1, :, :])/2.0
    ZE1 = (Z[1:, :, :]+Z[:-1, :, :])/2.0

    XE2 = (X[:, 1:, :]+X[:, :-1, :])/2.0
    YE2 = (Y[:, 1:, :]+Y[:, :-1, :])/2.0
    ZE2 = (Z[:, 1:, :]+Z[:, :-1, :])/2.0

    XE3 = (X[:, :, 1:]+X[:, :, :-1])/2.0
    YE3 = (Y[:, :, 1:]+Y[:, :, :-1])/2.0
    ZE3 = (Z[:, :, 1:]+Z[:, :, :-1])/2.0

    return (XE1, YE1, ZE1, XE2, YE2, ZE2, XE3, YE3, ZE3)


def getFacesFromNodal(X, Y, Z):
    """Get faces from nodal --"""

    XF1 = 1.0/4.0*(X[:, :-1, :-1]+X[:, 1:, :-1]+X[:, :-1, 1:]+X[:, 1:, 1:])
    YF1 = 1.0/4.0*(Y[:, :-1, :-1]+Y[:, 1:, :-1]+Y[:, :-1, 1:]+Y[:, 1:, 1:])
    ZF1 = 1.0/4.0*(Z[:, :-1, :-1]+Z[:, 1:, :-1]+Z[:, :-1, 1:]+Z[:, 1:, 1:])

    XF2 = 1.0/4.0*(X[:-1, :, :-1]+X[1:, :, :-1]+X[:-1, :, 1:]+X[1:, :, 1:])
    YF2 = 1.0/4.0*(Y[:-1, :, :-1]+Y[1:, :, :-1]+Y[:-1, :, 1:]+Y[1:, :, 1:])
    ZF2 = 1.0/4.0*(Z[:-1, :, :-1]+Z[1:, :, :-1]+Z[:-1, :, 1:]+Z[1:, :, 1:])

    XF3 = 1.0/4.0*(X[:-1, :-1, :]+X[1:, :-1, :]+X[:-1, 1:, :]+X[1:, 1:, :])
    YF3 = 1.0/4.0*(Y[:-1, :-1, :]+Y[1:, :-1, :]+Y[:-1, 1:, :]+Y[1:, 1:, :])
    ZF3 = 1.0/4.0*(Z[:-1, :-1, :]+Z[1:, :-1, :]+Z[:-1, 1:, :]+Z[1:, 1:, :])

    return (XF1, YF1, ZF1, XF2, YF2, ZF2, XF3, YF3, ZF3)


def projectEdgeVectorField(EV1, EV2, EV3, X, Y, Z):
    """Project Edge vector field"""

    t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z, nrm1, nrm2, nrm3 = getEdgeTangent(X, Y, Z)

    E1 = EV1[:, 0]*mkvc(t1x) + EV1[:, 1]*mkvc(t1y) + EV1[:, 2]*mkvc(t1z)
    E2 = EV2[:, 0]*mkvc(t2x) + EV2[:, 1]*mkvc(t2y) + EV2[:, 2]*mkvc(t2z)
    E3 = EV3[:, 0]*mkvc(t3x) + EV3[:, 1]*mkvc(t3y) + EV3[:, 2]*mkvc(t3z)

    return hstack((hstack((mkvc(E1), mkvc(E2))), mkvc(E3)))


def projectFaceVectorField(FV1, FV2, FV3, X, Y, Z):
    """Prolect Face vector field"""

    n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, ar1, ar2, ar3 = getFaceNormals(X, Y, Z)

    F1 = FV1[:, 0]*mkvc(n1x) + FV1[:, 1]*mkvc(n1y) + FV1[:, 2]*mkvc(n1z)
    F2 = FV2[:, 0]*mkvc(n2x) + FV2[:, 1]*mkvc(n2y) + FV2[:, 2]*mkvc(n2z)
    F3 = FV3[:, 0]*mkvc(n3x) + FV3[:, 1]*mkvc(n3y) + FV3[:, 2]*mkvc(n3z)

    return hstack((hstack((mkvc(F1), mkvc(F2))), mkvc(F3)))
