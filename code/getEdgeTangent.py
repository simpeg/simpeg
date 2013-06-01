from numpy import *
from utils import diff

#function[t1x,t1y,t1z,t2x,t2y,t2z,t3x,t3y,t3z,normt1,normt2,normt3] = getEdgeTangent(X,Y,Z)
#%[t1x,t1y,t1z,t2x,t2y,t2z,t3x,t3y,t3z,normt1,normt2,normt3] = getEdgeTangent(X,Y,Z)
#%
#%      node(i,j,k+1) ------ edgt2(i,j,k+1) ----- node(i,j+1,k+1)
#%           /                                    /
#%          /                                    / |
#%      edgt3(i,j,k)     fact1(i,j,k)        edgt3(i,j+1,k)
#%        /                                    /   |
#%       /                                    /    |
#% node(i,j,k) ------ edgt2(i,j,k) ----- node(i,j+1,k)
#%      |                                     |    |
#%      |                                     |   node(i+1,j+1,k+1)
#%      |                                     |    /
#% edgt1(i,j,k)      fact3(i,j,k)        edgt1(i,j+1.k)
#%      |                                     |  /
#%      |                                     | /
#%      |                                     |/
#% node(i+1,j,k) ------ edgt2(i+1,j,k) ----- node(i+1,j+1,k)


def getEdgeTangent(X, Y, Z):

    t1x = diff(X, 1)
    t1y = diff(Y, 1)
    t1z = diff(Z, 1)

    normt1 = sqrt(t1x**2+t1y**2+t1z**2)
    t1x = t1x/normt1
    t1y = t1y/normt1
    t1z = t1z/normt1

    t2x = diff(X, 2)
    t2y = diff(Y, 2)
    t2z = diff(Z, 2)
    normt2 = sqrt(t2x**2 + t2y**2 + t2z**2)
    t2x = t2x/normt2
    t2y = t2y/normt2
    t2z = t2z/normt2

    t3x = diff(X, 3)
    t3y = diff(Y, 3)
    t3z = diff(Z, 3)
    normt3 = sqrt(t3x**2+t3y**2+t3z**2)
    t3x = t3x/normt3
    t3y = t3y/normt3
    t3z = t3z/normt3

    # print t3x

    return (t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z, normt1, normt2, normt3)


if __name__ == '__main__':

    X, Y, Z = mgrid[0:4, 0:5, 0:6]

    t = getEdgeTangent(X, Y, Z)
