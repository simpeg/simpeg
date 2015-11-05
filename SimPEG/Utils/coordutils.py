import numpy as np

def crossProd(v0,v1):
    """
        Cross product of 2 vectors

        :param numpy.array v0: vector of length 3
        :param numpy.array v1: vector of length 3
        :rtype: numpy.array
        :return: cross product of v0,v1
    """ 
    # ensure both n0, n1 are vectors of length 1
    assert len(v0) == 3, "Length of v0 should be 3"
    assert len(v1) == 3, "Length of v1 should be 3"

    v2 = np.zeros(3,dtype=float)

    v2[0] = v0[1]*v1[2] - v1[1]*v0[2]
    v2[1] = v1[0]*v0[2] - v0[0]*v1[2]
    v2[2] = v0[0]*v1[1] - v1[0]*v0[1]

    return v2

def rotationMatrixFromNormals(v0,v1,tol=1e-20):
    """
        Performs the minimum number of rotations to define a rotation from the direction indicated by the vector n0 to the direction indicated by n1.
        The axis of rotation is n0 x n1
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        :param numpy.array v0: vector of length 3
        :param numpy.array v1: vector of length 3
        :param tol = 1e-20: tolerance. If the norm of the cross product between the two vectors is below this, no rotation is performed
        :rtype: numpy.array, 3x3
        :return: rotation matrix which rotates the frame so that n0 is aligned with n1

    """

    # ensure both n0, n1 are vectors of length 1
    assert len(v0) == 3, "Length of n0 should be 3"
    assert len(v1) == 3, "Length of n1 should be 3"

    # ensure both are true normals
    n0 = v0*1./np.linalg.norm(v0)
    n1 = v1*1./np.linalg.norm(v1)

    n0dotn1 = n0.dot(n1) 

    # define the rotation axis, which is the cross product of the two vectors
    rotAx = crossProd(n0,n1)

    if np.linalg.norm(rotAx) < tol:
        return np.eye(3,dtype=float)

    rotAx *= 1./np.linalg.norm(rotAx)

    cosT = n0dotn1/(np.linalg.norm(n0)*np.linalg.norm(n1))
    sinT = np.sqrt(1.-n0dotn1**2)

    ux = np.array([[0., -rotAx[2], rotAx[1]], [rotAx[2], 0., -rotAx[0]], [-rotAx[1], rotAx[0], 0.]],dtype=float)

    return np.eye(3,dtype=float) + sinT*ux + (1.-cosT)*(ux.dot(ux))


def rotatePointsFromNormals(XYZ,n0,n1,x0=np.r_[0.,0.,0.]):
    """
        rotates a grid so that the vector n0 is aligned with the vector n1

        :param numpy.array n0: vector of length 3, should have norm 1
        :param numpy.array n1: vector of length 3, should have norm 1
        :param numpy.array x0: vector of length 3, point about which we perform the rotation 
        :rtype: numpy.array, 3x3
        :return: rotation matrix which rotates the frame so that n0 is aligned with n1
    """

    R = rotationMatrixFromNormals(n0, n1)

    assert XYZ.shape[1] == 3, "Grid XYZ should be 3 wide"
    assert len(x0) == 3, "x0 should have length 3"

    return (XYZ - x0).dot(R.T) + x0 