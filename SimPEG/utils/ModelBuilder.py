import numpy as np


def getIndecesBlock(p0,p1,ccMesh):
    """
        Creates a vector containing the block indexes in the cell centerd mesh.
        Returns a tuple

        The block is defined by the points

        p0, describe the position of the left  upper  front corner, and

        p1, describe the position of the right bottom back  corner.

        ccMesh represents the cell-centered mesh

        The points p0 and p1 must live in the the same dimensional space as the mesh.
    """

    # Validation: p0 and p1 live in the same dimensional space
    assert len(p0) == len(p1), "Dimension mismatch. len(p0) != len(p1)"

    # Validation: mesh and points live in the same dimensional space
    dimMesh = np.size(ccMesh[0,:])
    assert len(p0) == dimMesh, "Dimension mismatch. len(p0) != dimMesh"

    if dimMesh == 1:
       # Define the reference points
        x1 = p0[0]
        x2 = p1[0]

        indX = (x1 <= ccMesh[:,0]) & (ccMesh[:,0] <= x2)
        ind  = np.where(indX)

    elif dimMesh == 2:
       # Define the reference points
        x1 = p0[0]
        y1 = p0[1]

        x2 = p1[0]
        y2 = p1[1]

        indX = (x1 <= ccMesh[:,0]) & (ccMesh[:,0] <= x2)
        indY = (y1 <= ccMesh[:,1]) & (ccMesh[:,1] <= y2)

        ind  = np.where(indX & indY)

    elif dimMesh == 3:
        # Define the points
        x1 = p0[0]
        y1 = p0[1]
        z1 = p0[2]

        x2 = p1[0]
        y2 = p1[1]
        z2 = p1[2]

        indX = (x1 <= ccMesh[:,0]) & (ccMesh[:,0] <= x2)
        indY = (y1 <= ccMesh[:,1]) & (ccMesh[:,1] <= y2)
        indZ = (z1 <= ccMesh[:,2]) & (ccMesh[:,2] <= z2)

        ind  = np.where(indX & indY & indZ)

    # Return a tuple
    return ind

def defineBlockConductivity(p0,p1,ccMesh,condVals):
    """
        Build a block with the conductivity specified by condVal.  Returns an array.
        condVals[0]  conductivity of the block
        condVals[1]  conductivity of the ground
    """
    sigma = np.zeros(ccMesh.shape[0]) + condVals[1]
    ind   = getIndecesBlock(p0,p1,ccMesh)

    sigma[ind] = condVals[0]

    return sigma

def defineTwoLayeredConductivity(depth,ccMesh,condVals):
    """
    Define a two layered model.  Depth of the first layer must be specified.
    CondVals vector with the conductivity values of the layers.  Eg:

    Convention to number the layers::

        <----------------------------|------------------------------------>
        0                          depth                                 zf
             1st layer                       2nd layer
    """
    sigma = np.zeros(ccMesh.shape[0]) + condVals[1]

    dim = np.size(ccMesh[0,:])

    p0 = np.zeros(dim)
    p1 = np.zeros(dim)

    # Identify 1st cell centered reference point
    p0[0] = ccMesh[0,0]
    if dim>1: p0[1] = ccMesh[0,1]
    if dim>2: p0[2] = ccMesh[0,2]

    # Identify the last cell-centered reference point
    p1[0] = ccMesh[-1,0]
    if dim>1: p1[1] = ccMesh[-1,1]
    if dim>2: p1[2] = ccMesh[-1,2]

    # The depth is always defined on the last one.
    p1[len(p1)-1] -= depth

    ind   = getIndecesBlock(p0,p1,ccMesh)

    sigma[ind] = condVals[0];

    return sigma

def scalarConductivity(ccMesh,pFunction):
    """
    Define the distribution conductivity in the mesh according to the
    analytical expression given in pFunction
    """
    dim = np.size(ccMesh[0,:])
    CC = [ccMesh[:,0]]
    if dim>1: CC.append(ccMesh[:,1])
    if dim>2: CC.append(ccMesh[:,2])


    sigma = pFunction(*CC)

    return sigma

if __name__ == '__main__':

    from SimPEG import TensorMesh
    from matplotlib import pyplot as plt

    # Define the mesh

    testDim = 2
    h1 = 0.3*np.ones(7)
    h1[0] = 0.5
    h1[-1] = 0.6
    h2 = .5 * np.ones(4)
    h3 = .4 * np.ones(6)
    x0 = np.zeros(3)

    if testDim == 1:
        h = [h1]
        x0 = x0[0]
    elif testDim == 2:
        h = [h1, h2]
        x0 = x0[0:2]
    else:
        h = [h1, h2, h3]

    M = TensorMesh(h, x0)

    ccMesh = M.gridCC

    # ------------------- Test conductivities! --------------------------
    print('Testing 1 block conductivity')

    p0 = np.array([0.5,0.5,0.5])[:testDim]
    p1 = np.array([1.0,1.0,1.0])[:testDim]
    condVals = np.array([100,1e-6])

    sigma = defineBlockConductivity(p0,p1,ccMesh,condVals)

    # Plot sigma model
    print sigma.shape
    M.plotImage(sigma)
    print 'Done with block! :)'
    plt.show()

    # -----------------------------------------
    print('Testing the two layered model')
    condVals = np.array([100,1e-5]);
    depth    = 1.0;

    sigma = defineTwoLayeredConductivity(depth,ccMesh,condVals)

    M.plotImage(sigma)
    print sigma
    print 'layer model!'
    plt.show()

    # -----------------------------------------
    print('Testing scalar conductivity')

    if testDim == 1:
        pFunction = lambda x: np.exp(x)
    elif testDim == 2:
        pFunction = lambda x,y: np.exp(x+y)
    elif testDim == 3:
        pFunction = lambda x,y,z: np.exp(x+y+z)

    sigma = scalarConductivity(ccMesh,pFunction)

    # Plot sigma model
    M.plotImage(sigma)
    print sigma
    print 'Scalar conductivity defined!'
    plt.show()

    # -----------------------------------------
