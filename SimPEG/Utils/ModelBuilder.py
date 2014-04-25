import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from matutils import mkvc


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

    for ii in range(len(p0)):
        p0[ii], p1[ii] = np.min([p0[ii], p1[ii]]), np.max([p0[ii], p1[ii]])

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

def defineBlock(ccMesh,p0,p1,vals=[0,1]):
    """
        Build a block with the conductivity specified by condVal.  Returns an array.
        vals[0]  conductivity of the block
        vals[1]  conductivity of the ground
    """
    sigma = np.zeros(ccMesh.shape[0]) + vals[1]
    ind   = getIndecesBlock(p0,p1,ccMesh)

    sigma[ind] = vals[0]

    return mkvc(sigma)

def defineElipse(ccMesh, center=[0,0,0], anisotropy=[1,1,1], slope=10., theta=0.):
    G = ccMesh.copy()
    dim = ccMesh.shape[1]
    for i in range(dim):
        G[:, i] = G[:,i] - center[i]

    theta = -theta*np.pi/180
    M = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    M = M[:dim,:dim]
    G = M.dot(G.T).T

    for i in range(dim):
        G[:, i] = G[:,i]/anisotropy[i]*2.

    D = np.sqrt(np.sum(G**2,axis=1))
    return -np.arctan((D-1)*slope)*(2./np.pi)/2.+0.5

def defineTwoLayers(ccMesh,depth,vals=[0,1]):
    """
    Define a two layered model.  Depth of the first layer must be specified.
    CondVals vector with the conductivity values of the layers.  Eg:

    Convention to number the layers::

        <----------------------------|------------------------------------>
        0                          depth                                 zf
             1st layer                       2nd layer
    """
    sigma = np.zeros(ccMesh.shape[0]) + vals[1]

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

    sigma[ind] = vals[0];

    return mkvc(sigma)

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

    return mkvc(sigma)



def randomModel(shape, seed=None, anisotropy=None, its=100, bounds=[0,1]):
    """
        Create a random model by convolving a kernal with a
        uniformly distributed model.

        :param int,tuple shape: shape of the model.
        :param int seed: pick which model to produce, prints the seed if you don't choose.
        :param numpy.ndarray,list anisotropy: this is the (3 x n) blurring kernal that is used.
        :param int its: number of smoothing iterations
        :param list bounds: bounds on the model, len(list) == 2
        :rtype: numpy.ndarray
        :return: M, the model


        .. plot::

            import matplotlib.pyplot as plt
            import SimPEG.Utils.ModelBuilder as MB
            plt.colorbar(plt.imshow(MB.randomModel((50,50),bounds=[-4,0])))
            plt.title('A very cool, yet completely random model.')
            plt.show()


    """

    if seed is None:
        seed = np.random.randint(1e3)
        print 'Using a seed of: ', seed

    if type(shape) in [int, long, float]:
        shape = (shape,) # make it a tuple for consistency

    np.random.seed(seed)
    mr = np.random.rand(*shape)
    if anisotropy is None:
        if len(shape) is 1:
            smth = np.array([1,10.,1],dtype=float)
        elif len(shape) is 2:
            smth = np.array([[1,7,1],[2,10,2],[1,7,1]],dtype=float)
        elif len(shape) is 3:
            kernal = np.array([1,4,1], dtype=float).reshape((1,3))
            smth = np.array(sp.kron(sp.kron(kernal,kernal.T).todense()[:],kernal).todense()).reshape((3,3,3))
    else:
        assert len(anisotropy.shape) is len(shape), 'Anisotropy must be the same shape.'
        smth = np.array(anisotropy,dtype=float)

    smth = smth/smth.sum() # normalize
    mi = mr
    for i in range(its):
        mi = ndi.convolve(mi, smth)

    # scale the model to live between the bounds.
    mi = (mi - mi.min())/(mi.max()-mi.min()) # scaled between 0 and 1
    mi = mi*(bounds[1]-bounds[0])+bounds[0]


    return mi



if __name__ == '__main__':

    from SimPEG.Mesh import TensorMesh
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
    vals = np.array([100,1e-6])

    sigma = defineBlockConductivity(ccMesh,p0,p1,vals)

    # Plot sigma model
    print sigma.shape
    M.plotImage(sigma)
    print 'Done with block! :)'
    plt.show()

    # -----------------------------------------
    print('Testing the two layered model')
    vals = np.array([100,1e-5]);
    depth    = 1.0;

    sigma = defineTwoLayeredConductivity(ccMesh,depth,vals)

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
