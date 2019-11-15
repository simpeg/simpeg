from __future__ import print_function
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from .matutils import mkvc
from scipy.spatial import Delaunay

import sys
if sys.version_info < (3,):
    num_types = [int, long, float]
else:
    num_types = [int, float]


def addBlock(gridCC, modelCC, p0, p1, blockProp):
    """
        Add a block to an exsisting cell centered model, modelCC

        :param numpy.array gridCC: mesh.gridCC is the cell centered grid
        :param numpy.array modelCC: cell centered model
        :param numpy.array p0: bottom, southwest corner of block
        :param numpy.array p1: top, northeast corner of block
        :blockProp float blockProp: property to assign to the model

        :return numpy.array, modelBlock: model with block
    """
    ind = getIndicesBlock(p0, p1, gridCC)
    modelBlock = modelCC.copy()
    modelBlock[ind] = blockProp
    return modelBlock


def getIndicesBlock(p0, p1, ccMesh):
    """
        Creates a vector containing the block indices in the cell centers mesh.
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


def defineBlock(ccMesh, p0, p1, vals=None):
    """
        Build a block with the conductivity specified by condVal.  Returns an array.
        vals[0]  conductivity of the block
        vals[1]  conductivity of the ground
    """
    if vals is None:
        vals = [0,1]
    sigma = np.zeros(ccMesh.shape[0]) + vals[1]
    ind   = getIndicesBlock(p0,p1,ccMesh)

    sigma[ind] = vals[0]

    return mkvc(sigma)


def defineElipse(ccMesh, center=None, anisotropy=None, slope=10., theta=0.):
    if center is None:
        center = [0,0,0]
    if anisotropy is None:
        anisotropy = [1,1,1]
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


def getIndicesSphere(center, radius, ccMesh):
    """
        Creates a vector containing the sphere indices in the cell centers mesh.
        Returns a tuple

        The sphere is defined by the points

        p0, describe the position of the center of the cell

        r, describe the radius of the sphere.

        ccMesh represents the cell-centered mesh

        The points p0 must live in the the same dimensional space as the mesh.

    """

    # Validation: mesh and point (p0) live in the same dimensional space
    dimMesh = np.size(ccMesh[0,:])
    assert len(center) == dimMesh, "Dimension mismatch. len(p0) != dimMesh"

    if dimMesh == 1:
       # Define the reference points

        ind  = np.abs(center[0] - ccMesh[:,0]) < radius

    elif dimMesh == 2:
       # Define the reference points

        ind = np.sqrt( ( center[0] - ccMesh[:,0] )**2 + ( center[1] - ccMesh[:,1] )**2 ) < radius

    elif dimMesh == 3:
        # Define the points
        ind = np.sqrt( ( center[0] - ccMesh[:,0] )**2 + ( center[1] - ccMesh[:,1] )**2 + ( center[2] - ccMesh[:,2] )**2 ) < radius

    # Return a tuple
    return ind


def defineTwoLayers(ccMesh, depth, vals=None):
    """
    Define a two layered model.  Depth of the first layer must be specified.
    CondVals vector with the conductivity values of the layers.  Eg:

    Convention to number the layers::

        <----------------------------|------------------------------------>
        0                          depth                                 zf
             1st layer                       2nd layer
    """
    if vals is None:
        vals = [0,1]
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

    ind   = getIndicesBlock(p0,p1,ccMesh)

    sigma[ind] = vals[0];

    return mkvc(sigma)


def scalarConductivity(ccMesh, pFunction):
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


def layeredModel(ccMesh, layerTops, layerValues):
    """
        Define a layered model from layerTops (z-positive up)

        :param numpy.array ccMesh: cell-centered mesh
        :param numpy.array layerTops: z-locations of the tops of each layer
        :param numpy.array layerValue: values of the property to assign for each layer (starting at the top)
        :rtype: numpy.array
        :return: M, layered model on the mesh
    """

    descending = np.linalg.norm(sorted(layerTops, reverse=True) - layerTops) < 1e-20

    # TODO: put an error check to make sure that there is an ordering... needs to work with inf elts
    # assert ascending or descending, "Layers must be listed in either ascending or descending order"

    # start from bottom up
    if not descending:
        zprop = np.hstack([mkvc(layerTops,2),mkvc(layerValues,2)])
        zprop.sort(axis=0)
        layerTops, layerValues  = zprop[::-1,0], zprop[::-1,1]

    # put in vector form
    layerTops, layerValues = mkvc(layerTops), mkvc(layerValues)

    # initialize with bottom layer
    dim = ccMesh.shape[1]
    if dim == 3:
        z = ccMesh[:,2]
    elif dim == 2:
        z = ccMesh[:,1]
    elif dim == 1:
        z = ccMesh[:,0]

    model = np.zeros(ccMesh.shape[0])

    for i, top in enumerate(layerTops):
        zind = z <= top
        model[zind] = layerValues[i]

    return model


def randomModel(shape, seed=None, anisotropy=None, its=100, bounds=None):
    """
        Create a random model by convolving a kernel with a
        uniformly distributed model.

        :param tuple shape: shape of the model.
        :param int seed: pick which model to produce, prints the seed if you don't choose.
        :param numpy.ndarray anisotropy: this is the (3 x n) blurring kernel that is used.
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
    if bounds is None:
        bounds = [0,1]

    if seed is None:
        seed = np.random.randint(1e3)
        print('Using a seed of: ', seed)

    if type(shape) in num_types:
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


def PolygonInd(mesh, pts):
    """
        Finde a volxel indices included in mpolygon (2D) or polyhedra (3D)
        uniformly distributed model.

        :param tuple shape: shape of the model.
        :param int seed: pick which model to produce, prints the seed if you don't choose.
        :param numpy.ndarray anisotropy: this is the (3 x n) blurring kernel that is used.
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
    if mesh.dim == 1:
        assert "Only works for a mesh greater than 1-dimension"
    elif mesh.dim == 2:
        assert ~(pts.shape[1] != 2), "Please input (*,2) array"
    elif mesh.dim == 3:
        assert ~(pts.shape[1] != 3), "Please input (*,3) array"
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.gridCC)>=0
    return inds
