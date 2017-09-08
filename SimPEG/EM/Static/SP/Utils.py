import pandas as pd
import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import Delaunay
from SimPEG import Utils, Mesh

# def in_hull(p, hull):
#     """
#     Test if points in `p` are in `hull`

#     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
#     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
#     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
#     will be computed
#     """
#     from scipy.spatial import Delaunay
#     if not isinstance(hull,Delaunay):
#         hull = Delaunay(hull)
#     return hull.find_simplex(p)>=0

def PolygonInd(mesh, pts):
    # if mesh.dim == 2:
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.gridCC)>=0
    # else:
    return inds

def readSeepageModel(fname, mesh=None, xsurf=None, ysurf=None):

    fluiddata = pd.read_csv(fname)
    header = fluiddata.keys()
    xyz = np.c_[fluiddata['X (m)'].values,fluiddata['Y (m)'].values]
    h = fluiddata['Total Head (m)'].values
    Ux = fluiddata['X-Velocity Magnitude (m/sec)'].values
    Uy = fluiddata['Y-Velocity Magnitude (m/sec)'].values
    Gradx = fluiddata['X-Gradient'].values
    Grady = fluiddata['Y-Gradient'].values

    Pressure = fluiddata['Pressure Head (m)'].values
    Sw = fluiddata[fluiddata.keys()[17]].values
    Kx = fluiddata["X-Conductivity (m/sec)"]
    Ky = fluiddata["Y-Conductivity (m/sec)"]

    if mesh is None:
        # TODO: this is a specific set up ...
        xmin, ymin = xyz[:,0].min(), xyz[:,1].min()
        cs = 0.5
        ncx = 152*2
        ncy = 36*2
        npad = 5
        hx = [(cs,npad, -1.3),(cs,ncx),(cs,npad, 1.3)]
        hy = [(cs,npad, -1.3),(cs,ncy)]
        mesh = Mesh.TensorMesh([hx, hy], x0 = [xmin, ymin])
        mesh._x0 = np.r_[xmin-mesh.hx[:10].sum(), xmin-mesh.hy[:10].sum()]
        # ...
        xsurf = np.r_[-1e10, 55, 90, 94, 109, 112, 126.5, +1e10]
        ysurf = np.r_[27.5, 27.5, 43.2, 43.2, 35, 35, 27.5, 27.5]
        yup = np.ones_like(ysurf)*45
        actind = Utils.surface2ind_topo(mesh, np.c_[xsurf, ysurf])
        waterheight = 40.
        waterind = (np.logical_and(~actind, mesh.gridCC[:,1]<40.)) & (mesh.gridCC[:,0]<90.)


    F_hlin = LinearNDInterpolator(xyz, h)
    hccIn = F_hlin(mesh.gridCC[actind,:])
    F_hnear = NearestNDInterpolator(mesh.gridCC[actind,:], hccIn)
    hcc = F_hnear(mesh.gridCC)

    fluiddata = {"xyz": xyz, "h":h, "Sw":Sw, "Kx":Kx, "Ky":Ky, "P":Pressure, "Ux":Ux, "Uy":Uy,\
                 "Gradx":Gradx, "Grady":Grady, \
                 "hcc": hcc, "mesh":mesh, "actind":actind, "waterind": waterind, \
                 "xsurf":xsurf, "ysurf":ysurf, "yup":yup}

    return fluiddata


def gettopoCC(mesh, airind):
# def gettopoCC(mesh, airind):

    """
        Get topography from active indices of mesh.
    """

    if mesh.dim == 3:

        mesh2D = Mesh.TensorMesh([mesh.hx, mesh.hy], mesh.x0[:2])
        zc = mesh.gridCC[:,2]
        AIRIND = airind.reshape((mesh.vnC[0]*mesh.vnC[1],mesh.vnC[2]), order='F')
        ZC = zc.reshape((mesh.vnC[0]*mesh.vnC[1], mesh.vnC[2]), order='F')
        topo = np.zeros(ZC.shape[0])
        topoCC = np.zeros(ZC.shape[0])
        for i in range(ZC.shape[0]):
            ind  = np.argmax(ZC[i,:][~AIRIND[i,:]])
            topo[i] = ZC[i,:][~AIRIND[i,:]].max() + mesh.hz[~AIRIND[i,:]][ind]*0.5
            topoCC[i] = ZC[i,:][~AIRIND[i,:]].max()

        return mesh2D, topoCC

    elif mesh.dim == 2:

        mesh1D = Mesh.TensorMesh([mesh.hx], [mesh.x0[0]])
        yc = mesh.gridCC[:,1]
        AIRIND = airind.reshape((mesh.vnC[0],mesh.vnC[1]), order='F')
        YC = yc.reshape((mesh.vnC[0], mesh.vnC[1]), order='F')
        topo = np.zeros(YC.shape[0])
        topoCC = np.zeros(YC.shape[0])
        for i in range(YC.shape[0]):
            ind  = np.argmax(YC[i,:][~AIRIND[i,:]])
            topo[i] = YC[i,:][~AIRIND[i,:]].max() + mesh.hy[~AIRIND[i,:]][ind]*0.5
            topoCC[i] = YC[i,:][~AIRIND[i,:]].max()

        return mesh1D, topoCC

def drapeTopotoLoc(mesh, topo, pts):
    """
        Drape
    """
    if mesh.dim ==2:
        if pts.ndim > 1:
            raise Exception("pts should be 1d array")
    elif mesh.dim ==3:
        if pts.shape[1] == 3:
            raise Exception("shape of pts should be (x,3)")
    else:
        raise NotImplementedError()
    airind = Utils.surface2ind_topo(mesh, topo)
    meshtemp, topoCC = gettopoCC(mesh, ~airind)
    inds = Utils.closestPoints(meshtemp, pts)

    return np.c_[pts, topoCC[inds]]
