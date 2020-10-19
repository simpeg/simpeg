import numpy as np
from scipy.spatial import Delaunay
from SimPEG import utils
from discretize import TensorMesh, TreeMesh


def polygon_index(mesh, pts):
    # if mesh.dim == 2:
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.grid_cell_centers) >= 0
    # else:
    return inds


def readSeepageModel(fname, mesh=None, x_surface=None, y_surface=None):

    fluiddata = pd.read_csv(fname)
    header = fluiddata.keys()
    xyz = np.c_[fluiddata["X (m)"].values, fluiddata["Y (m)"].values]
    h = fluiddata["Total Head (m)"].values
    Ux = fluiddata["X-Velocity Magnitude (m/sec)"].values
    Uy = fluiddata["Y-Velocity Magnitude (m/sec)"].values
    Gradx = fluiddata["X-Gradient"].values
    Grady = fluiddata["Y-Gradient"].values

    Pressure = fluiddata["Pressure Head (m)"].values
    Sw = fluiddata[fluiddata.keys()[17]].values
    Kx = fluiddata["X-Conductivity (m/sec)"]
    Ky = fluiddata["Y-Conductivity (m/sec)"]

    if mesh is None:
        # TODO: this is a specific set up ...
        xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
        cs = 0.5
        ncx = 152 * 2
        ncy = 36 * 2
        npad = 5
        hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, ncy)]
        mesh = TensorMesh([hx, hy], x0=[xmin, ymin])
        mesh._x0 = np.r_[xmin - mesh.hx[:10].sum(), xmin - mesh.hy[:10].sum()]
        # ...
        x_surface = np.r_[-1e10, 55, 90, 94, 109, 112, 126.5, +1e10]
        y_surface = np.r_[27.5, 27.5, 43.2, 43.2, 35, 35, 27.5, 27.5]
        yup = np.ones_like(y_surface) * 45
        actind = Utils.surface2ind_topo(mesh, np.c_[x_surface, y_surface])
        waterheight = 40.0
        waterind = (np.logical_and(~actind, mesh.grid_cell_centers[:, 1] < 40.0)) & (
            mesh.grid_cell_centers[:, 0] < 90.0
        )

    F_hlin = LinearNDInterpolator(xyz, h)
    hccIn = F_hlin(mesh.grid_cell_centers[actind, :])
    F_hnear = NearestNDInterpolator(mesh.grid_cell_centers[actind, :], hccIn)
    hcc = F_hnear(mesh.grid_cell_centers)
    fluiddata = {
        "xyz": xyz,
        "h": h,
        "Sw": Sw,
        "Kx": Kx,
        "Ky": Ky,
        "P": Pressure,
        "Ux": Ux,
        "Uy": Uy,
        "Gradx": Gradx,
        "Grady": Grady,
        "hcc": hcc,
        "mesh": mesh,
        "actind": actind,
        "waterind": waterind,
        "x_surface": x_surface,
        "y_surface": y_surface,
        "yup": yup,
    }

    return fluiddata


def writeVectorUBC(mesh, fileName, model):
    """
        Writes a vector model associated with a SimPEG TensorMesh
        to a UBC-GIF format model file.

        :param string fileName: File to write to
        :param numpy.ndarray model: The model
    """

    modelMatTR = np.zeros_like(model)

    for ii in range(3):
        # Reshape model to a matrix
        modelMat = mesh.r(model[:, ii], "CC", "CC", "M")
        # Transpose the axes
        modelMatT = modelMat.transpose((2, 0, 1))
        # Flip UBC order
        modelMatTR[:, ii] = Utils.mkvc(modelMatT[::-1, :, :])

        # Flip z to positive down for MeshTools3D
        if ii == 2:
            modelMatTR[:, ii] *= -1
    np.savetxt(fileName, modelMatTR)


def readVectorUBC(mesh, fileName):
    """Read UBC 3DVector model and generate 3D Vector mesh model

    Input:
    :param string fileName: path to the UBC GIF mesh file to read

    Output:
    :rtype: numpy.ndarray
    :return: model with TensorMesh ordered x3 nC
    """
    model = np.loadtxt(fileName)
    # Fist line is the size of the model
    # model = np.array(model.ravel()[0].split(), dtype=float)

    vx = np.reshape(model[:, 0], (mesh.nCz, mesh.nCx, mesh.nCy), order="F")
    vx = vx[::-1, :, :]
    vx = np.transpose(vx, (1, 2, 0))
    vx = Utils.mkvc(vx)

    vy = np.reshape(model[:, 1], (mesh.nCz, mesh.nCx, mesh.nCy), order="F")
    vy = vy[::-1, :, :]
    vy = np.transpose(vy, (1, 2, 0))
    vy = Utils.mkvc(vy)

    vz = np.reshape(model[:, 2], (mesh.nCz, mesh.nCx, mesh.nCy), order="F")
    vz = vz[::-1, :, :]
    vz = np.transpose(vz, (1, 2, 0))
    vz = Utils.mkvc(vz)

    # Flip z to positive up from MeshTools3D to SimPEG
    model = np.r_[vx, vy, -vz]
    return model
