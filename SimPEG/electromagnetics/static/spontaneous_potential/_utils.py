import numpy as np
import discretize
from discretize.utils import active_from_xyz
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator


def readSeepageModel(fname, mesh=None, xsurf=None, ysurf=None):
    fluiddata = pd.read_csv(fname)
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
        mesh = discretize.TensorMesh([hx, hy], x0=[xmin, ymin])
        mesh._x0 = np.r_[xmin - mesh.h[0][:10].sum(), xmin - mesh.h[1][:10].sum()]
        # ...
        xsurf = np.r_[-1e10, 55, 90, 94, 109, 112, 126.5, +1e10]
        ysurf = np.r_[27.5, 27.5, 43.2, 43.2, 35, 35, 27.5, 27.5]
        yup = np.ones_like(ysurf) * 45
        actind = active_from_xyz(mesh, np.c_[xsurf, ysurf])
        waterind = (np.logical_and(~actind, mesh.gridCC[:, 1] < 40.0)) & (
            mesh.gridCC[:, 0] < 90.0
        )

    F_hlin = LinearNDInterpolator(xyz, h)
    hccIn = F_hlin(mesh.gridCC[actind, :])
    F_hnear = NearestNDInterpolator(mesh.gridCC[actind, :], hccIn)
    hcc = F_hnear(mesh.gridCC)
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
        "xsurf": xsurf,
        "ysurf": ysurf,
        "yup": yup,
    }

    return fluiddata
