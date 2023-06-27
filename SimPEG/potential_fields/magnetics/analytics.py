from scipy.constants import mu_0
from SimPEG import utils

# from SimPEG import Mesh
import numpy as np


def MagSphereAnaFun(x, y, z, R, x0, y0, z0, mu1, mu2, H0, flag="total"):
    r"""
    test
    Analytic function for Magnetics problem. The set up here is
    magnetic sphere in whole-space assuming that the inducing field is oriented
    in the x-direction.

    * (x0, y0, z0)
    * (x0, y0, z0 ): is the center location of sphere
    * r: is the radius of the sphere

    .. math::

        \mathbf{H}_0 = H_0\hat{x}

    """

    if ~np.size(x) == np.size(y) == np.size(z):
        print("Specify same size of x, y, z")
        return
    x = utils.mkvc(x)
    y = utils.mkvc(y)
    z = utils.mkvc(z)

    ind = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) < R
    r = utils.mkvc(np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2))
    Bx = np.zeros(x.size)
    By = np.zeros(x.size)
    Bz = np.zeros(x.size)

    # Inside of the sphere
    rf2 = 3 * mu1 / (mu2 + 2 * mu1)
    if flag == "total" and any(ind):
        Bx[ind] = mu2 * H0 * (rf2)
    elif flag == "secondary":
        Bx[ind] = mu2 * H0 * (rf2) - mu1 * H0

    By[ind] = 0.0
    Bz[ind] = 0.0
    # Outside of the sphere
    rf1 = (mu2 - mu1) / (mu2 + 2 * mu1)
    if flag == "total":
        Bx[~ind] = mu1 * (
            H0
            + H0
            / r[~ind] ** 5
            * (R**3)
            * rf1
            * (2 * (x[~ind] - x0) ** 2 - (y[~ind] - y0) ** 2 - (z[~ind] - z0) ** 2)
        )
    elif flag == "secondary":
        Bx[~ind] = mu1 * (
            H0
            / r[~ind] ** 5
            * (R**3)
            * rf1
            * (2 * (x[~ind] - x0) ** 2 - (y[~ind] - y0) ** 2 - (z[~ind] - z0) ** 2)
        )

    By[~ind] = mu1 * (
        H0 / r[~ind] ** 5 * (R**3) * rf1 * (3 * (x[~ind] - x0) * (y[~ind] - y0))
    )
    Bz[~ind] = mu1 * (
        H0 / r[~ind] ** 5 * (R**3) * rf1 * (3 * (x[~ind] - x0) * (z[~ind] - z0))
    )
    return (
        np.reshape(Bx, x.shape, order="F"),
        np.reshape(By, x.shape, order="F"),
        np.reshape(Bz, x.shape, order="F"),
    )


def CongruousMagBC(mesh, Bo, chi):
    r"""
    Computing boundary condition using Congrous sphere method.

    This is designed for secondary field formulation.

    >> Input

    * mesh:   Mesh class
    * Bo:     np.array([Box, Boy, Boz]): Primary magnetic flux
    * chi:    susceptibility at cell volume

    .. math::

        \vec{B}(r) =
            \frac{\mu_0}{4\pi}
            \frac{
                m
            }{
                \| \vec{r} - \vec{r}_0 \|^3
            }
            [3\hat{m}\cdot\hat{r}-\hat{m}]

    """

    ind = chi > 0.0
    V = mesh.cell_volumes[ind].sum()

    gamma = 1 / V * (chi * mesh.cell_volumes).sum()  # like a mass!

    Bot = np.sqrt(sum(Bo**2))
    mx = Bo[0] / Bot
    my = Bo[1] / Bot
    mz = Bo[2] / Bot

    mom = 1 / mu_0 * Bot * gamma * V / (1 + gamma / 3)
    xc = sum(chi[ind] * mesh.gridCC[:, 0][ind]) / sum(chi[ind])
    yc = sum(chi[ind] * mesh.gridCC[:, 1][ind]) / sum(chi[ind])
    zc = sum(chi[ind] * mesh.gridCC[:, 2][ind]) / sum(chi[ind])

    indxd, indxu, indyd, indyu, indzd, indzu = mesh.face_boundary_indices

    const = mu_0 / (4 * np.pi) * mom

    def rfun(x):
        return np.sqrt((x[:, 0] - xc) ** 2 + (x[:, 1] - yc) ** 2 + (x[:, 2] - zc) ** 2)

    mdotrx = (
        mx
        * (mesh.gridFx[(indxd | indxu), 0] - xc)
        / rfun(mesh.gridFx[(indxd | indxu), :])
        + my
        * (mesh.gridFx[(indxd | indxu), 1] - yc)
        / rfun(mesh.gridFx[(indxd | indxu), :])
        + mz
        * (mesh.gridFx[(indxd | indxu), 2] - zc)
        / rfun(mesh.gridFx[(indxd | indxu), :])
    )

    Bbcx = (
        const
        / (rfun(mesh.gridFx[(indxd | indxu), :]) ** 3)
        * (
            3
            * mdotrx
            * (mesh.gridFx[(indxd | indxu), 0] - xc)
            / rfun(mesh.gridFx[(indxd | indxu), :])
            - mx
        )
    )

    mdotry = (
        mx
        * (mesh.gridFy[(indyd | indyu), 0] - xc)
        / rfun(mesh.gridFy[(indyd | indyu), :])
        + my
        * (mesh.gridFy[(indyd | indyu), 1] - yc)
        / rfun(mesh.gridFy[(indyd | indyu), :])
        + mz
        * (mesh.gridFy[(indyd | indyu), 2] - zc)
        / rfun(mesh.gridFy[(indyd | indyu), :])
    )

    Bbcy = (
        const
        / (rfun(mesh.gridFy[(indyd | indyu), :]) ** 3)
        * (
            3
            * mdotry
            * (mesh.gridFy[(indyd | indyu), 1] - yc)
            / rfun(mesh.gridFy[(indyd | indyu), :])
            - my
        )
    )

    mdotrz = (
        mx
        * (mesh.gridFz[(indzd | indzu), 0] - xc)
        / rfun(mesh.gridFz[(indzd | indzu), :])
        + my
        * (mesh.gridFz[(indzd | indzu), 1] - yc)
        / rfun(mesh.gridFz[(indzd | indzu), :])
        + mz
        * (mesh.gridFz[(indzd | indzu), 2] - zc)
        / rfun(mesh.gridFz[(indzd | indzu), :])
    )

    Bbcz = (
        const
        / (rfun(mesh.gridFz[(indzd | indzu), :]) ** 3)
        * (
            3
            * mdotrz
            * (mesh.gridFz[(indzd | indzu), 2] - zc)
            / rfun(mesh.gridFz[(indzd | indzu), :])
            - mz
        )
    )

    return np.r_[Bbcx, Bbcy, Bbcz], (1 / gamma - 1 / (3 + gamma)) * 1 / V


def MagSphereAnaFunA(x, y, z, R, xc, yc, zc, chi, Bo, flag):
    r"""
    Computing boundary condition using Congrous sphere method.

    This is designed for secondary field formulation.

    >> Input
    mesh:   Mesh class
    Bo:     np.array([Box, Boy, Boz]): Primary magnetic flux
    Chi:    susceptibility at cell volume

    .. math::

        \vec{B}(r) =
            \frac{\mu_0}{4\pi}
            \frac{
                m
            }{
                \| \vec{r}-\vec{r}_0 \|^3
            }
            [3\hat{m}\cdot\hat{r}-\hat{m}]

    """
    if ~np.size(x) == np.size(y) == np.size(z):
        print("Specify same size of x, y, z")
        return
    x = utils.mkvc(x)
    y = utils.mkvc(y)
    z = utils.mkvc(z)

    Bot = np.sqrt(sum(Bo**2))
    mx = Bo[0] / Bot
    my = Bo[1] / Bot
    mz = Bo[2] / Bot

    ind = np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2) < R

    Bx = np.zeros(x.size)
    By = np.zeros(x.size)
    Bz = np.zeros(x.size)

    # Inside of the sphere
    rf2 = 3 / (chi + 3) * (1 + chi)
    if flag == "total":
        Bx[ind] = Bo[0] * (rf2)
        By[ind] = Bo[1] * (rf2)
        Bz[ind] = Bo[2] * (rf2)
    elif flag == "secondary":
        Bx[ind] = Bo[0] * (rf2) - Bo[0]
        By[ind] = Bo[1] * (rf2) - Bo[1]
        Bz[ind] = Bo[2] * (rf2) - Bo[2]

    r = utils.mkvc(np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2))
    V = 4 * np.pi * R**3 / 3
    mom = Bot / mu_0 * chi / (1 + chi / 3) * V
    const = mu_0 / (4 * np.pi) * mom
    mdotr = (
        mx * (x[~ind] - xc) / r[~ind]
        + my * (y[~ind] - yc) / r[~ind]
        + mz * (z[~ind] - zc) / r[~ind]
    )
    Bx[~ind] = const / (r[~ind] ** 3) * (3 * mdotr * (x[~ind] - xc) / r[~ind] - mx)
    By[~ind] = const / (r[~ind] ** 3) * (3 * mdotr * (y[~ind] - yc) / r[~ind] - my)
    Bz[~ind] = const / (r[~ind] ** 3) * (3 * mdotr * (z[~ind] - zc) / r[~ind] - mz)

    return Bx, By, Bz


def IDTtoxyz(Inc, Dec, Btot):
    """Convert from Inclination, Declination,
    Total intensity of earth field to x, y, z
    """
    Bx = Btot * np.cos(Inc / 180.0 * np.pi) * np.sin(Dec / 180.0 * np.pi)
    By = Btot * np.cos(Inc / 180.0 * np.pi) * np.cos(Dec / 180.0 * np.pi)
    Bz = Btot * np.sin(Inc / 180.0 * np.pi)

    return np.r_[Bx, By, Bz]


def MagSphereFreeSpace(x, y, z, R, xc, yc, zc, chi, Bo):
    """Computing the induced response of magnetic sphere in free-space.

    >> Input
    x, y, z:   Observation locations
    R:     radius of the sphere
    xc, yc, zc: Location of the sphere
    chi: Susceptibility of sphere
    Bo: Inducing field components [bx, by, bz]*|H0|
    """
    if ~np.size(x) == np.size(y) == np.size(z):
        print("Specify same size of x, y, z")
        return

    x = utils.mkvc(x)
    y = utils.mkvc(y)
    z = utils.mkvc(z)
    """
    nobs = len(x)

    Bot = np.sqrt(sum(Bo**2))

    mx = np.ones([nobs]) * Bo[0] * R**3 / 3. * chi
    my = np.ones([nobs]) * Bo[1] * R**3 / 3. * chi
    mz = np.ones([nobs]) * Bo[2] * R**3 / 3. * chi

    M = np.c_[mx, my, mz]

    rx = (x - xc)
    ry = (y - yc)
    rz = (z - zc)

    rvec = np.c_[rx, ry, rz]
    r = np.sqrt((rx)**2+(ry)**2+(rz)**2)

    B = -utils.sdiag(1./r**3)*M + \
        utils.sdiag((3 * np.sum(M*rvec, axis=1))/r**5)*rvec

    Bx = B[:, 0]
    By = B[:, 1]
    Bz = B[:, 2]
    """

    rx = x - xc
    ry = y - yc
    rz = z - zc
    rx2 = rx * rx
    ry2 = ry * ry
    rz2 = rz * rz

    b_hat = Bo / np.linalg.norm(Bo)

    unit_conv = 1 / (4 * np.pi)

    r = np.sqrt(rx2 + ry2 + rz2)
    bot = r * r * r * r * r

    M = np.empty_like(x)  # create a vector of "Ms" if the point is outide
    M[r >= R] = R**3 * 4.0 / 3.0 * np.pi * chi  # outside points
    M[r < R] = r[r < R] ** 3 * 4.0 / 3.0 * np.pi * chi  # inside points

    g = unit_conv * (1.0 / bot) * M

    gxx = g * (2 * rx2 - ry2 - rz2)
    gyy = g * (2 * ry2 - rx2 - rz2)
    gzz = -gxx - gyy
    gxy = g * (3 * rx * ry)
    gxz = g * (3 * rx * rz)
    gyz = g * (3 * ry * rz)

    Bx = gxx * Bo[0] + gxy * Bo[1] + gxz * Bo[2]
    By = gxy * Bo[0] + gyy * Bo[1] + gyz * Bo[2]
    Bz = gxz * Bo[0] + gyz * Bo[1] + gzz * Bo[2]

    tmi = Bx * b_hat[0] + By * b_hat[1] + Bz * b_hat[2]

    return Bx, By, Bz, tmi
