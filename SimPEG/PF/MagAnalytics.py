from __future__ import print_function
from scipy.constants import mu_0
from SimPEG import *
from SimPEG import Utils
from SimPEG import Mesh
import numpy as np
from SimPEG.Utils import kron3, speye, sdiag
import matplotlib.pyplot as plt


def spheremodel(mesh, x0, y0, z0, r):
    """
        Generate model indicies for sphere
        - (x0, y0, z0 ): is the center location of sphere
        - r: is the radius of the sphere
        - it returns logical indicies of cell-center model
    """
    ind = np.sqrt( (mesh.gridCC[:, 0]-x0)**2+(mesh.gridCC[:, 1]-y0)**2+(mesh.gridCC[:, 2]-z0)**2 ) < r
    return ind


def MagSphereAnaFun(x, y, z, R, x0, y0, z0, mu1, mu2, H0, flag='total'):
    """
        test
        Analytic function for Magnetics problem. The set up here is
        magnetic sphere in whole-space assuming that the inducing field is oriented in the x-direction.

        * (x0, y0, z0)
        * (x0, y0, z0 ): is the center location of sphere
        * r: is the radius of the sphere

    .. math::

        \mathbf{H}_0 = H_0\hat{x}


    """

    if (~np.size(x)==np.size(y)==np.size(z)):
        print("Specify same size of x, y, z")
        return
    dim = x.shape
    x = Utils.mkvc(x)
    y = Utils.mkvc(y)
    z = Utils.mkvc(z)

    ind = np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2) < R
    r = Utils.mkvc(np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2))
    Bx = np.zeros(x.size)
    By = np.zeros(x.size)
    Bz = np.zeros(x.size)

    # Inside of the sphere
    rf2 = 3*mu1/(mu2+2*mu1)
    if flag is 'total' and any(ind):
        Bx[ind] = mu2*H0*(rf2)
    elif (flag == 'secondary'):
        Bx[ind] = mu2*H0*(rf2)-mu1*H0

    By[ind] = 0.
    Bz[ind] = 0.
    # Outside of the sphere
    rf1 = (mu2-mu1)/(mu2+2*mu1)
    if (flag == 'total'):
        Bx[~ind] = mu1*(H0+H0/r[~ind]**5*(R**3)*rf1*(2*(x[~ind]-x0)**2-(y[~ind]-y0)**2-(z[~ind]-z0)**2))
    elif (flag == 'secondary'):
        Bx[~ind] = mu1*(H0/r[~ind]**5*(R**3)*rf1*(2*(x[~ind]-x0)**2-(y[~ind]-y0)**2-(z[~ind]-z0)**2))

    By[~ind] = mu1*(H0/r[~ind]**5*(R**3)*rf1*(3*(x[~ind]-x0)*(y[~ind]-y0)))
    Bz[~ind] = mu1*(H0/r[~ind]**5*(R**3)*rf1*(3*(x[~ind]-x0)*(z[~ind]-z0)))
    return np.reshape(Bx, x.shape, order='F'), np.reshape(By, x.shape, order='F'), np.reshape(Bz, x.shape, order='F')


def CongruousMagBC(mesh, Bo, chi):
    """Computing boundary condition using Congrous sphere method.
    This is designed for secondary field formulation.

    >> Input

    * mesh:   Mesh class
    * Bo:     np.array([Box, Boy, Boz]): Primary magnetic flux
    * chi:    susceptibility at cell volume

    .. math::

        \\vec{B}(r) = \\frac{\mu_0}{4\pi} \\frac{m}{ \| \\vec{r} - \\vec{r}_0\|^3}[3\hat{m}\cdot\hat{r}-\hat{m}]

    """

    ind = chi > 0.
    V = mesh.vol[ind].sum()

    gamma = 1/V*(chi*mesh.vol).sum()  # like a mass!

    Bot = np.sqrt(sum(Bo**2))
    mx = Bo[0]/Bot
    my = Bo[1]/Bot
    mz = Bo[2]/Bot

    mom = 1/mu_0*Bot*gamma*V/(1+gamma/3)
    xc = sum(chi[ind]*mesh.gridCC[:, 0][ind])/sum(chi[ind])
    yc = sum(chi[ind]*mesh.gridCC[:, 1][ind])/sum(chi[ind])
    zc = sum(chi[ind]*mesh.gridCC[:, 2][ind])/sum(chi[ind])

    indxd, indxu, indyd, indyu, indzd, indzu =  mesh.faceBoundaryInd

    const = mu_0/(4*np.pi)*mom
    rfun = lambda x: np.sqrt((x[:, 0]-xc)**2 + (x[:, 1]-yc)**2 + (x[:, 2]-zc)**2)

    mdotrx = (mx*(mesh.gridFx[(indxd|indxu), 0]-xc)/rfun(mesh.gridFx[(indxd|indxu), :]) +
              my*(mesh.gridFx[(indxd|indxu), 1]-yc)/rfun(mesh.gridFx[(indxd|indxu), :]) +
              mz*(mesh.gridFx[(indxd|indxu), 2]-zc)/rfun(mesh.gridFx[(indxd|indxu), :]))

    Bbcx = const/(rfun(mesh.gridFx[(indxd|indxu), :])**3)*(3*mdotrx*(mesh.gridFx[(indxd|indxu), 0]-xc)/rfun(mesh.gridFx[(indxd|indxu), :])-mx)

    mdotry = (mx*(mesh.gridFy[(indyd|indyu), 0]-xc)/rfun(mesh.gridFy[(indyd|indyu), :]) +
              my*(mesh.gridFy[(indyd|indyu), 1]-yc)/rfun(mesh.gridFy[(indyd|indyu), :]) +
              mz*(mesh.gridFy[(indyd|indyu), 2]-zc)/rfun(mesh.gridFy[(indyd|indyu), :]))

    Bbcy = const/(rfun(mesh.gridFy[(indyd|indyu), :])**3)*(3*mdotry*(mesh.gridFy[(indyd|indyu), 1]-yc)/rfun(mesh.gridFy[(indyd|indyu), :])-my)

    mdotrz = (mx*(mesh.gridFz[(indzd|indzu), 0]-xc)/rfun(mesh.gridFz[(indzd|indzu), :])  +
              my*(mesh.gridFz[(indzd|indzu), 1]-yc)/rfun(mesh.gridFz[(indzd|indzu), :]) +
              mz*(mesh.gridFz[(indzd|indzu), 2]-zc)/rfun(mesh.gridFz[(indzd|indzu), :]))

    Bbcz = const/(rfun(mesh.gridFz[(indzd|indzu), :])**3)*(3*mdotrz*(mesh.gridFz[(indzd|indzu), 2]-zc)/rfun(mesh.gridFz[(indzd|indzu), :])-mz)

    return np.r_[Bbcx, Bbcy, Bbcz], (1/gamma-1/(3+gamma))*1/V


def MagSphereAnaFunA(x, y, z, R, xc, yc, zc, chi, Bo, flag):
    """Computing boundary condition using Congrous sphere method.
    This is designed for secondary field formulation.
    >> Input
    mesh:   Mesh class
    Bo:     np.array([Box, Boy, Boz]): Primary magnetic flux
    Chi:    susceptibility at cell volume

    .. math::

        \\vec{B}(r) = \\frac{\mu_0}{4\pi}\\frac{m}{\| \\vec{r}-\\vec{r}_0\|^3}[3\hat{m}\cdot\hat{r}-\hat{m}]

    """
    if (~np.size(x)==np.size(y)==np.size(z)):
        print("Specify same size of x, y, z")
        return
    dim = x.shape
    x = Utils.mkvc(x)
    y = Utils.mkvc(y)
    z = Utils.mkvc(z)

    Bot = np.sqrt(sum(Bo**2))
    mx = Bo[0]/Bot
    my = Bo[1]/Bot
    mz = Bo[2]/Bot

    ind = np.sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2 ) < R

    Bx = np.zeros(x.size)
    By = np.zeros(x.size)
    Bz = np.zeros(x.size)

    # Inside of the sphere
    rf2 = 3/(chi+3)*(1+chi)
    if (flag == 'total'):
        Bx[ind] = Bo[0]*(rf2)
        By[ind] = Bo[1]*(rf2)
        Bz[ind] = Bo[2]*(rf2)
    elif (flag == 'secondary'):
        Bx[ind] = Bo[0]*(rf2)-Bo[0]
        By[ind] = Bo[1]*(rf2)-Bo[1]
        Bz[ind] = Bo[2]*(rf2)-Bo[2]

    r = Utils.mkvc(np.sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2 ))
    V = 4*np.pi*R**3/3
    mom = Bot/mu_0*chi/(1+chi/3)*V
    const = mu_0/(4*np.pi)*mom
    mdotr = (mx*(x[~ind]-xc)/r[~ind] + my*(y[~ind]-yc)/r[~ind] + mz*(z[~ind]-zc)/r[~ind])
    Bx[~ind] = const/(r[~ind]**3)*(3*mdotr*(x[~ind]-xc)/r[~ind]-mx)
    By[~ind] = const/(r[~ind]**3)*(3*mdotr*(y[~ind]-yc)/r[~ind]-my)
    Bz[~ind] = const/(r[~ind]**3)*(3*mdotr*(z[~ind]-zc)/r[~ind]-mz)

    return Bx, By, Bz


def IDTtoxyz(Inc, Dec, Btot):
    """Convert from Inclination, Declination,
    Total intensity of earth field to x, y, z
    """
    Bx = Btot*np.cos(Inc/180.*np.pi)*np.sin(Dec/180.*np.pi)
    By = Btot*np.cos(Inc/180.*np.pi)*np.cos(Dec/180.*np.pi)
    Bz = -Btot*np.sin(Inc/180.*np.pi)

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
    if (~np.size(x) == np.size(y) == np.size(z)):
        print("Specify same size of x, y, z")
        return

    x = Utils.mkvc(x)
    y = Utils.mkvc(y)
    z = Utils.mkvc(z)

    nobs = len(x)

    Bot = np.sqrt(sum(Bo**2))

    mx = np.ones([nobs]) * Bo[0] * R**3 / 3. * chi
    my = np.ones([nobs]) * Bo[1] * R**3 / 3. * chi
    mz = np.ones([nobs]) * Bo[2] * R**3 / 3. * chi

    M = np.c_[mx, my, mz]

    rx = (x - xc)
    ry = (y - yc)
    rz = (zc - z)

    rvec = np.c_[rx, ry, rz]
    r = np.sqrt((rx)**2+(ry)**2+(rz)**2)

    B = -Utils.sdiag(1./r**3)*M + \
        Utils.sdiag((3 * np.sum(M*rvec, axis=1))/r**5)*rvec

    Bx = B[:, 0]
    By = B[:, 1]
    Bz = B[:, 2]

    return Bx, By, Bz
