from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from scipy.special import erf
from SimPEG import Utils

def Qfun(R, L, f, alpha=None):
    if alpha is None:
        omega = np.pi*2*f
        tau = L/R
        alpha = omega*tau
    Q = (alpha**2+1j*alpha) / (1+alpha**2)
    return alpha, Q

def Mijfun(x,y,z,incl,decl,x1,y1,z1,incl1,decl1, area=1.,area0=1.):
    """
        Compute mutual inductance between two loops

        This

        Parameters
        ----------
        x : array
            x location of the Tx loop
        y : array
            y location of the Tx loop
        z : array
            z location of the Tx loop
        incl:
            XXX
        decl:
            XXX
        x1 : array
            XXX
        y1 : array
            XXX
        z1 : array
            XXX
        incl1:
            XXX
        decl1:
            XXX
    """

    # Pretty sure below assumes dipole
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    x1 = np.array(x1, dtype=float)
    y1 = np.array(y1, dtype=float)
    z1 = np.array(z1, dtype=float)
    incl = np.array(incl, dtype=float)
    decl = np.array(decl, dtype=float)
    incl1 = np.array(incl1, dtype=float)
    decl1 = np.array(decl1, dtype=float)


    di=np.pi*incl/180.0
    dd=np.pi*decl/180.0

    cx=np.cos(di)*np.cos(dd)
    cy=np.cos(di)*np.sin(dd)
    cz=np.sin(di)


    ai=np.pi*incl1/180.0
    ad=np.pi*decl1/180.0

    ax=np.cos(ai)*np.cos(ad)
    ay=np.cos(ai)*np.sin(ad)
    az=np.sin(ai)

    # begin the calculation
    a=x-x1
    b=y-y1
    h=z-z1

    rt=np.sqrt(a**2.+b**2.+h**2.)**5.

    txy=3.*a*b/rt
    txz=3.*a*h/rt
    tyz=3.*b*h/rt

    txx=(2.*a**2.-b**2.-h**2.)/rt
    tyy=(2.*b**2.-a**2.-h**2.)/rt
    tzz=-(txx+tyy)

    scale = mu_0*np.pi*area*area0/4
    # scale = 1.

    bx= (txx*cx+txy*cy+txz*cz)
    by= (txy*cx+tyy*cy+tyz*cz)
    bz= (txz*cx+tyz*cy+tzz*cz)

    return scale*(bx*ax+by*ay+bz*az)

def Cfun(L,R,xc,yc,zc,incl,decl,S,ht,f,xyz):
    """
        Compute coupling coefficients

        .. math::
            - \frac{M_{12} M_{23}}{M_{13}L_2}

        Parameters
        ----------

    """
    L = np.array(L, dtype=float)
    R = np.array(R, dtype=float)
    xc = np.array(xc, dtype=float)
    yc = np.array(yc, dtype=float)
    zc = np.array(zc, dtype=float)
    incl = np.array(incl, dtype=float)
    decl = np.array(decl, dtype=float)
    S = np.array(S, dtype=float)
    f = np.array(f, dtype=float)

    # This is a bug, hence needs to be fixed later
    x = xyz[:,1]
    y = xyz[:,0]
    z = xyz[:,2]

    # simulate anomalies
    yt=y-S/2.
    yr=y+S/2.

    dm=-S/2.
    dp= S/2.

    # Computes mutual inducances
    # Mijfun(x,y,z,incl,decl,x1,y1,z1,incl1,decl1)
    M13=Mijfun(0.,dm,0.,90.,0., 0., dp, 0., 90.,0.)
    M12=Mijfun(x,yt,z,90.,0.,xc,yc,zc,incl,decl,area=1.,area0=3.)
    M23=Mijfun(xc,yc,zc,incl,decl,x,yr,z,90.,0.,area=3.,area0=1.)

    C = -M12*M23/(M13*L)
    return C, M12, M23, M13*np.ones_like(C)

if __name__ == '__main__':
    out = Mijfun(0., 0., 0., 0., 0., 10., 0, 0., 0., 0.)
    anal = mu_0*np.pi / (2*10**3)
    err = abs(out-anal)
    print err
    showIt = False
    import matplotlib.pyplot as plt
    f = np.logspace(-3, 3, 61)
    alpha, Q = Qfun(1., 0.1, f)
    if showIt:
        plt.semilogx(alpha, Q.real)
        plt.semilogx(alpha, Q.imag)
        plt.show()

    L = 1.
    R = 2000.
    xc = 0.
    yc = 0.
    zc = 2.
    incl = 0.
    decl = 90.
    S = 4.
    ht = 0.
    f = 10000.
    xmin = -10.
    xmax = 10.
    dx = 0.25

    xp = np.linspace(xmin, xmax, 101)
    yp = xp.copy()
    zp = np.r_[-ht]
    [Y, X] = np.meshgrid(yp, xp)
    xyz = np.c_[X.flatten(), Y.flatten(), np.ones_like(X.flatten())*ht]
    C, M12, M23, M13 = Cfun(L,R,xc,yc,zc,incl,decl,S,ht,f,xyz)
    [Xp, Yp] = np.meshgrid(xp, yp)
    if showIt:
        plt.contourf(X, Y, C.reshape(X.shape), 100)
        plt.show()

    # xyz = np.c_[xp, np.zeros_like(yp), np.zeros_like(yp)]
    # C, M12, M23, M13 = Cfun(L,R,xc,yc,zc,incl,decl,S,ht,f,xyz)
    # plt.plot(xp, C, 'k')
    # plt.plot(xp, M12, 'b')
    # plt.plot(xp, M23, 'g')
    # plt.plot(xp, M13, 'r')
    # plt.show()
