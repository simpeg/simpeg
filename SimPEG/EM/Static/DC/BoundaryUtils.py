from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def getxBCyBC_CC(mesh, alpha, beta, gamma):

    """
    This is a subfunction generating mixed-boundary condition:

    .. math::

        \\nabla \cdot \\vec{j} = -\\nabla \cdot \\vec{j}_s = q

        \\rho \\vec{j} = -\\nabla \phi

        \\alpha \phi + \\beta \\frac{\partial \phi}{\partial r} = \gamma
        \quad \\vec{r} \subset \partial \Omega

        xBC = f_1(\\alpha, \\beta, \\gamma)

        yBC = f(\\alpha, \\beta, \\gamma)


    Computes xBC and yBC for cell-centered discretizations

    """

    if mesh.dim == 1:  # 1D
        if (len(alpha) != 2 or len(beta) != 2 or len(gamma) != 2):
            raise Exception("Lenght of list, alpha should be 2")
        fCCxm, fCCxp = mesh.cellBoundaryInd
        nBC = fCCxm.sum()+fCCxp.sum()
        h_xm, h_xp = mesh.gridCC[fCCxm], mesh.gridCC[fCCxp]

        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]

        # h_xm, h_xp = mesh.gridCC[fCCxm], mesh.gridCC[fCCxp]
        h_xm, h_xp = mesh.hx[0], mesh.hx[-1]

        a_xm = gamma_xm/(0.5*alpha_xm-beta_xm/h_xm)
        b_xm = (0.5*alpha_xm+beta_xm/h_xm)/(0.5*alpha_xm-beta_xm/h_xm)
        a_xp = gamma_xp/(0.5*alpha_xp-beta_xp/h_xp)
        b_xp = (0.5*alpha_xp+beta_xp/h_xp)/(0.5*alpha_xp-beta_xp/h_xp)

        xBC_xm = 0.5*a_xm
        xBC_xp = 0.5*a_xp/b_xp
        yBC_xm = 0.5*(1.-b_xm)
        yBC_xp = 0.5*(1.-1./b_xp)

        xBC = np.r_[xBC_xm, xBC_xp]
        yBC = np.r_[yBC_xm, yBC_xp]

    elif mesh.dim == 2:  # 2D
        if (len(alpha) != 4 or len(beta) != 4 or len(gamma) != 4):
            raise Exception("Lenght of list, alpha should be 4")

        fxm, fxp, fym, fyp = mesh.faceBoundaryInd
        nBC = fxm.sum()+fxp.sum()+fxm.sum()+fxp.sum()

        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]
        alpha_ym, beta_ym, gamma_ym = alpha[2], beta[2], gamma[2]
        alpha_yp, beta_yp, gamma_yp = alpha[3], beta[3], gamma[3]

        # h_xm, h_xp = mesh.gridCC[fCCxm,0], mesh.gridCC[fCCxp,0]
        # h_ym, h_yp = mesh.gridCC[fCCym,1], mesh.gridCC[fCCyp,1]

        h_xm = mesh.hx[0]*np.ones_like(alpha_xm)
        h_xp = mesh.hx[-1]*np.ones_like(alpha_xp)
        h_ym = mesh.hy[0]*np.ones_like(alpha_ym)
        h_yp = mesh.hy[-1]*np.ones_like(alpha_yp)

        a_xm = gamma_xm/(0.5*alpha_xm-beta_xm/h_xm)
        b_xm = (0.5*alpha_xm+beta_xm/h_xm)/(0.5*alpha_xm-beta_xm/h_xm)
        a_xp = gamma_xp/(0.5*alpha_xp-beta_xp/h_xp)
        b_xp = (0.5*alpha_xp+beta_xp/h_xp)/(0.5*alpha_xp-beta_xp/h_xp)

        a_ym = gamma_ym/(0.5*alpha_ym-beta_ym/h_ym)
        b_ym = (0.5*alpha_ym+beta_ym/h_ym)/(0.5*alpha_ym-beta_ym/h_ym)
        a_yp = gamma_yp/(0.5*alpha_yp-beta_yp/h_yp)
        b_yp = (0.5*alpha_yp+beta_yp/h_yp)/(0.5*alpha_yp-beta_yp/h_yp)

        xBC_xm = 0.5*a_xm
        xBC_xp = 0.5*a_xp/b_xp
        yBC_xm = 0.5*(1.-b_xm)
        yBC_xp = 0.5*(1.-1./b_xp)
        xBC_ym = 0.5*a_ym
        xBC_yp = 0.5*a_yp/b_yp
        yBC_ym = 0.5*(1.-b_ym)
        yBC_yp = 0.5*(1.-1./b_yp)

        sortindsfx = np.argsort(np.r_[np.arange(mesh.nFx)[fxm],
                                      np.arange(mesh.nFx)[fxp]])
        sortindsfy = np.argsort(np.r_[np.arange(mesh.nFy)[fym],
                                      np.arange(mesh.nFy)[fyp]])

        xBC_x = np.r_[xBC_xm, xBC_xp][sortindsfx]
        xBC_y = np.r_[xBC_ym, xBC_yp][sortindsfy]
        yBC_x = np.r_[yBC_xm, yBC_xp][sortindsfx]
        yBC_y = np.r_[yBC_ym, yBC_yp][sortindsfy]

        xBC = np.r_[xBC_x, xBC_y]
        yBC = np.r_[yBC_x, yBC_y]

    elif mesh.dim == 3:  # 3D
        if (len(alpha) != 6 or len(beta) != 6 or len(gamma) != 6):
            raise Exception("Lenght of list, alpha should be 6")
        # fCCxm,fCCxp,fCCym,fCCyp,fCCzm,fCCzp = mesh.cellBoundaryInd
        fxm, fxp, fym, fyp, fzm, fzp = mesh.faceBoundaryInd
        nBC = fxm.sum()+fxp.sum()+fxm.sum()+fxp.sum()

        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]
        alpha_ym, beta_ym, gamma_ym = alpha[2], beta[2], gamma[2]
        alpha_yp, beta_yp, gamma_yp = alpha[3], beta[3], gamma[3]
        alpha_zm, beta_zm, gamma_zm = alpha[4], beta[4], gamma[4]
        alpha_zp, beta_zp, gamma_zp = alpha[5], beta[5], gamma[5]

        # h_xm, h_xp = mesh.gridCC[fCCxm,0], mesh.gridCC[fCCxp,0]
        # h_ym, h_yp = mesh.gridCC[fCCym,1], mesh.gridCC[fCCyp,1]
        # h_zm, h_zp = mesh.gridCC[fCCzm,2], mesh.gridCC[fCCzp,2]

        h_xm = mesh.hx[0]*np.ones_like(alpha_xm)
        h_xp = mesh.hx[-1]*np.ones_like(alpha_xp)
        h_ym = mesh.hy[0]*np.ones_like(alpha_ym)
        h_yp = mesh.hy[-1]*np.ones_like(alpha_yp)
        h_zm = mesh.hz[0]*np.ones_like(alpha_zm)
        h_zp = mesh.hz[-1]*np.ones_like(alpha_zp)

        a_xm = gamma_xm/(0.5*alpha_xm-beta_xm/h_xm)
        b_xm = (0.5*alpha_xm+beta_xm/h_xm)/(0.5*alpha_xm-beta_xm/h_xm)
        a_xp = gamma_xp/(0.5*alpha_xp-beta_xp/h_xp)
        b_xp = (0.5*alpha_xp+beta_xp/h_xp)/(0.5*alpha_xp-beta_xp/h_xp)

        a_ym = gamma_ym/(0.5*alpha_ym-beta_ym/h_ym)
        b_ym = (0.5*alpha_ym+beta_ym/h_ym)/(0.5*alpha_ym-beta_ym/h_ym)
        a_yp = gamma_yp/(0.5*alpha_yp-beta_yp/h_yp)
        b_yp = (0.5*alpha_yp+beta_yp/h_yp)/(0.5*alpha_yp-beta_yp/h_yp)

        a_zm = gamma_zm/(0.5*alpha_zm-beta_zm/h_zm)
        b_zm = (0.5*alpha_zm+beta_zm/h_zm)/(0.5*alpha_zm-beta_zm/h_zm)
        a_zp = gamma_zp/(0.5*alpha_zp-beta_zp/h_zp)
        b_zp = (0.5*alpha_zp+beta_zp/h_zp)/(0.5*alpha_zp-beta_zp/h_zp)

        xBC_xm = 0.5*a_xm
        xBC_xp = 0.5*a_xp/b_xp
        yBC_xm = 0.5*(1.-b_xm)
        yBC_xp = 0.5*(1.-1./b_xp)
        xBC_ym = 0.5*a_ym
        xBC_yp = 0.5*a_yp/b_yp
        yBC_ym = 0.5*(1.-b_ym)
        yBC_yp = 0.5*(1.-1./b_yp)
        xBC_zm = 0.5*a_zm
        xBC_zp = 0.5*a_zp/b_zp
        yBC_zm = 0.5*(1.-b_zm)
        yBC_zp = 0.5*(1.-1./b_zp)

        sortindsfx = np.argsort(np.r_[np.arange(mesh.nFx)[fxm],
                                      np.arange(mesh.nFx)[fxp]])
        sortindsfy = np.argsort(np.r_[np.arange(mesh.nFy)[fym],
                                      np.arange(mesh.nFy)[fyp]])
        sortindsfz = np.argsort(np.r_[np.arange(mesh.nFz)[fzm],
                                      np.arange(mesh.nFz)[fzp]])

        xBC_x = np.r_[xBC_xm, xBC_xp][sortindsfx]
        xBC_y = np.r_[xBC_ym, xBC_yp][sortindsfy]
        xBC_z = np.r_[xBC_zm, xBC_zp][sortindsfz]

        yBC_x = np.r_[yBC_xm, yBC_xp][sortindsfx]
        yBC_y = np.r_[yBC_ym, yBC_yp][sortindsfy]
        yBC_z = np.r_[yBC_zm, yBC_zp][sortindsfz]

        xBC = np.r_[xBC_x, xBC_y, xBC_z]
        yBC = np.r_[yBC_x, yBC_y, yBC_z]

    return xBC, yBC
