import numpy as np
from SimPEG import (Maps, Utils, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)


def spectral_ip_mappings(
    mesh, indActive=None,
    inactive_eta=1e-4,
    inactive_tau=1e-4,
    inactive_c=1e-4,
    is_log_eta=True,
    is_log_tau=True,
    is_log_c=True
):
    """
    Generates Mappings for Spectral Induced Polarization Problem.
    Three parameters are required to be input:
    Chargeability (eta), Time constant (tau), and Frequency dependency (c).
    If there is no topography (indActive is None),
    model (m) can be either set to

    m = np.r_[log(eta), log(tau), log(c)] or m = np.r_[eta, tau, c]

    When indActive is not None, m is

    m = np.r_[log(eta[indAcitve]), log(tau[indAcitve]), log(c[indAcitve])] or
    m = np.r_[eta[indAcitve], tau[indAcitve], c[indAcitve]] or

    TODO: Illustrate input and output variables
    """

    if indActive is None:
        indActive = np.ones(mesh.nC, dtype=bool)

    actmap_eta = Maps.InjectActiveCells(
        mesh, indActive=indActive, valInactive=inactive_eta
    )
    actmap_tau = Maps.InjectActiveCells(
        mesh, indActive=indActive, valInactive=inactive_tau
    )
    actmap_c = Maps.InjectActiveCells(
        mesh, indActive=indActive, valInactive=inactive_c
    )

    wires = Maps.Wires(
        ('eta', indActive.sum()),
        ('tau', indActive.sum()),
        ('c', indActive.sum())
    )

    if is_log_eta:
        eta_map = actmap_eta*Maps.ExpMap(nP=actmap_eta.nP)*wires.eta
    else:
        eta_map = actmap_eta*wires.eta

    if is_log_tau:
        tau_map = actmap_tau*Maps.ExpMap(nP=actmap_tau.nP)*wires.tau
    else:
        tau_map = actmap_tau*wires.tau

    if is_log_c:
        c_map = actmap_c*Maps.ExpMap(nP=actmap_c.nP)*wires.c
    else:
        c_map = actmap_c*wires.c

    return eta_map, tau_map, c_map, wires


def run_inversion(
    m0, survey, actind, mesh, wires,
    std, eps,
    maxIter=15, beta0_ratio=1e0,
    coolingFactor=2, coolingRate=2,
    maxIterLS=20, maxIterCG=10, LSshorten=0.5,
    eta_lower=1e-5, eta_upper=1,
    tau_lower=1e-6, tau_upper=2.,
    c_lower=1e-2, c_upper=1.,
    is_log=True,
    mref=None
):
    """
    Run Spectral Spectral IP inversion
    """
    dmisfit = DataMisfit.l2_DataMisfit(survey)
    uncert = abs(survey.dobs) * std + eps
    dmisfit.W = 1./uncert
    # Map for a regularization
    # Related to inversion

    # Set Upper and Lower bounds
    e = np.ones(actind.sum())

    if np.isscalar(eta_lower):
        eta_lower = e * eta_lower
    if np.isscalar(tau_lower):
        tau_lower = e * tau_lower
    if np.isscalar(c_lower):
        c_lower = e * c_lower

    if np.isscalar(eta_upper):
        eta_upper = e * eta_upper
    if np.isscalar(tau_upper):
        tau_upper = e * tau_upper
    if np.isscalar(c_upper):
        c_upper = e * c_upper

    if is_log:
        m_upper = np.log(np.r_[eta_upper, tau_upper, c_upper])
        m_lower = np.log(np.r_[eta_lower, tau_lower, c_lower])
    else:
        m_upper = np.r_[eta_upper, tau_upper, c_upper]
        m_lower = np.r_[eta_lower, tau_lower, c_lower]

    # Set up regularization
    reg_eta = Regularization.Tikhonov(
        mesh, mapping=wires.eta, indActive=actind
        )
    reg_tau = Regularization.Tikhonov(
        mesh, mapping=wires.tau, indActive=actind
        )

    reg_c = Regularization.Tikhonov(mesh, mapping=wires.c, indActive=actind)

    # Todo:
    reg_eta.alpha_s = 1e-6
    reg_tau.alpha_s = 1./mesh.hx.min()
    reg_c.alpha_s = 1./mesh.hx.min()

    reg = reg_eta + reg_tau + reg_c

    # Use Projected Gauss Newton scheme
    opt = Optimization.ProjectedGNCG(
        maxIter=maxIter, upper=m_upper, lower=m_lower,
        maxIterLS=maxIterLS, maxIterCG=maxIterCG, LSshorten=LSshorten
        )
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
    beta = Directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    target = Directives.TargetMisfit()

    directiveList = [
            beta, betaest, target
    ]
    inv = Inversion.BaseInversion(
        invProb, directiveList=directiveList
        )
    opt.LSshorten = 0.5
    opt.remember('xc')

    # Run inversion
    mopt = inv.run(m0)
    return mopt, invProb.dpred
