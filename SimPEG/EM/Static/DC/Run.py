import numpy as np
from SimPEG import (Maps, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)


def run_inversion(
    m0, survey, actind, mesh,
    std, eps,
    maxIter=15, beta0_ratio=1e0,
    coolingFactor=5, coolingRate=2,
    upper=np.inf, lower=-np.inf,
    use_sensitivity_weight=True,
    alpha_s=1e-4,
    alpha_x=1.,
    alpha_y=1.,
    alpha_z=1.,
):
    """
    Run DC inversion
    """
    dmisfit = DataMisfit.l2_DataMisfit(survey)
    uncert = abs(survey.dobs) * std + eps
    dmisfit.W = 1./uncert
    # Map for a regularization
    regmap = Maps.IdentityMap(nP=int(actind.sum()))
    # Related to inversion
    if use_sensitivity_weight:
        reg = Regularization.Simple(mesh, indActive=actind, mapping=regmap)
        reg.alpha_s = alpha_s
        reg.alpha_x = alpha_x
        reg.alpha_y = alpha_y
        reg.alpha_z = alpha_z
    else:
        reg = Regularization.Tikhonov(mesh, indActive=actind, mapping=regmap)
        reg.alpha_s = alpha_s
        reg.alpha_x = alpha_x
        reg.alpha_y = alpha_y
        reg.alpha_z = alpha_z

    opt = Optimization.ProjectedGNCG(maxIter=maxIter, upper=upper, lower=lower)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
    beta = Directives.BetaSchedule(
        coolingFactor=coolingFactor, coolingRate=coolingRate
    )
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    target = Directives.TargetMisfit()
    # Need to have basice saving function
    update_Jacobi = Directives.UpdatePreconditioner()
    if use_sensitivity_weight:
        updateSensW = Directives.UpdateSensitivityWeights()
        directiveList = [
            beta, betaest, target, updateSensW, update_Jacobi
        ]
    else:
        directiveList = [
            beta, betaest, target, update_Jacobi
        ]
    inv = Inversion.BaseInversion(
        invProb, directiveList=directiveList
        )
    opt.LSshorten = 0.5
    opt.remember('xc')

    # Run inversion
    mopt = inv.run(m0)
    return mopt, invProb.dpred

