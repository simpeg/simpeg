from ..utils.code_utils import deprecate_class
from .directives import (
    InversionDirective,
    DirectiveList,
    BetaEstimateMaxDerivative,
    BetaEstimate_ByEig,
    BetaSchedule,
    TargetMisfit,
    SaveEveryIteration,
    SaveModelEveryIteration,
    SaveOutputEveryIteration,
    SaveOutputDictEveryIteration,
    UpdatePreconditioner,
    Update_Wj,
    AlphasSmoothEstimate_ByEig,
    MultiTargetMisfits,
    ScalingMultipleDataMisfits_ByEig,
    JointScalingSchedule,
    UpdateSensitivityWeights,
    ProjectSphericalBounds,
)
from ._regularization import UpdateIRLS

from .pgi_directives import (
    PGI_UpdateParameters,
    PGI_BetaAlphaSchedule,
    PGI_AddMrefInSmooth,
)

from .sim_directives import (
    SimilarityMeasureInversionDirective,
    SimilarityMeasureSaveOutputEveryIteration,
    PairedBetaEstimate_ByEig,
    PairedBetaSchedule,
    MovingAndMultiTargetStopping,
)


@deprecate_class(removal_version="0.22.0", future_warn=True)
class Update_IRLS(UpdateIRLS):
    """Deprecated class, replaced by UpdateIRLS."""

    pass
