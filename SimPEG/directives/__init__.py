"""
=============================================
Directives (:mod:`simpeg.directives`)
=============================================

.. currentmodule:: simpeg.directives

Directives are classes that allow us to control the inversion, perform tasks
between iterations, save information about our inversion process and more.
Directives are passed to the ``simpeg.inversion.BaseInversion`` class through
the ``directiveList`` argument. The tasks specified through the directives are
executed after each inversion iteration, following the same order as in which
they are passed in the ``directiveList``.

Although you can write your own directive classes and plug them into your
inversion, we provide a set of useful directive classes that cover a wide range
of applications:


General purpose directives
==========================

.. autosummary::
   :toctree: generated/

   AlphasSmoothEstimate_ByEig
   BetaEstimateMaxDerivative
   BetaEstimate_ByEig
   BetaSchedule
   JointScalingSchedule
   MultiTargetMisfits
   ProjectSphericalBounds
   ScalingMultipleDataMisfits_ByEig
   TargetMisfit
   UpdatePreconditioner
   UpdateSensitivityWeights
   Update_Wj


Directives to save inversion results
====================================

.. autosummary::
   :toctree: generated/

   SaveEveryIteration
   SaveModelEveryIteration
   SaveOutputDictEveryIteration
   SaveOutputEveryIteration


Directives related to sparse inversions
=======================================

.. autosummary::
   :toctree: generated/

   Update_IRLS


Directives related to PGI
=========================

.. autosummary::
   :toctree: generated/

    PGI_AddMrefInSmooth
    PGI_BetaAlphaSchedule
    PGI_UpdateParameters


Directives related to joint inversions
======================================

.. autosummary::
   :toctree: generated/

    SimilarityMeasureInversionDirective
    SimilarityMeasureSaveOutputEveryIteration
    PairedBetaEstimate_ByEig
    PairedBetaSchedule
    MovingAndMultiTargetStopping


Base directive classes
======================
The ``InversionDirective`` class defines the basic class for all directives.
Inherit from this class when writing your own directive. The ``DirectiveList``
is used under the hood to handle the execution of all directives passed to the
``simpeg.inversion.BaseInversion``.

.. autosummary::
   :toctree: generated/

   InversionDirective
   DirectiveList

"""

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
    Update_IRLS,
    UpdatePreconditioner,
    Update_Wj,
    AlphasSmoothEstimate_ByEig,
    MultiTargetMisfits,
    ScalingMultipleDataMisfits_ByEig,
    JointScalingSchedule,
    UpdateSensitivityWeights,
    ProjectSphericalBounds,
)

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
