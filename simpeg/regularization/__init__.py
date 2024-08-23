r"""
=============================================
Regularization (:mod:`simpeg.regularization`)
=============================================

.. currentmodule:: simpeg.regularization

``Regularization`` classes are used to impose constraints on models recovered through geophysical
inversion. Constraints may be straight forward, such as: requiring the recovered model be
spatially smooth, or using a reference model to add a-priori information. Constraints may also
be more sophisticated; e.g. cross-validation and petrophysically-guided regularization.
In SimPEG, constraints on the recovered model can be defined using a single ``Regularization``
object, or defined as a weighted sum of ``Regularization`` objects.

Basic Theory
------------

Most geophysical inverse problems suffer from non-uniqueness; i.e. there is an infinite number
of models (:math:`m`) capable of reproducing the observed data to within a specified
degree of uncertainty. The challenge is recovering a model which 1) reproduces the observed data,
and 2) reasonably approximates the subsurface structures responsible for the observed geophysical
response. To accomplish this, regularization is used to ensure the solution to the inverse
problem is unique and is geologically plausible. The regularization applied to solve the inverse
problem depends on user assumptions and a priori information.

SimPEG uses a deterministic inversion approach to recover an appropriate model.
The algorithm does this by finding the model (:math:`m`) which minimizes a global objective
function (or penalty function) of the form:

.. math::
    \phi (m) = \phi_d (m) + \beta \, \phi_m (m)

The global objective function contains two terms: a data misfit term :math:`\phi_d` which
ensures data predicted by the recovered model adequately reproduces the observed data,
and the model objective function :math:`\phi_m` which is comprised of one or more
regularization functions (objective functions) :math:`\phi_i (m)`. I.e.:

.. math::
    \phi_m (m) = \sum_i \alpha_i \, \phi_i (m)

The model objective function imposes all the desired constraints on the recovered model.
Constants :math:`\alpha_i` weight the relative contributions of the regularization
functions comprising the model objective function. The trade-off parameter :math:`\beta`
balances the relative contribution of the data misfit and regularization functions on the
global objective function.

Regularization classes within SimPEG correspond to different regularization (objective)
functions that can be used individually or combined to define the model objective function
:math:`\phi_m (\mathbf{m})`. For example, a combination of regularization functions that ensures
the values in the recovered model are not too large and are spatially smooth in the x and
y-directions can be expressed as:

.. math::
    \phi_m (m) =
    \alpha_s \! \int_\Omega \Bigg [ w_s(r) \, m(r)^2 \Bigg ] \, dv +
    \alpha_x \! \int_\Omega \Bigg [ w_x(r)
    \bigg ( \frac{\partial m}{\partial x} \bigg )^2 \Bigg ] \, dv +
    \alpha_y \! \int_\Omega \Bigg [ w_y(r)
    \bigg ( \frac{\partial m}{\partial y} \bigg )^2 \Bigg ] \, dv

where :math:`w_s(r), w_x(r), w_y(r)` are user-defined weighting functions.
For practical implementation within SimPEG, the regularization function and all its dependent
variables are discretized to a numerical grid (or mesh). The model is therefore defined as a
discrete set of model parameters :math:`\mathbf{m}`.
And the regularization is implemented using a weighted sum of objective functions:

.. math::
    \phi_m (\mathbf{m}) \approx \alpha_s \big \| \mathbf{W_s m} \big \|^2 +
    \alpha_x \big \| \mathbf{W_x G_x m} \big \|^2 +
    \alpha_y \big \| \mathbf{W_y G_y m} \big \|^2

where :math:`\mathbf{G_x}` and :math:`\mathbf{G_y}` are partial gradient operators along the x and
y-directions, respectively. :math:`\mathbf{W_s}`, :math:`\mathbf{W_x}` and :math:`\mathbf{W_y}`
are weighting matrices that apply user-defined weights and account for cell dimensions
in the inversion mesh.


The API
=======

Weighted Least Squares Regularization
-------------------------------------
Weighted least squares regularization functions are defined as weighted L2-norms on the model,
its first-order directional derivative(s), or its second-order directional derivative(s).

.. autosummary::
  :toctree: generated/

  WeightedLeastSquares
  Smallness
  SmoothnessFirstOrder
  SmoothnessSecondOrder
  SmoothnessFullGradient

Sparse Norm Regularization
--------------------------
Sparse norm regularization allows for the recovery of compact and/or blocky structures.
An iteratively re-weighted least-squares approach allows smallness and smoothness
regularization functions to be defined using norms between 0 and 2.

.. autosummary::
  :toctree: generated/

  Sparse
  SparseSmallness
  SparseSmoothness

Vector Regularizations
----------------------
Vector regularization allows for the recovery of vector models; that is, a model
where the parameters for each cell define directional components of a vector quantity.

.. autosummary::
  :toctree: generated/

  CrossReferenceRegularization
  VectorAmplitude
  AmplitudeSmallness
  AmplitudeSmoothnessFirstOrder

Joint Regularizations
---------------------
Regularization functions for joint inversion involving one or more physical properties.

.. autosummary::
  :toctree: generated/

  CrossGradient
  JointTotalVariation
  PGI
  PGIsmallness
  LinearCorrespondence

Base Regularization Classes
---------------------------
Base regularization classes. Inherited by other classes and not used directly
to constrain inversions.

.. autosummary::
  :toctree: generated/

  RegularizationMesh
  BaseRegularization
  BaseSimilarityMeasure
  BaseSparse
  BaseVectorRegularization
  BaseAmplitude

"""

from ..utils.code_utils import deprecate_class
from .base import (
    BaseRegularization,
    WeightedLeastSquares,
    BaseSimilarityMeasure,
    Smallness,
    SmoothnessFirstOrder,
    SmoothnessSecondOrder,
)
from .regularization_mesh import RegularizationMesh
from .sparse import BaseSparse, SparseSmallness, SparseSmoothness, Sparse
from .pgi import PGIsmallness, PGI
from .cross_gradient import CrossGradient
from .correspondence import LinearCorrespondence
from .jtv import JointTotalVariation
from .vector import (
    BaseVectorRegularization,
    CrossReferenceRegularization,
    BaseAmplitude,
    VectorAmplitude,
    AmplitudeSmallness,
    AmplitudeSmoothnessFirstOrder,
)
from ._gradient import SmoothnessFullGradient


@deprecate_class(removal_version="0.19.0", error=True)
class SimpleSmall(Smallness):
    """Deprecated class, replaced by Smallness."""

    pass


@deprecate_class(removal_version="0.19.0", error=True)
class SimpleSmoothDeriv(SmoothnessFirstOrder):
    """Deprecated class, replaced by SmoothnessFirstOrder."""

    pass


@deprecate_class(removal_version="0.19.0", error=True)
class Simple(WeightedLeastSquares):
    """Deprecated class, replaced by WeightedLeastSquares."""

    def __init__(self, mesh=None, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0, **kwargs):
        # These alphas are now refered to as length_scalse in the
        # new WeightedLeastSquares regularization
        super().__init__(
            mesh=mesh,
            length_scale_x=alpha_x,
            length_scale_y=alpha_y,
            length_scale_z=alpha_z,
            **kwargs,
        )


@deprecate_class(removal_version="0.19.0", error=True)
class Tikhonov(WeightedLeastSquares):
    """Deprecated class, replaced by WeightedLeastSquares."""

    def __init__(
        self, mesh=None, alpha_s=1e-6, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0, **kwargs
    ):
        super().__init__(
            mesh=mesh,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            **kwargs,
        )


@deprecate_class(removal_version="0.19.0", error=True)
class Small(Smallness):
    """Deprecated class, replaced by Smallness."""

    pass


@deprecate_class(removal_version="0.19.0", error=True)
class SmoothDeriv(SmoothnessFirstOrder):
    """Deprecated class, replaced by SmoothnessFirstOrder."""

    pass


@deprecate_class(removal_version="0.19.0", error=True)
class SmoothDeriv2(SmoothnessSecondOrder):
    """Deprecated class, replaced by SmoothnessSecondOrder."""

    pass


@deprecate_class(removal_version="0.19.0", error=True)
class PGIwithNonlinearRelationshipsSmallness(PGIsmallness):
    """Deprecated class, replaced by PGIsmallness."""

    def __init__(self, gmm, **kwargs):
        super().__init__(gmm, non_linear_relationships=True, **kwargs)


@deprecate_class(removal_version="0.19.0", error=True)
class PGIwithRelationships(PGI):
    """Deprecated class, replaced by PGI."""

    def __init__(self, mesh, gmmref, **kwargs):
        super().__init__(mesh, gmmref, non_linear_relationships=True, **kwargs)
