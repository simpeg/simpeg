"""
=============================================
Regularization (:mod:`SimPEG.regularization`)
=============================================
.. currentmodule:: SimPEG.regularization

If there is one model that has a misfit that equals the desired tolerance,
then there are infinitely many other models which can fit to the same degree.
The challenge is to find that model which has the desired characteristics and
is compatible with a priori information. A single model can be selected from
an infinite ensemble by measuring the length, or norm, of each model. Then a
smallest, or sometimes largest, member can be isolated. Our goal is to design
a norm that embodies our prior knowledge and, when minimized, yields a
realistic candidate for the solution of our problem. The norm can penalize
variation from a reference model, spatial derivatives of the model, or some
combination of these.

WeightedLeastSquares Regularization
===================================

Here we will define regularization of a model, m, in general however, this
should be thought of as (m-m_ref) but otherwise it is exactly the same:

.. math::

    R(m) = \int_\Omega \frac{\alpha_x}{2}\left(\frac{\partial m}{\partial x}\right)^2 + \frac{\alpha_y}{2}\left(\frac{\partial m}{\partial y}\right)^2 \partial v

Our discrete gradient operator works on cell centers and gives the derivative
on the cell faces, which is not where we want to be evaluating this integral.
We need to average the values back to the cell-centers before we integrate. To
avoid null spaces, we square first and then average. In 2D with ij notation it
looks like this:

.. math::

    R(m) \approx \sum_{ij} \left[\frac{\alpha_x}{2}\left[\left(\frac{m_{i+1,j} - m_{i,j}}{h}\right)^2 + \left(\frac{m_{i,j} - m_{i-1,j}}{h}\right)^2\right] \\
    + \frac{\alpha_y}{2}\left[\left(\frac{m_{i,j+1} - m_{i,j}}{h}\right)^2 + \left(\frac{m_{i,j} - m_{i,j-1}}{h}\right)^2\right]
    \right]h^2

If we let D_1 be the derivative matrix in the x direction

.. math::

    \mathbf{D}_1 = \mathbf{I}_2\otimes\mathbf{d}_1

.. math::

    \mathbf{D}_2 = \mathbf{d}_2\otimes\mathbf{I}_1

Where d_1 is the one dimensional derivative:

.. math::

    \mathbf{d}_1 = \frac{1}{h} \left[ \begin{array}{cccc}
    -1 & 1 & & \\
     & \ddots & \ddots&\\
     &  & -1 & 1\end{array} \right]

.. math::

    R(m) \approx \mathbf{v}^\top \left[\frac{\alpha_x}{2}\mathbf{A}_1 (\mathbf{D}_1 m) \odot (\mathbf{D}_1 m) + \frac{\alpha_y}{2}\mathbf{A}_2 (\mathbf{D}_2 m) \odot (\mathbf{D}_2 m) \right]

Recall that this is really a just point wise multiplication, or a diagonal
matrix times a vector. When we multiply by something in a diagonal we can
interchange and it gives the same results (i.e. it is point wise)

.. math::

    \mathbf{a\odot b} = \text{diag}(\mathbf{a})\mathbf{b} = \text{diag}(\mathbf{b})\mathbf{a} = \mathbf{b\odot a}

and the transpose also is true (but the sizes have to make sense...):

.. math::

    \mathbf{a}^\top\text{diag}(\mathbf{b}) = \mathbf{b}^\top\text{diag}(\mathbf{a})

So R(m) can simplify to:

.. math::

    R(m) \approx  \mathbf{m}^\top \left[\frac{\alpha_x}{2}\mathbf{D}_1^\top \text{diag}(\mathbf{A}_1^\top\mathbf{v}) \mathbf{D}_1 +  \frac{\alpha_y}{2}\mathbf{D}_2^\top \text{diag}(\mathbf{A}_2^\top \mathbf{v}) \mathbf{D}_2 \right] \mathbf{m}

We will define W_x as:

.. math::

    \mathbf{W}_x = \sqrt{\alpha_x}\text{diag}\left(\sqrt{\mathbf{A}_1^\top\mathbf{v}}\right) \mathbf{D}_1


And then W as a tall matrix of all of the different regularization terms:

.. math::

    \mathbf{W} = \left[ \begin{array}{c}
    \mathbf{W}_s\\
    \mathbf{W}_x\\
    \mathbf{W}_y\end{array} \right]

Then we can write

.. math::

    R(m) \approx \frac{1}{2}\mathbf{m^\top W^\top W m}


The API
=======

Least Squares Regularizations
-----------------------------
.. autosummary::
  :toctree: generated/

  WeightedLeastSquares
  Smallness
  SmoothnessFirstOrder
  SmoothnessSecondOrder

Sparse Regularizations
----------------------
We have also implemented several sparse regularizations with a variable norm.

.. autosummary::
  :toctree: generated/

  Sparse
  SparseSmallness
  SparseSmoothness

Joint Regularizations
---------------------
There are several joint inversion regularizers available

.. autosummary::
  :toctree: generated/

  CrossGradient
  JointTotalVariation
  PGI
  PGIsmallness
  LinearCorrespondence

Base Regularization classes
---------------------------
.. autosummary::
  :toctree: generated/

  RegularizationMesh
  BaseRegularization
  BaseSimilarityMeasure
  BaseSparse

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


@deprecate_class(removal_version="0.x.0", future_warn=True)
class SimpleSmall(Smallness):
    """Deprecated class, replaced by Smallness."""

    pass


@deprecate_class(removal_version="0.x.0", future_warn=True)
class SimpleSmoothDeriv(SmoothnessFirstOrder):
    """Deprecated class, replaced by SmoothnessFirstOrder."""

    pass


@deprecate_class(removal_version="0.x.0", future_warn=True)
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
            **kwargs
        )


@deprecate_class(removal_version="0.x.0", future_warn=True)
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
            **kwargs
        )


@deprecate_class(removal_version="0.x.0", future_warn=True)
class Small(Smallness):
    """Deprecated class, replaced by Smallness."""

    pass


@deprecate_class(removal_version="0.x.0", future_warn=True)
class SmoothDeriv(SmoothnessFirstOrder):
    """Deprecated class, replaced by SmoothnessFirstOrder."""

    pass


@deprecate_class(removal_version="0.x.0", future_warn=True)
class SmoothDeriv2(SmoothnessSecondOrder):
    """Deprecated class, replaced by SmoothnessSecondOrder."""

    pass


@deprecate_class(removal_version="0.x.0", future_warn=True)
class PGIwithNonlinearRelationshipsSmallness(PGIsmallness):
    """Deprecated class, replaced by PGIsmallness."""

    def __init__(self, gmm, **kwargs):
        super().__init__(gmm, non_linear_relationships=True, **kwargs)


@deprecate_class(removal_version="0.x.0", future_warn=True)
class PGIwithRelationships(PGI):
    """Deprecated class, replaced by PGI."""

    def __init__(self, mesh, gmmref, **kwargs):
        super().__init__(mesh, gmmref, non_linear_relationships=True, **kwargs)
