
Regularization
**************

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


The API
=======

.. autoclass:: SimPEG.regularization.RegularizationMesh
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.BaseRegularization
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.BaseComboRegularization
    :members:
    :undoc-members:

Tikhonov Regularization
=======================

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
-------

.. autoclass:: SimPEG.regularization.Tikhonov
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.Small
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.SmoothDeriv
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.SmoothDeriv2
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.Simple
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.SimpleSmall
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.SimpleSmoothDeriv
    :members:
    :undoc-members:

Sparse Regularization
=====================

We have also implemented several sparse regularizations with a variable norm.

The API
-------

.. autoclass:: SimPEG.regularization.Sparse
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.SparseSmall
    :members:
    :undoc-members:

.. autoclass:: SimPEG.regularization.SparseDeriv
    :members:
    :undoc-members:
