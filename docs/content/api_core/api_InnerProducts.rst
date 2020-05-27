.. _api_InnerProducts:


Inner Products
**************

By using the weak formulation of many of the PDEs in geophysical applications,
we can rapidly develop discretizations. Much of this work, however, needs a
good understanding of how to approximate inner products on our discretized
meshes. We will define the inner product as:

.. math::

    \left(a,b\right) = \int_\Omega{a \cdot b}{\partial v}

where a and b are either scalars or vectors.

.. note::

    The InnerProducts class is a base class providing inner product matrices
    for meshes and cannot run on its own.


Example problem for DC resistivity
----------------------------------

We will start with the formulation of the Direct Current (DC) resistivity
problem in geophysics.


.. math::

        \frac{1}{\sigma}\vec{j} = \nabla \phi \\

        \nabla\cdot \vec{j} = q

In the following discretization, :math:`\sigma` and :math:`\phi`
will be discretized on the cell-centers and the flux, :math:`\vec{j}`,
will be on the faces. We will use the weak formulation to discretize
the DC resistivity equation.

We can define in weak form by integrating with a general face function
:math:`\vec{f}`:

.. math::

    \int_{\Omega}{\sigma^{-1}\vec{j} \cdot \vec{f}} = \int_{\Omega}{\nabla \phi  \cdot \vec{f}}

Here we can integrate the right side by parts,

.. math::

    \nabla\cdot(\phi\vec{f})=\nabla\phi\cdot\vec{f} + \phi\nabla\cdot\vec{f}

and rearrange it, and apply the Divergence theorem.

.. math::

    \int_{\Omega}{\sigma^{-1}\vec{j} \cdot \vec{f}} =
    - \int_{\Omega}{(\phi \nabla \cdot \vec{f})}
    + \int_{\partial \Omega}{ \phi  \vec{f} \cdot \mathbf{n}}

We can then discretize for every cell:

.. math::

    v_{\text{cell}} \sigma^{-1} (\mathbf{J}_x \mathbf{F}_x +\mathbf{J}_y \mathbf{F}_y  + \mathbf{J}_z \mathbf{F}_z ) = -\phi^{\top} v_{\text{cell}} \mathbf{D}_{\text{cell}} \mathbf{F}  + \text{BC}

.. note::

    We have discretized the dot product above, but remember that we do not
    really have a single vector :math:`\mathbf{J}`, but approximations of
    :math:`\vec{j}` on each face of our cell. In 2D that means 2
    approximations of :math:`\mathbf{J}_x` and 2 approximations of
    :math:`\mathbf{J}_y`. In 3D we also have 2 approximations of
    :math:`\mathbf{J}_z`.

Regardless of how we choose to approximate this dot product, we can represent
this in vector form (again this is for every cell), and will generalize for
the case of anisotropic (tensor) sigma.

.. math::

    \mathbf{F}_c^{\top} (\sqrt{v_{\text{cell}}} \Sigma^{-1} \sqrt{v_{\text{cell}}})  \mathbf{J}_c =
    -\phi^{\top} v_{\text{cell}} \mathbf{D}_{\text{cell}} \mathbf{F})
    + \text{BC}

We multiply by  square-root of volume on each side of the tensor conductivity
to keep symmetry in the system. Here :math:`\mathbf{J}_c` is the Cartesian
:math:`\mathbf{J}` (on the faces that we choose to use in our approximation)
and must be calculated differently depending on the mesh:

.. math::
    \mathbf{J}_c = \mathbf{Q}_{(i)}\mathbf{J}_\text{TENSOR} \\
    \mathbf{J}_c = \mathbf{N}_{(i)}^{-1}\mathbf{Q}_{(i)}\mathbf{J}_\text{Curv}

Here the :math:`i` index refers to where we choose to approximate this integral, as discussed in the note above.
We will approximate this integral by taking the fluxes clustered around every node of the cell, there are 8 combinations in 3D, and 4 in 2D. We will use a projection matrix :math:`\mathbf{Q}_{(i)}` to pick the appropriate fluxes. So, now that we have 8 approximations of this integral, we will just take the average. For the TensorMesh, this looks like:

.. math::

    \mathbf{F}^{\top}
        {1\over 8}
        \left(\sum_{i=1}^8
        \mathbf{Q}_{(i)}^{\top} \sqrt{v_{\text{cell}}} \Sigma^{-1} \sqrt{v_{\text{cell}}}  \mathbf{Q}_{(i)}
        \right)
        \mathbf{J}
        =
        -\mathbf{F}^{\top} \mathbf{D}_{\text{cell}}^{\top} v_{\text{cell}} \phi   + \text{BC}

Or, when generalizing to the entire mesh and dropping our general face function:

.. math::

    \mathbf{M}^f_{\Sigma^{-1}} \mathbf{J}
        =
        - \mathbf{D}^{\top} \text{diag}(\mathbf{v}) \phi   + \text{BC}

By defining the faceInnerProduct (8 combinations of fluxes in 3D, 4 in 2D, 2 in 1D) to be:

.. math::

    \mathbf{M}^f_{\Sigma^{-1}} =
        \sum_{i=1}^{2^d}
        \mathbf{P}_{(i)}^{\top} \Sigma^{-1} \mathbf{P}_{(i)}

Where :math:`d` is the dimension of the mesh.
The :math:`\mathbf{M}^f` is returned when given the input of :math:`\Sigma^{-1}`.

Here each :math:`\mathbf{P} ~ \in ~ \mathbb{R}^{(d*nC, nF)}` is a combination
of the projection, volume, and any normalization to Cartesian coordinates
(where the dot product is well defined):

.. math::

    \mathbf{P}_{(i)} =  \sqrt{ \frac{1}{2^d} \mathbf{I}^d \otimes \text{diag}(\mathbf{v})} \overbrace{\mathbf{N}_{(i)}^{-1}}^{\text{Curv only}} \mathbf{Q}_{(i)}

.. note::

    This is actually completed for each cell in the mesh at the same time, and the full matrices are returned.

If ``returnP=True`` is requested in any of these methods the projection matrices are returned as a list ordered by nodes around which the fluxes were approximated::

    # In 3D
    P = [P000, P100, P010, P110, P001, P101, P011, P111]
    # In 2D
    P = [P00, P10, P01, P11]
    # In 1D
    P = [P0, P1]

The derivation for ``edgeInnerProducts`` is exactly the same, however, when we
approximate the integral using the fields around each node, the projection
matrices look a bit different because we have 12 edges in 3D instead of just 6
faces. The interface to the code is exactly the same.


Defining Tensor Properties
--------------------------

**For 3D:**

Depending on the number of columns (either 1, 3, or 6) of mu, the material
property is interpreted as follows:

.. math::

    \vec{\mu} = \left[\begin{matrix} \mu_{1} & 0 & 0 \\ 0 & \mu_{1} & 0 \\ 0 & 0 & \mu_{1}  \end{matrix}\right]

    \vec{\mu} = \left[\begin{matrix} \mu_{1} & 0 & 0 \\ 0 & \mu_{2} & 0 \\ 0 & 0 & \mu_{3}  \end{matrix}\right]

    \vec{\mu} = \left[\begin{matrix} \mu_{1} & \mu_{4} & \mu_{5} \\ \mu_{4} & \mu_{2} & \mu_{6} \\ \mu_{5} & \mu_{6} & \mu_{3}  \end{matrix}\right]

**For 2D:**

 Depending on the number of columns (either 1, 2, or 3) of mu, the material property is interpreted as follows:

.. math::
    \vec{\mu} = \left[\begin{matrix} \mu_{1} & 0 \\ 0 & \mu_{1} \end{matrix}\right]

    \vec{\mu} = \left[\begin{matrix} \mu_{1} & 0 \\ 0 & \mu_{2} \end{matrix}\right]

    \vec{\mu} = \left[\begin{matrix} \mu_{1} & \mu_{3} \\ \mu_{3} & \mu_{2} \end{matrix}\right]


Structure of Matrices
---------------------

Both the isotropic, and anisotropic material properties result in a diagonal mass matrix.
Which is nice and easy to invert if necessary, however, in the fully anisotropic case which is not aligned with the grid, the matrix is not diagonal. This can be seen for a 3D mesh in the figure below.

.. plot::

    import numpy as np
    import discretize
    mesh = discretize.TensorMesh([10,50,3])
    m1 = np.random.rand(mesh.nC)
    m2 = np.random.rand(mesh.nC,3)
    m3 = np.random.rand(mesh.nC,6)
    M = list(range(3))
    M[0] = mesh.getFaceInnerProduct(m1)
    M[1] = mesh.getFaceInnerProduct(m2)
    M[2] = mesh.getFaceInnerProduct(m3)
    plt.figure(figsize=(13,5))
    for i, lab in enumerate(['Isotropic','Anisotropic','Tensor']):
        plt.subplot(131 + i)
        plt.spy(M[i],ms=0.5,color='k')
        plt.tick_params(axis='both',which='both',labeltop='off',labelleft='off')
        plt.title(lab + ' Material Property')
    plt.show()


Taking Derivatives
------------------

We will take the derivative of the fully anisotropic tensor for a 3D mesh, the
other cases are easier and will not be discussed here. Let us start with one
part of the sum which makes up :math:`\mathbf{M}^f_\Sigma` and take the
derivative when this is multiplied by some vector :math:`\mathbf{v}`:

.. math::

    \mathbf{P}^\top \boldsymbol{\Sigma} \mathbf{Pv}

Here we will let :math:`\mathbf{Pv} = \mathbf{y}` and :math:`\mathbf{y}` will have the form:

.. math::

    \mathbf{y} = \mathbf{Pv} =
    \left[
        \begin{matrix}
            \mathbf{y}_1\\
            \mathbf{y}_2\\
            \mathbf{y}_3\\
        \end{matrix}
    \right]

.. math::

    \mathbf{P}^\top\Sigma\mathbf{y} =
    \mathbf{P}^\top
    \left[\begin{matrix}
        \boldsymbol{\sigma}_{1} & \boldsymbol{\sigma}_{4} & \boldsymbol{\sigma}_{5} \\
        \boldsymbol{\sigma}_{4} & \boldsymbol{\sigma}_{2} & \boldsymbol{\sigma}_{6} \\
        \boldsymbol{\sigma}_{5} & \boldsymbol{\sigma}_{6} & \boldsymbol{\sigma}_{3}
    \end{matrix}\right]
    \left[
        \begin{matrix}
            \mathbf{y}_1\\
            \mathbf{y}_2\\
            \mathbf{y}_3\\
        \end{matrix}
    \right]
    =
    \mathbf{P}^\top
    \left[
        \begin{matrix}
            \boldsymbol{\sigma}_{1}\circ \mathbf{y}_1 + \boldsymbol{\sigma}_{4}\circ \mathbf{y}_2 + \boldsymbol{\sigma}_{5}\circ \mathbf{y}_3\\
            \boldsymbol{\sigma}_{4}\circ \mathbf{y}_1 + \boldsymbol{\sigma}_{2}\circ \mathbf{y}_2 + \boldsymbol{\sigma}_{6}\circ \mathbf{y}_3\\
            \boldsymbol{\sigma}_{5}\circ \mathbf{y}_1 + \boldsymbol{\sigma}_{6}\circ \mathbf{y}_2 + \boldsymbol{\sigma}_{3}\circ \mathbf{y}_3\\
        \end{matrix}
    \right]

Now it is easy to take the derivative with respect to any one of the
parameters, for example,
:math:`\frac{\partial}{\partial\boldsymbol{\sigma}_1}`

.. math::
    \frac{\partial}{\partial \boldsymbol{\sigma}_1}\left(\mathbf{P}^\top\Sigma\mathbf{y}\right)
    =
    \mathbf{P}^\top
    \left[
        \begin{matrix}
            \text{diag}(\mathbf{y}_1)\\
            0\\
            0\\
        \end{matrix}
    \right]

Whereas :math:`\frac{\partial}{\partial\boldsymbol{\sigma}_4}`, for
example, is:

.. math::
    \frac{\partial}{\partial \boldsymbol{\sigma}_4}\left(\mathbf{P}^\top\Sigma\mathbf{y}\right)
    =
    \mathbf{P}^\top
    \left[
        \begin{matrix}
            \text{diag}(\mathbf{y}_2)\\
            \text{diag}(\mathbf{y}_1)\\
            0\\
        \end{matrix}
    \right]

These are computed for each of the 8 projections, horizontally concatenated,
and returned.

The API
-------

See the `discretize docs <http://discretize.simpeg.xyz/>`_
