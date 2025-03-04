.. _documentation:

Documentation
-------------

Documentation helps others use your code! Please document new contributions.
SimPEG tries to follow the `numpydoc` style of docstrings (check out the
`style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_).
SimPEG then uses `sphinx <https://www.sphinx-doc.org/>`_ to build the documentation.
When documenting a new class or function, please include a description
(with math if it solves an equation), inputs, outputs and preferably a small example.

For example:

.. code:: python


    class WeightedLeastSquares(BaseComboRegularization):
        r"""Weighted least squares measure on model smallness and smoothness.

        L2 regularization with both smallness and smoothness (first order
        derivative) contributions.

        Parameters
        ----------
        mesh : discretize.base.BaseMesh
            The mesh on which the model parameters are defined. This is used
            for constructing difference operators for the smoothness terms.
        active_cells : array_like of bool or int, optional
            List of active cell indices, or a `mesh.n_cells` boolean array
            describing active cells.
        alpha_s : float, optional
            Smallness weight
        alpha_x, alpha_y, alpha_z : float or None, optional
            First order smoothness weights for the respective dimensions.
            `None` implies setting these weights using the `length_scale`
            parameters.
        alpha_xx, alpha_yy, alpha_zz : float, optional
            Second order smoothness weights for the respective dimensions.
        length_scale_x, length_scale_y, length_scale_z : float, optional
            First order smoothness length scales for the respective dimensions.
        mapping : simpeg.maps.IdentityMap, optional
            A mapping to apply to the model before regularization.
        reference_model : array_like, optional
        reference_model_in_smooth : bool, optional
            Whether to include the reference model in the smoothness terms.
        weights : None, array_like, or dict or array_like, optional
            User defined weights. It is recommended to interact with weights using
            the `get_weights`, `set_weights` functionality.

        Notes
        -----
        The function defined here approximates:

        .. math::
            \phi_m(\mathbf{m}) = \alpha_s \| W_s (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
            + \alpha_x \| W_x \frac{\partial}{\partial x} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
            + \alpha_y \| W_y \frac{\partial}{\partial y} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
            + \alpha_z \| W_z \frac{\partial}{\partial z} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2

        Note if the key word argument `reference_model_in_smooth` is False, then mref is not
        included in the smoothness contribution.

        If length scales are used to set the smoothness weights, alphas are respectively set internally using:
        >>> alpha_x = (length_scale_x * min(mesh.edge_lengths)) ** 2
        """



Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can build the documentation pages locally to see how the new changes will
look. First, make sure that you have :ref:`created and activated an environment
<setting-up-environment>` with simpeg installed in it. Then, navigate to the
``docs`` folder:

.. code:: bash

    cd docs

And run the following to build the docs:

.. code:: bash

    make html

.. note::

   This command will build all documentation pages, including all the examples
   and tutorials. Running the examples might take considerable amount of time.

   If you want to build the docs, but avoid running the examples, you can
   alternatively run:

   .. code:: bash

       make html-noplot

Serving the documentation locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the documentation is built, you can view it using the following
command (make sure you are in the ``docs`` directory):

.. code:: bash

    make serve

It will automatically serve the docs and you can see them in your browser.

Alternatively, you can open your file manager and open the ``index.html`` file
in the ``docs/_build/html`` folder.
