.. _testing:

Testing
=======

.. image:: https://dev.azure.com/simpeg/simpeg/_apis/build/status/simpeg.simpeg?branchName=main
    :target: https://dev.azure.com/simpeg/simpeg/_build/latest?definitionId=2&branchName=main
    :alt: Azure pipeline

.. image:: https://codecov.io/gh/simpeg/simpeg/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/simpeg/simpeg
    :alt: Coverage status

On each update, SimPEG is tested using the continuous integration service
`Azure pipelines <https://azure.microsoft.com/en-us/products/devops/pipelines>`_.
We use `Codecov <https://codecov.io>`_ to check and provide stats on how much
of the code base is covered by tests. This tells which lines of code have been
run in the test suite. It does not tell you about the quality of the tests run!
In order to assess that, have a look at the tests we are running - they tell you
the assumptions that we do not want to break within the code base.

Within the repository, the tests are located in the top-level **tests**
directory. Tests are organized similar to the structure of the repository.
There are several types of tests we employ, this is not an exhaustive list,
but meant to provide a few places to look when you are developing and would
like to check that the code you wrote satisfies the assumptions you think it
should.

Testing is performed with :code:`pytest` which is available through PyPI.
Checkout the docs on `pytest <https://docs.pytest.org/>`_.


Compare with known values
-------------------------

In a simple case, you might know the exact value of what the output should be
and you can :code:`assert` that this is in fact the case. For example,
we setup a 3D :code:`BaseRectangularMesh` and assert that it has 3 dimensions.

.. code:: python

    from discretize.base import BaseRectangularMesh
    import numpy as np

    mesh = BaseRectangularMesh([6, 2, 3])

    def test_mesh_dimensions():
        assert mesh.dim == 3

All functions with the naming convention :code:`test_XXX`
are run. Here we check that the dimensions are correct for the 3D mesh.

If the value is not an integer, you can be subject to floating point errors,
so :code:`assert ==` might be too harsh. In this case, you will want to use
the ``numpy.testing`` module to check for approximate equals. For instance,

.. code:: python

    import numpy as np
    import discretize
    from simpeg import maps

    def test_map_multiplication(self):
        mesh = discretize.TensorMesh([2,3])
        exp_map = maps.ExpMap(mesh)
        vert_map = maps.SurjectVertical1D(mesh)
        combo = exp_map*vert_map
        m = np.arange(3.0)
        t_true = np.exp(np.r_[0,0,1,1,2,2.])
        np.testing.assert_allclose(combo * m, t_true)

These are rather simple examples, more advanced tests might include `solving an
electromagnetic problem numerically and comparing it to an analytical solution
<https://github.com/simpeg/simpeg/blob/main/tests/em/fdem/forward/test_FDEM_analytics.py>`_
, or `performing an adjoint test
<https://github.com/simpeg/simpeg/blob/main/tests/em/fdem/inverse/adjoint/test_FDEM_adjointEB.py>`_
to test :code:`Jvec` and :code:`Jtvec`.


.. _order_test:

Order and Derivative Tests
--------------------------

Order tests can be used when you are testing differential operators (we are
using a second-order, staggered grid discretization for our operators). For
example, testing a 2D curl operator in `test_operators.py
<https://github.com/simpeg/discretize/blob/main/tests/base/test_operators.py>`_

.. code:: python

    import numpy as np
    import unittest
    from discretize.tests import OrderTest

    class TestCurl2D(OrderTest):
        name = "Cell Grad 2D - Dirichlet"
        meshTypes = ['uniformTensorMesh']
        meshDimension = 2
        meshSizes = [8, 16, 32, 64]

        def getError(self):
            # Test function
            ex = lambda x, y: np.cos(y)
            ey = lambda x, y: np.cos(x)
            sol = lambda x, y: -np.sin(x)+np.sin(y)

            sol_curl2d = call2(sol, self.M.gridCC)
            Ec = cartE2(self.M, ex, ey)
            sol_ana = self.M.edge_curl*self.M.project_face_vector(Ec)
            err = np.linalg.norm((sol_curl2d-sol_ana), np.inf)

            return err

        def test_order(self):
            self.orderTest()

Derivative tests are a particular type of :ref:`order_test`, and since they
are used so extensively, discretize includes a :code:`check_derivative` method.

In the case
of testing a derivative, we consider a Taylor expansion of a function about
:math:`x`. For a small perturbation :math:`\Delta x`,

.. math::

    f(x + \Delta x) \simeq f(x) + J(x) \Delta x + \mathcal{O}(h^2)

As :math:`\Delta x` decreases, we expect :math:`\|f(x) - f(x + \Delta x)\|` to
have first order convergence (e.g. the improvement in the approximation is
directly related to how small :math:`\Delta x` is, while if we include the
first derivative in our approximation, we expect that :math:`\|f(x) +
J(x)\Delta x - f(x + \Delta x)\|` to converge at a second-order rate. For
example, all `maps have an associated derivative test <https://github.com/simpeg/simpeg/blob/main/simpeg/maps.py#L127>`_ . An example from `test_FDEM_derivs.py <ht
tps://github.com/simpeg/simpeg/blob/main/tests/em/fdem/inverse/derivs/test_F
DEM_derivs.py>`_

.. code:: python

    def deriv_test(fdemType, comp):

        # setup simulation, survey

        def fun(x):
            return survey.dpred(x), lambda x: sim.Jvec(x0, x)
        return tests.check_derivative(fun, x0, num=2, plotIt=False, eps=FLR)
