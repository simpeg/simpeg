.. _examples_Mesh_Operators_CahnHilliard:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Mesh: Operators: Cahn Hilliard
==============================

This example is based on the example in the FiPy_ library.
Please see their documentation for more information about the Cahn-Hilliard equation.

The "Cahn-Hilliard" equation separates a field \\( \\phi \\) into 0 and 1 with smooth transitions.

.. math::

    \frac{\partial \phi}{\partial t} = \nabla \cdot D \nabla \left( \frac{\partial f}{\partial \phi} - \epsilon^2 \nabla^2 \phi \right)

Where \\( f \\) is the energy function \\( f = ( a^2 / 2 )\\phi^2(1 - \\phi)^2 \\)
which drives \\( \\phi \\) towards either 0 or 1, this competes with the term
\\(\\epsilon^2 \\nabla^2 \\phi \\) which is a diffusion term that creates smooth changes in \\( \\phi \\).
The equation can be factored:

.. math::

    \frac{\partial \phi}{\partial t} = \nabla \cdot D \nabla \psi \\
    \psi = \frac{\partial^2 f}{\partial \phi^2} (\phi - \phi^{\text{old}}) + \frac{\partial f}{\partial \phi} - \epsilon^2 \nabla^2 \phi

Here we will need the derivatives of \\( f \\):

.. math::

    \frac{\partial f}{\partial \phi} = (a^2/2)2\phi(1-\phi)(1-2\phi)
    \frac{\partial^2 f}{\partial \phi^2} = (a^2/2)2[1-6\phi(1-\phi)]

The implementation below uses backwards Euler in time with an exponentially increasing time step.
The initial \\( \\phi \\) is a normally distributed field with a standard deviation of 0.1 and mean of 0.5.
The grid is 60x60 and takes a few seconds to solve ~130 times. The results are seen below, and you can see the
field separating as the time increases.

.. _FiPy: http://www.ctcms.nist.gov/fipy/examples/cahnHilliard/generated/examples.cahnHilliard.mesh2DCoupled.html



.. plot::

    from SimPEG import Examples
    Examples.Mesh_Operators_CahnHilliard.run()

.. literalinclude:: ../../SimPEG/Examples/Mesh_Operators_CahnHilliard.py
    :language: python
    :linenos:
