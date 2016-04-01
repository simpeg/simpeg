.. _examples_EM_Schenkel_Morrison_Casing:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


EM: Schenkel and Morrison Casing Model
======================================

Here we create and run a FDEM forward simulation to calculate the vertical
current inside a steel-cased. The model is based on the Schenkel and
Morrison Casing Model, and the results are used in a 2016 SEG abstract by
Yang et al.

- Schenkel, C.J., and H.F. Morrison, 1990, Effects of well casing on potential field measurements using downhole current sources: Geophysical prospecting, 38, 663-686.


The model consists of:
- Air: Conductivity 1e-8 S/m, above z = 0
- Background: conductivity 1e-2 S/m, below z = 0
- Casing: conductivity 1e6 S/m
    - 300m long
    - radius of 0.1m
    - thickness of 6e-3m

Inside the casing, we take the same conductivity as the background.

We are using an EM code to simulate DC, so we use frequency low enough
that the skin depth inside the casing is longer than the casing length (f
= 1e-6 Hz). The plot produced is of the current inside the casing.

These results are shown in the SEG abstract by Yang et al., 2016: 3D DC
resistivity modeling of steel casing for reservoir monitoring using
equivalent resistor network. The solver used to produce these results and
achieve the CPU time of ~30s is Mumps, which was installed using pymatsolver_

.. _pymatsolver: https://github.com/rowanc1/pymatsolver

This example is on figshare: https://dx.doi.org/10.6084/m9.figshare.3126961.v1

If you would use this example for a code comparison, or build upon it, a
citation would be much appreciated!



.. plot::

    from SimPEG import Examples
    Examples.EM_Schenkel_Morrison_Casing.run()

.. literalinclude:: ../../SimPEG/Examples/EM_Schenkel_Morrison_Casing.py
    :language: python
    :linenos:
