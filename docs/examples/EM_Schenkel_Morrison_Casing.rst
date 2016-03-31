.. _examples_EM_Schenkel_Morrison_Casing:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Schenkel and Morrison Casing Model
==================================

 we create and run a FDEM forward simulation to calculate the vertical
ent inside a steel-cased. The model is based on the Schenkel and
ison Casing Model, and the results are used in a 2016 SEG abstract by
 et al.

henkel, C.J., and H.F. Morrison, 1990, Effects of well casing on potential field measurements using downhole current sources: Geophysical prospecting, 38, 663-686.


model consists of:
r: Conductivity 1e-8 S/m, above z = 0
ckground: conductivity 1e-2 S/m, below z = 0
sing: conductivity 1e6 S/m
- 300m long
- radius of 0.1m
- thickness of 6e-3m

de the casing, we take the same conductivity as the background.

re using an EM code to simulate DC, so we use frequency low enough
 the skin depth inside the casing is longer than the casing length (f
-6 Hz). The plot produced is of the current inside the casing.

e results are shown in the SEG abstract by Yang et al., 2016: 3D DC
stivity modeling of steel casing for reservoir monitoring using
valent resistor network. The solver used to produce these results and
eve the CPU time of ~30s is Mumps, which was installed using pymatsolver_

pymatsolver: https://github.com/rowanc1/pymatsolver

 example is on figshare: https://dx.doi.org/10.6084/m9.figshare.3126961.v1

ou would use this example for a code comparison, or build upon it, a
tion would be much appreciated!



.. plot::

    from SimPEG import Examples
    Examples.EM_Schenkel_Morrison_Casing.run()

.. literalinclude:: ../../SimPEG/Examples/EM_Schenkel_Morrison_Casing.py
    :language: python
    :linenos:
