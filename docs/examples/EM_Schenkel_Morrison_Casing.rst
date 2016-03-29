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

 we will create and run a FDEM forward simulation based on the
nkel and Morrison Casing Model

chenkel, C.J., and H.F. Morrison, 1990`_, Effects of well casing on potential field measurements using downhole current sources: Geophysical prospecting, 38, 663-686.

Schenkel, C.J., and H.F. Morrison, 1990: http://onlinelibrary.wiley.com/store/10.1111/j.1365-2478.1990.tb01868.x/asset/j.1365-2478.1990.tb01868.x.pdf?v=1&t=imdt3o85&s=38248af166c2887ed587d94c8ccafad30b480529

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



.. plot::

    from SimPEG import Examples
    Examples.EM_Schenkel_Morrison_Casing.run()

.. literalinclude:: ../../SimPEG/Examples/EM_Schenkel_Morrison_Casing.py
    :language: python
    :linenos:
