.. _examples_EM_FDEM_SusEffects:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


EM: FDEM: Effects of susceptibility
===================================

When airborne freqeuncy domain EM (AFEM) survey is flown over
the earth including significantly susceptible bodies (magnetite-rich rocks),
negative data is often observed in the real part of the lowest frequency
(e.g. Dighem system 900 Hz). This phenomenon mostly based upon magnetization
occurs due to a susceptible body when the magnetic field is applied.

To clarify what is happening in the earth when we are exciting the earth with
a loop source in the frequency domain we run three forward modelling:

    - F[:math:`\sigma`, :math:`\mu`]: Anomalous conductivity and susceptibility
    - F[:math:`\sigma`, :math:`\mu_0`]: Anomalous conductivity
    - F[:math:`\sigma_{air}`, :math:`\mu_0`]: primary field

We plot vector magnetic fields in the earth. For secondary fields we provide
F[:math:`\sigma`, :math:`\mu`]-F[:math:`\sigma`, :math:`\mu_0`]. Following
figure show both real and parts.



.. plot::

    from SimPEG import Examples
    Examples.EM_FDEM_SusEffects.run()

.. literalinclude:: ../../SimPEG/Examples/EM_FDEM_SusEffects.py
    :language: python
    :linenos:
