
Viscous Remanent Magnetization
==============================

`SimPEG.VRM` uses SimPEG as the framework for forward modeling and inversion viscous remanent magnetization responses in the time domain.

To solve for predicted data, we follow the framework shown below. The model is
what we invert for. This is mapped to a physical property on the simulation
mesh. A source which is used to excite the system is specified. Having a model
and a source, we can solve Maxwell's equations for fields. We sample these
fields with receivers to give us predicted data.



.. image:: ../../images/simpegEM_noMath.png
    :scale: 50%


.. toctree::
   :maxdepth: 2

   background
   api_VRM





