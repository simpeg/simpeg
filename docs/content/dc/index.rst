.. math::

    \renewcommand{\div}{\nabla\cdot\,}
    \newcommand{\grad}{\vec \nabla}
    \newcommand{\curl}{{\vec \nabla}\times\,}
    \newcommand{\dcurl}{{\mathbf C}}
    \newcommand{\dgrad}{{\mathbf G}}
    \newcommand{\Acf}{{\mathbf A_c^f}}
    \newcommand{\Ace}{{\mathbf A_c^e}}
    \renewcommand{\S}{{\mathbf \Sigma}}
    \renewcommand{\Div}{{\mathbf {Div}}}
    \renewcommand{\Grad}{{\mathbf {Grad}}}
    \newcommand{\St}{{\mathbf \Sigma_\tau}}
    \newcommand{\diag}{\mathbf{diag}}
    \newcommand{\M}{{\mathbf M}}
    \newcommand{\Me}{{\M^e}}
    \newcommand{\Mes}[1]{{\M^e_{#1}}}
    \newcommand{\be}{\mathbf{e}}
    \newcommand{\bj}{\mathbf{j}}
    \newcommand{\bphi}{\mathbf{\phi}}
    \newcommand{\bq}{\mathbf{q}}
    \newcommand{\bJ}{\mathbf{J}}
    \newcommand{\bG}{\mathbf{G}}
    \newcommand{\bP}{\mathbf{P}}
    \newcommand{\bA}{\mathbf{A}}
    \newcommand{\bm}{\mathbf{m}}
    \newcommand{\B}{\vec{B}}
    \newcommand{\D}{\vec{D}}
    \renewcommand{\H}{\vec{H}}
    \renewcommand {\j}  { {\vec j} }
    \newcommand {\h}  { {\vec h} }
    \renewcommand {\b}  { {\vec b} }
    \newcommand {\e}  { {\vec e} }
    \newcommand {\c}  { {\vec c} }
    \renewcommand {\d}  { {\vec d} }
    \renewcommand {\u}  { {\vec u} }
    \newcommand{\I}{\vec{I}}


Direct Current Resistivity
**************************

`SimPEG.electromagnetics.static.resistivity` and `SimPEG.electromagnetics.static.induced_polarization` uses SimPEG as the framework for the forward and inverse
direct current (DC) resistivity and induced polarization (IP) geophysical problems.


DC resistivity survey
=====================

Electrical resistivity of subsurface materials is measured by causing an
electrical current to flow in the earth between one pair of electrodes while
the voltage across a second pair of electrodes is measured. The result is an
"apparent" resistivity which is a value representing the weighted average
resistivity over a volume of the earth. Variations in this measurement are
caused by variations in the soil, rock, and pore fluid electrical resistivity.
Surveys require contact with the ground, so they can be labour intensive.
Results are sometimes interpreted directly, but more commonly, 1D, 2D or 3D
models are estimated using inversion procedures (`GPG
<http://gpg.geosci.xyz>`_).


Background
==========

As direct current (DC) implies, in DC resistivity survey, we assume steady-state. We consider Maxwell's equations in steady state as

.. math::

    \curl \frac{1}{\mu} \vec{b} - \j = \j_s \\

    \curl \e = 0

Then by taking \\(\\div\\) of the first equation, we have

.. math::

     - \div\j = q \\


where

.. math::

    \div \j_s = q = I(\delta(\vec{r}-\vec{r}_{s+})-\delta(\vec{r}-\vec{r}_{s-}))

Since \\(\\curl \\e = 0\\), we have

.. math::

    \e = \grad \phi

And by Ohm's law, we have

.. math::

    \j = \sigma \grad \phi

Finally, we can compute the solution of the system:

.. math::

    - \div\j = q

    \j = \sigma \grad \phi

    \frac{\partial \phi}{\partial r}\Big|_{\partial \Omega_{BC}} = 0


Discretization
==============

By using finite volume method (FVM), we discretize our system as

.. math::

    -\Div \bj = \bq

    \diag(\Acf^{T}\sigma^{-1}) \bj = \Grad \bphi

Here boundary condtions are embedded in the discrete differential operators. With some linear algebra we have

.. math::

    \bA\bphi  = -\bq

where

.. math::

    \bA = \Div (\diag(\Acf^{T}\sigma^{-1}))^{-1} \Grad

By solving this linear equation, we can compute the solution of \\(\\phi\\). Based on this discretization, we derive sensitivity in discretized space. Sensitivity matrix can be in general can be written as

.. math ::

    \bJ = -\bP\bA^{-1}\bG

where

.. math ::

    \bP: \text{Projection}

    \bJ = \bP\frac{\partial \phi}{\partial \bm}

Here \\(\\bm\\) indicates model parameters in discretized space.


.. image:: /content/examples/04-dcip/images/sphx_glr_plot_dc_analytic_001.png
    :target: /content/examples/04-dcip/plot_dc_analytic.html
    :align: center


API for DC codes
================

Simulation
----------

.. automodule:: SimPEG.electromagnetics.static.resistivity.simulation
    :show-inheritance:
    :members:
    :undoc-members:


.. automodule:: SimPEG.electromagnetics.static.resistivity.simulation_2d
    :show-inheritance:
    :members:
    :undoc-members:


.. automodule:: SimPEG.electromagnetics.static.resistivity.simulation_1d
    :show-inheritance:
    :members:
    :undoc-members:

Survey
------

.. automodule:: SimPEG.electromagnetics.static.resistivity.survey
    :show-inheritance:
    :members:
    :undoc-members:

.. automodule:: SimPEG.electromagnetics.static.resistivity.sources
    :show-inheritance:
    :members:
    :undoc-members:

.. automodule:: SimPEG.electromagnetics.static.resistivity.receivers
    :show-inheritance:
    :members:
    :undoc-members:

Fields
------

.. automodule:: SimPEG.electromagnetics.static.resistivity.fields
    :show-inheritance:
    :members:
    :undoc-members:

.. automodule:: SimPEG.electromagnetics.static.resistivity.fields_2d
    :show-inheritance:
    :members:
    :undoc-members:

Utils
-----

.. automodule:: SimPEG.electromagnetics.static.utils
    :show-inheritance:
    :members:
    :undoc-members:

.. automodule:: SimPEG.electromagnetics.static.resistivity.utils
    :show-inheritance:
    :members:
    :undoc-members:

.. automodule:: SimPEG.electromagnetics.static.resistivity.boundary_utils
    :show-inheritance:
    :members:
    :undoc-members:
