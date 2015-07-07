.. _api_FDEM:

.. math::
    \renewcommand{\div}{\nabla\cdot\,}
    \newcommand{\grad}{\vec \nabla}
    \newcommand{\curl}{{\vec \nabla}\times\,}


Frequency Domain Electromagnetics
*********************************

Electromagnetic (EM) geophysical methods are used in a variety of applications from resource exploration, including for hydrocarbons and minerals, to environmental applications, such as groundwater monitoring. The primary physical property of interest in EM is electrical conductivity, which describes the ease with which electric current flows through a material.


Background
==========

Electromagnetic phenomena are governed by Maxwell's equations. They describe the behavior of EM fields and fluxes. Electromagnetic theory for geophysical applications by Ward and Hohmann (1988) is a highly recommended resource on this topic.

Fourier Transform Convention
----------------------------
In order to examine Maxwell's equations in the frequency domain, we must first define our choice of harmonic time-dependence by choosing a Fourier transform convention. We use the \\(e^{i \\omega t} \\) convention, so we define our Fourier Transform pair as

.. math ::
	F(\omega) = \int_{-\infty}^{\infty} f(t) e^{- i \omega t} dt \\

	f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} F(\omega) e^{i \omega t} d \omega

where \\(\\omega\\) is angular frequency, \\(t\\) is time, \\(F(\\omega)\\) is the function defined in the frequency domain and \\(f(t)\\) is the function defined in the time domain.


Maxwell's Equations
===================
In the frequency domain, Maxwell's equations are given by

.. math ::
	\curl \vec{E} = - i \omega \vec{B} \\

	\curl \vec{H} = \vec{J} + i \omega \vec{D} + \vec{S} \\

	\div \vec{B} = 0 \\

	\div \vec{D} = \rho_f

where:

	- \\(\\vec{E}\\) : electric field (\\(V/m\\))
	- \\(\\vec{H}\\) : magnetic field (\\(A/m\\))
	- \\(\\vec{B}\\) : magnetic flux density (\\(Wb/m^2\\))
	- \\(\\vec{D}\\) : electric displacement / electric flux density (\\(C/m^2\\))
	- \\(\\vec{J}\\) : electric current density (\\(A/m^2\\))
	- \\(\\rho_f\\) : free charge density
	
The source term is \\(\\vec{S}\\)


Constitutive Relations
----------------------
The fields and fluxes are related through the constitutive relations. At each frequency, they are given by

.. math ::
	\vec{J} = \sigma \vec{E} \\

	\vec{B} = \mu \vec{H} \\

	\vec{D} = \varepsilon \vec{E}

where:

- \\(\\sigma\\) : electrical conductivity \\(S/m\\)
- \\(\\mu\\) : magnetic permeability \\(H/m\\)
- \\(\\varepsilon\\) : dielectric permittivity \\(F/m\\)

\\(\\sigma\\), \\(\\mu\\), \\(\\varepsilon\\) are physical properties which depend on the material. \\(\\sigma\\) describes how easily electric current passes through a material, \\(\\mu\\) describes how easily a material is magnetized, and \\(\\varepsilon\\) describes how easily a material is electrically polarized. In most geophysical applications of EM, \\(\\sigma\\) is the the primary physical property of interest, and \\(\\mu\\), \\(\\varepsilon\\) are assumed to have their free-space values \\(\\mu_0 = 4\\pi \\times 10^{-7} H/m \\), \\(\\varepsilon_0 = 8.85 \\times 10^{-12} F/m\\)


Quasi-static Approximation
--------------------------
For the frequency range typical of most geophysical surveys, the contribution of the electric displacement is negligible compared to the electric current density. In this case, we use the Quasi-static approximation and assume that this term can be neglected, giving

.. math ::
	\nabla \times \vec{E} = -i \omega \vec{B} \\
	\nabla \times \vec{H} = \vec{J} + \vec{S}


Implementation in simpegEM
==========================
We consider two formulations in simpegEM, both first-order and both in terms of one field and one flux. We allow for the definition of magnetic and electric sources (see for example: Ward and Hohmann, starting on page 144). The E-B formulation is in terms of the electric field and the magnetic flux:

.. math ::
	\nabla \times \vec{E} + i \omega \vec{B} = \vec{S}_m \\
	\nabla \times \mu^{-1} \vec{B} - \sigma \vec{E} = \vec{S}_e

The H-J formulation is in terms of the current density and the magnetic field:

.. math ::
	\nabla \times \sigma^{-1} \vec{J} + i \omega \mu \vec{H} = \vec{S}_m \\
	\nabla \times \vec{H} - \vec{J} = \vec{S}_e


Discretizing
------------
For both formulations, we use a finite volume discretization 
and discretize fields on cell edges, fluxes on cell faces and 
physical properties in cell centers. This is particularly 
important when using symmetry to reduce the dimensionality of a problem
(for instance on a 2D CylMesh, there are \\(r\\), \\(z\\) faces and \\(\\theta\\) edges) 

.. figure:: ./images/finitevolrealestate.png
	:align: center
	:scale: 60 %

For the two formulations, the discretization of the physical properties, fields and fluxes are summarized below. 

.. figure:: ./images/ebjhdiscretizations.png
	:align: center
	:scale: 60 %

Note that resistivity is the inverse of conductivity, \\(\\rho = \\sigma^{-1}\\). 


E-B Formulation:
****************

.. math ::
	\mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\
	\mathbf{C^T} \mathbf{M^f_{\mu^{-1}}} \mathbf{b} - \mathbf{M^e_\sigma} \mathbf{e} = \mathbf{s_e}

H-J Formulation:
****************

.. math ::
	\mathbf{C^T} \mathbf{M^f_\rho} \mathbf{j} + i \omega \mathbf{M^e_\mu} \mathbf{h} = \mathbf{s_m} \\
	\mathbf{C} \mathbf{h} - \mathbf{j} = \mathbf{s_e}


.. Forward Problem
.. ===============

.. Inverse Problem
.. ===============

API
===
.. automodule:: simpegEM.FDEM.FDEM
    :show-inheritance:
    :members:
    :undoc-members:


FDEM Survey
-----------

.. automodule:: simpegEM.FDEM.SurveyFDEM
    :show-inheritance:
    :members:
    :undoc-members:
