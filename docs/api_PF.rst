.. _api_PF:


.. math::

    \renewcommand{\div}{\nabla\cdot\,}
    \newcommand{\grad}{\vec \nabla}
    \newcommand{\curl}{{\vec \nabla}\times\,}
    \newcommand {\J}{{\vec J}}
    \renewcommand{\H}{{\vec H}}
    \newcommand {\E}{{\vec E}}
    \newcommand{\dcurl}{{\mathbf C}}
    \newcommand{\dgrad}{{\mathbf G}}
    \newcommand{\Acf}{{\mathbf A_c^f}}
    \newcommand{\Ace}{{\mathbf A_c^e}}
    \renewcommand{\S}{{\mathbf \Sigma}}
    \renewcommand{\Div}{{\mathbf {Div}}}
    \newcommand{\St}{{\mathbf \Sigma_\tau}}
    \newcommand{\T}{{\mathbf T}}
    \newcommand{\Tt}{{\mathbf T_\tau}}
    \newcommand{\diag}{\mathbf{diag}}
    \newcommand{\M}{{\mathbf M}}
    \newcommand{\MfMui}{{\M^f_{\mu^{-1}}}}    
    \newcommand{\dMfMuI}{{d_m (\M^f_{\mu^{-1}})^{-1}}}
    \newcommand{\MeSig}{{\M^e_\sigma}}
    \newcommand{\MeSigInf}{{\M^e_{\sigma_\infty}}}
    \newcommand{\MeSigO}{{\M^e_{\sigma_0}}}
    \newcommand{\Me}{{\M^e}}
    \newcommand{\Mes}[1]{{\M^e_{#1}}}
    \newcommand{\Mee}{{\M^e_e}}
    \newcommand{\Mej}{{\M^e_j}}
    \newcommand{\BigO}[1]{\mathcal{O}\bigl(#1\bigr)}
    \newcommand{\bE}{\mathbf{E}}
    \newcommand{\bH}{\mathbf{H}}
    \newcommand{\B}{\vec{B}}
    \newcommand{\D}{\vec{D}}
    \renewcommand{\H}{\vec{H}}
    \newcommand{\s}{\vec{s}}
    \newcommand{\bfJ}{\bf{J}}
    \newcommand{\vecm}{\vec m}
    \renewcommand{\Re}{\mathsf{Re}}
    \renewcommand{\Im}{\mathsf{Im}}
    \renewcommand {\j}  { {\vec j} }
    \newcommand {\h}  { {\vec h} }
    \renewcommand {\b}  { {\vec b} }
    \newcommand {\e}  { {\vec e} }
    \newcommand {\c}  { {\vec c} }
    \renewcommand {\d}  { {\vec d} }
    \renewcommand {\u}  { {\vec u} }
    \newcommand{\I}{\vec{I}}

Magnetics
*********


The geomagnetic field can be ranked as the longest studied of all the geophysical properties of the earth. In addition, magnetic survey, has been used broadly in diverse realm e.g., minining, oil and gas industry and envrionmental engineering. Although, this geophysical application is quite common in geoscience; however, we do not have modular, well-documented and well-tested open-source codes, which perform forward and inverse problems of magnetic survey. Therefore, here we are going to build up magnetic forward and inverse modeling code based on  two common methodologies for forward problem - differential equation and integral equation approaches. \

First, we start with some backgrounds of magnetics, e.g., Maxwell's equations. Based on that secondly, we use differential equation approach to solve forward problem with seocondary field formulation. In order to discretzie our system here, we use finite volume approach with weak formulation. Third, we solve inverse problem through Gauss-Newton method.

Backgrounds
===========
Maxwell's equations for static case with out current source can be written as

.. math::

    \nabla U = \frac{1}{\mu}\vec{B} \\

    \nabla \cdot \vec{B} = 0

where \\(\\vec{B}\\) is magnetic flux (\\(\T\\)) and \\(\U\\) is magnetic potential and \\(\\mu\\) is permeability. Since we do not have any source term in above equations, boundary condition is going to be the driving force of our system as given below

.. math ::

    (\vec{B}\cdot{\vec{n}})_{\partial\Omega} = B_{BC}

where \\(\\vec{n}\\) means the unit normal vector on the boundary surface (\\(\\partial \\Omega\\)). By using seocondary field formulation we can rewrite above equations as

.. math ::

    \frac{1}{\mu}\vec{B}_s = (\frac{1}{\mu}_0-\frac{1}{\mu})\vec{B}_0+\nabla\phi_s

    \nabla \cdot \vec{B}_s = 0

    (\vec{B}_s\cdot{\vec{n}})_{\partial\Omega} = B_{sBC}

where \\(\\vec{B}_s\\) is the secondary magnetic flux and \\(\\vec{B}_0\\) is the backgroud or primary magnetic flux. In practice, we consider our earth field, which we can get from International Geomagnetic Reference Field (IGRF) by specifying the time and location, as  \\(\\vec{B}_0\\). And based on this background fields, we compute secondary fields (\\(\\vec{B}_s\\)). Now we introduce the susceptibility as

.. math ::

    \chi = \frac{\mu}{\mu_0} - 1 \\

    \mu = \mu_0(1+\chi)

Since most materials in the earth  have lower permeability than \\(\\mu_0\\), usually \\(\\chi\\) is greater than 0.

.. note ::

    Actually, this is an asumption, which means we are not sure exactly this is true, although we are sure, it is very rare that we can encounter those materials. Anyway, practical range of the susceptibility is \\(0 < \\chi < 1 \\).

Since we compute secondary field based on the earth field, which can be different from different locations in the world, we can expect different anomalous responses in different locations in the earth. For instance, assume we have two susceptible spheres, which are exactly same. However, anomalous responses in Seoul and Vancouver are going to be different.


.. plot :: examples/Mag_back_1.py

Since we can measure total fields ( \\(\\vec{B}\\)), and usually have reasonably accurate earth field (\\(\\vec{B}_0\\)), we can compute anmalous fields, \\(\\vec{B}_s\\) from our observed data. If you want to download earth magnetic fields at specific location see this website (`noaa <http://www.ngdc.noaa.gov/geomag-web/>`_).

What is our data?
-----------------

In applied geophysics, which means in practice, it is common to refer to measurements as "the magnetic anomlay" and we can consider this as our observed data. For further descriptions in `GPG <http://www.eos.ubc.ca/courses/eosc350/content/>`_ materials for magnetic survey. Now we have the simple relation ship between "the magnetic anomlay" and the total field as 

.. math ::

	\triangle\vec{B} = |\vec{B}_0-\vec{B}_s|-|\vec{B}_0| \approx |\vec{B}_s|cos \theta

where \\(\\theta\\) is the angle between total and anomalous fields. Equivalently, we can use the vector dot product to show that the anomalous field is aproximately equal to the projection of that field onto the direction of the inducing field. Using this approach we would write

.. math ::

	\triangle\vec{B} = |\vec{B}_s|cos \theta = |\vec{B}_o||\vec{B}_s|cos \theta = \vec{B}_o \cdot \vec{B}_s

This is important because, in practice we usually use a total field magnetometer (like a proton precession or optically pumped sensor), which can measure only that part of the anomalous field which is in the direction of the earth's main field.

Sphere in a whole space
-----------------------


Forward problem
===============



Inverse problem
===============


Mag Differential eq. approach
*****************************

.. autoclass:: simpegPF.Magnetics.MagneticsDiffSecondary
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:


Mag Integral eq. approach
*************************

Mag analytic solutions
**********************

.. automodule:: simpegPF.MagAnalytics
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:
