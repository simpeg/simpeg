.. _api_TDEM:


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
    \newcommand{\St}{{\mathbf \Sigma_\tau}}
    \newcommand{\T}{{\mathbf T}}
    \newcommand{\Tt}{{\mathbf T_\tau}}
    \newcommand{\diag}[1]{\,{\sf diag}\left( #1 \right)}
    \newcommand{\M}{{\mathbf M}}
    \newcommand{\MfMui}{{\M^f_{\mu^{-1}}}}
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


Time Domain Electromagnetics
****************************

The quasi-static time domain Maxwell's equations are given by

.. math::
    \curl \vec{e} + \frac{\partial \vec{b}}{\partial t} = \vec{s}_m \\
    \curl \vec{h} - \vec{j} = \vec{s}_e

where:

- :math:`\vec{e}` is the electric field
- :math:`\vec{b}` is the magnetic flux
- :math:`\vec{h}` is the magnetic field
- :math:`\vec{j}` is the current density
- :math:`\vec{s}_m` is the magnetic source term
- :math:`\vec{s}_e` is the electric source term

.. note::

    .. raw:: html

       <div class="col-md-2" align="center">
          <a href="https://github.com/simpeg/tutorials/"><i class="fa fa-wrench fa-4x" aria-hidden="true"></i></a>
       </div>

    These docs are under construction. More to come soon!

.. Sensitivity Calculation
.. -----------------------

.. .. math::

..     \begin{align}
..         \dcurl \e^{(t+1)} + \frac{\b^{(t+1)} - \b^{(t)}}{\delta t} = 0 \\
..         \dcurl^\top \MfMui \b^{(t+1)} - \MeSig \e^{(t+1)} = \Me \j_s^{(t+1)}
..     \end{align}

.. Using Gauss-Newton to solve the inverse problem requires the ability to calculate the product of the
.. Jacobian and a vector, as well as the transpose of the Jacobian times a vector.
.. The above system can be rewritten as:

.. .. math::

..     \begin{align}
..         \mathbf{A} \u^{(t+1)} + \mathbf{B} \u^{(t)}= \s^{(t+1)}
..     \end{align}

.. where

.. .. math::

..     \begin{align}
..         \mathbf{A} =
..         \left[
..             \begin{array}{cc}
..                 \frac{1}{\delta t} \MfMui & \MfMui\dcurl \\
..                 \dcurl^\top \MfMui & -\MeSig
..             \end{array}
..         \right] \\
..         \mathbf{B} =
..         \left[
..             \begin{array}{cc}
..                 -\frac{1}{\delta t} \MfMui & 0 \\
..                 0 & 0
..             \end{array}
..         \right] \\
..         \u^{(k)} = \left[
..         \begin{array}{c}
..             \b^{(k)}\\
..             \e^{(k)}
..         \end{array}
..         \right] \\
..         \s^{(k)} = \left[
..         \begin{array}{c}
..             0\\
..             \Me \j^{(k)}_s
..         \end{array}
..         \right]
..     \end{align}

.. .. note::

..     Here we have multiplied through by \\(\\MfMui\\) to make A and B symmetric!

.. The entire time dependent system can be written in a single matrix expression

.. .. math::

..     \begin{align}
..         \hat{\mathbf{A}} \hat{u} = \hat{s}
..     \end{align}

.. where

.. .. math::

..     \begin{align}
..         \mathbf{\hat{A}} = \left[
..         \begin{array}{cccc}
..             A & 0 & & \\
..             B & A & & \\
..               & \ddots & \ddots & \\
..               & & B & A
..         \end{array}
..         \right] \\
..         \hat{u} = \left[
..             \begin{array}{c}
..                 \u^{(1)} \\
..                 \u^{(2)} \\
..                 \vdots \\
..                 \u^{(N)}
..             \end{array} \right]\\
..         \hat{s} = \left[
..             \begin{array}{c}
..                 \s^{(1)} - \mathbf{B} \u^{(0)} \\
..                 \s^{(2)} \\
..                 \vdots \\
..                 \s^{(N)}
..             \end{array}
..         \right]
..     \end{align}

.. For the fields \\(\\u\\), the measured data is given by

.. .. math::

..     \begin{align}
..         \vec{d} = \mathbf{Q} \u
..     \end{align}

.. The sensitivity matrix **J** is then defined as

.. .. math::

..     \begin{align}
..         \mathbf{J} = \mathbf{Q} \frac{\partial \u}{\partial \sigma}
..     \end{align}


.. Defining the function \\(\\c(m,\\u)\\) to be

.. .. math::

..     \begin{align}
..         \vec{c}(m,\u) = \hat{\mathbf{A}} \vec{u} - \vec{q} = \vec{0}
..     \end{align}

.. then

.. .. math::

..     \begin{align}
..         \frac{\partial \vec{c}}{\partial m} \partial m
..         + \frac{\partial \vec{c}}{\partial \u} \partial \vec{u} = 0
..     \end{align}

.. or

.. .. math::

..     \begin{align}
..         \frac{\partial \vec{u}}{\partial m} = -\left(\frac{\partial \vec{c}}{\partial \u} \right)^{-1} \frac{\partial \vec{c}}{\partial m}
..     \end{align}


.. Differentiating, we find that

.. .. math::

..     \begin{align}
..         \frac{\partial \vec{c}}{\partial \hat{u}} = \hat{\mathbf{A}}
..     \end{align}

.. and

.. .. math::

..     \begin{align}
..         \frac{\partial \vec{c}}{\partial \sigma} = \mathbf{G}_\sigma =
..         \left[
..             \begin{array}{c}
..                 g_\sigma^{(1)}\\
..                 g_\sigma^{(2)}\\
..                 \vdots \\
..                 g_\sigma^{(N)}
..             \end{array}
..         \right]
..     \end{align}

.. with

.. .. math::

..     \begin{align}
..         g_\sigma^{(n)} =
..         \left[
..             \begin{array}{c}
..                 \mathbf{0} \\
..                 - \diag{\e^{(n)}} \Ace \diag{\vec{V}}
..             \end{array}
..         \right]
..     \end{align}


.. Implementing **J** times a vector
.. ---------------------------------

.. Multiplying **J** onto a vector can be broken into three steps


.. * Compute \\(\\vec{p} = \\mathbf{G}m\\)
.. * Solve \\(\\hat{\\mathbf{A}} \\vec{y} = \\vec{p}\\)
.. * Compute \\(\\vec{w} = -\\mathbf{Q} \\vec{y}\\)

.. .. math::

..     \begin{align}
..         \vec{p}^{(n)} = \left[
..             \begin{array}{c}
..                 \vec{p}_b^{(n)} \\
..                 \vec{p}_e^{(n)}
..             \end{array}
..         \right] \\
..         \vec{p}_b^{(n)} = 0 \\
..         \vec{p}_e^{(n)} = - \diag{\e^{(n)}} \Ace \diag{V} m
..     \end{align}


.. For all time steps:

.. .. math::

..     \begin{align}
..         \frac{1}{\delta t} \MfMui\vec{y}_{b}^{(t+1)} + \MfMui\dcurl \vec{y}_{e}^{(t+1)}
..         - \frac{1}{\delta t} \MfMui \vec{y}_{b}^{(t)}
..         = \vec{p}_b^{(t+1)} \\
..         \dcurl^\top \MfMui \vec{y}_b^{(t+1)} - \MeSig \vec{y}_e^{(t+1)} = \vec{p}_e^{(t+1)}
..     \end{align}

.. and

.. .. math::

..     \begin{align}
..         \left( \MfMui \dcurl \MeSig^{-1} \dcurl^\top \MfMui + \frac{1}{\delta t} \MfMui \right) \vec{y}_{b}^{(t+1)} =
..         \frac{1}{\delta t} \MfMui \vec{y}_b^{(t)}
..         + \MfMui \dcurl \MeSig^{-1} \vec{p}_e^{(t+1)} + \vec{p}_b^{(t+1)} \\
..         \vec{y}_e^{(t+1)} = \MeSig^{-1} \dcurl^\top \MfMui \vec{y}_b^{(t+1)} - \MeSig^{-1} \vec{p}_e^{(t+1)}
..     \end{align}

.. .. note::

..     For the first time step, \\\(t=0\\\), the term: \\\(\\frac{1}{\\delta t} \\MfMui \\vec{y}_b^{(0)}\\\) is zero.




.. Implementing **J** transpose times a vector
.. -------------------------------------------

.. Multiplying \\(\\mathbf{J}^\\top\\) onto a vector can be broken into three steps


.. * Compute \\(\\vec{p} = \\mathbf{Q}^\\top \\vec{v}\\)
.. * Solve \\(\\hat{\\mathbf{A}}^\\top \\vec{y} = \\vec{p}\\)
.. * Compute \\(\\vec{w} = -\\mathbf{G}^\\top y\\)


.. .. math::

..     \mathbf{\hat{A}}^\top = \left[
..         \begin{array}{cccc}
..             A & B & & \\
..               & \ddots & \ddots & \\
..               & & A & B \\
..               & & 0 & A
..         \end{array}
..     \right]

.. For the all time-steps (going backwards in time):


.. .. math::

..     A \vec{y}^{(t)} + B \vec{y}^{(t+1)} = \vec{p}^{(t)}


.. .. math::

..     \begin{align}
..         \frac{1}{\delta t} \MfMui\vec{y}_{b}^{(t)} + \MfMui\dcurl \vec{y}_{e}^{(t)}
..         - \frac{1}{\delta t} \MfMui \vec{y}_{b}^{(t+1)}
..         = \vec{p}_b^{(t)} \\
..         \dcurl^\top \MfMui \vec{y}_b^{(t)} - \MeSig \vec{y}_e^{(t)} = \vec{p}_e^{(t)}
..     \end{align}

.. and

.. .. math::

..     \begin{align}
..         \left( \MfMui \dcurl \MeSig^{-1} \dcurl^\top \MfMui + \frac{1}{\delta t} \MfMui \right) \vec{y}_{b}^{(t)} =
..         \frac{1}{\delta t} \MfMui \vec{y}_b^{(t+1)}
..         + \MfMui \dcurl \MeSig^{-1} \vec{p}_e^{(t)} + \vec{p}_b^{(t)} \\
..         \vec{y}_e^{(t)} = \MeSig^{-1} \dcurl^\top \MfMui \vec{y}_b^{(t)} - \MeSig^{-1} \vec{p}_e^{(t)}
..     \end{align}


.. .. note::

..     For the last time step, \\\(t=N\\\), the term: \\\(\\frac{1}{\\delta t} \\MfMui \\vec{y}_b^{(N+1)}\\\) is zero.



TDEM Problem
============

.. automodule:: SimPEG.EM.TDEM.ProblemTDEM
    :show-inheritance:
    :members:
    :undoc-members:


Fields
======

.. automodule:: SimPEG.EM.TDEM.FieldsTDEM
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:


Sources
=======

.. automodule:: SimPEG.EM.TDEM.SrcTDEM
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:

Survey
======

.. automodule:: SimPEG.EM.TDEM.SurveyTDEM
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:


.. Base Classes
.. ============

.. .. automodule:: SimPEG.EM.TDEM.BaseTDEM
..     :show-inheritance:
..     :members:
..     :undoc-members:
..     :inherited-members:

