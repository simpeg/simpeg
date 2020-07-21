.. _api_DataMisfit:


Data Misfit
***********

The data misfit using an l_2 norm is:

.. math::

    \mu_\text{data} = {1\over 2}\left| \mathbf{W}_d (\mathbf{d}_\text{pred} - \mathbf{d}_\text{obs}) \right|_2^2

If the field, u, is provided, the calculation of the data is fast:

.. math::

    \mathbf{d}_\text{pred} = \mathbf{Pu(m)}

    \mathbf{R} = \mathbf{W}_d (\mathbf{d}_\text{pred} - \mathbf{d}_\text{obs})

Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
u is the field of interest; d_obs is the observed data; and \\\(\\mathbf{W}_d\\\) is the weighting matrix.

The derivative of this, with respect to the model, is:

.. math::

    \frac{\partial \mu_\text{data}}{\partial \mathbf{m}} = \mathbf{J}^\top \mathbf{W}_d \mathbf{R}

The second derivative is:

.. math::

    \frac{\partial^2 \mu_\text{data}}{\partial^2 \mathbf{m}} = \mathbf{J}^\top \mathbf{W}_d \mathbf{W}_d \mathbf{J}


The API
=======

.. autoclass:: SimPEG.data_misfit.BaseDataMisfit
    :members:
    :undoc-members:

Common Data Misfits
===================

l2 norm
-------

.. autoclass:: SimPEG.data_misfit.L2DataMisfit
    :members:
    :undoc-members:
