import numpy as np
from . import sdiag


def refine_1d_layer(t, c, M):
    r"""Subdiscretize the model.

    This method is used to increase the number of layers for
    improving the precision of the DOI estimation.

    Parameters
    ----------
    t : numpy.ndarray
        Input layers ticknesses.
    c : numpy.ndarray
        Input layers physical properties.
    M : Integer
        The number of layers in the refined model.

    Returns
    -------
    t_out : numpy.ndarray
        Refined layers thicknesses.
    c_out : numpy.ndarray
        Refined layers physical properties.
    """
    t_ = np.r_[t, t[-1]]
    z_ = np.r_[np.cumsum(t_)]
    z_max = np.max(z_)
    t_out = np.linspace(0, z_max, M)
    c_out = np.zeros(M)
    a = 0.0
    n = len(c)
    for i in range(n):
        b = z_[i]
        c_out[(t_out >= a) & (t_out <= b)] = c[i]
        a = b

    return np.diff(t_out), c_out


def doi_1d_layer_CA2012(J, t, std_data, threshold=0.8):
    r"""Compute the depth of investigation for a layered half space.

    Compute the cumulative sensitivity and determine the
    Depth of Investigation (DOI) as in Christiansen et al. (2012).

    First we define the error normalized sensitivity :math:`s_j` for each
    model parameter :math:`j`

    .. math::
        s_j = \sum_{i=1}^N\frac{G_{ij}}{\Delta d_i}.

    We also define the error and thickness normalized sensitivity as

    .. math::
        s^*_j = \frac{\sum_{i=1}^N\frac{G_{ij}}{\Delta d_i}}{t_j}.

    The cumulated sensitivities :math:`S_j` is given by

    .. math::
        S_j = \sum_{i=M,-1}^j{s_i}.

    References

    Vest Christiansen, A., & Auken, E. (2012). A global measure for
    depth of investigation. Geophysics, 77(4), WB171-WB177.

    Parameters
    ----------
    J : array-like, shape (n_data, n_layers)
        Sensitivity (Jacobian) matrix.

    thicknesses : array-like, shape (n_layers,)
        The thickness of each layer in meters.

    std_data : array-like or scalar
        The standard deviation(s) of data points. If array-like, it should
        have shape (n_data,).

    threshold : float, default=0.8
         The threshold fraction (of maximum normalized aggregated
         sensitivity) required for a layer to be considered well-resolved.

    Returns
    -------
    doi : float
        The Depth of Investigation, defined as the center depth
        (in meters) of the deepest layer with :math:`S_j > threshold`.
    Sj_star : numpy.ndarray, shape (n_layers,)
        Error and thickness normalized sensitivity for each model parameter.
    S : numpy.ndarray, shape (n_layers,)
        Cumulative (from bottom to top) sensitivity.
    """
    J_n = sdiag(1 / std_data) * J
    Sj = abs(J_n).sum(axis=0)
    Sj_star = Sj[:-1] / t
    S = np.flip(np.cumsum(Sj_star[::-1]))
    active = S - threshold > 0.0
    depth = np.cumsum(t)
    doi = depth[active].max()

    return doi, Sj_star
