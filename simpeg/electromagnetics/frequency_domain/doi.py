from ... import maps
from ...utils import refine_1d_layer, doi_1d_layer_CA2012
from .simulation_1d import Simulation1DLayered


def doi_fdem_1d_layer_CA2012(t, m, survey, std_data, threshold=0.8):
    r"""Compute FDEM DOI for a half-space layered model.

    It is a wrapper for the subroutine
    ``simpeg.utils.doi.doi_1d_layer_CA2012``.

    Parameters
    ----------
    t : array-like, shape (n_layers,)
        The thicknesses of each layer in meters.

    m : array-like, shape (n_layers,)
        Conductivity model with values for each layer
        thickness.

    survey : .frequency_domain.survey.Survey
        The frequency-domain EM survey.

    std_data : array-like or scalar
        The standard deviation(s) of data points. If
        array-like, it should have shape (n_data,).

    threshold : float, default=0.8
         The threshold fraction (of maximum normalized
         aggregated sensitivity) required for a layer to
         be considered well-resolved.

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

    t_star, m_star = refine_1d_layer(t, m, 100)
    conductivity_map = maps.IdentityMap(nP=len(m_star))

    simulation_HM = Simulation1DLayered(
        survey=survey,
        thicknesses=t_star,
        sigmaMap=conductivity_map,
    )

    J = simulation_HM.getJ(m_star).copy()
    J = J["ds"]

    doi, Sj_star, S = doi_1d_layer_CA2012(J, t_star, m_star, std_data, threshold)

    return doi, t_star, m_star, Sj_star, S
