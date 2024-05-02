from ... import survey
from ...utils import validate_string


class Point(survey.BaseRx):
    """Point receiver class for gravity simulations

    The **Point** receiver defines the locations and the components of the gravity
    field that are simulated at each location. The length of the resulting data
    vector is *n_loc X n_comp*, and is organized by location then component.

    .. important::

        Density model is assumed to be in g/cc.

    .. important::

        Acceleration components ("gx", "gy", "gz") are returned in mgal
        (:math:`10^{-5} m/s^2`).

    .. important::

        Gradient components ("gxx", "gyy", "gzz", "gxy", "gxz", "gyz") are
        returned in Eotvos (:math:`10^{-9} s^{-2}`).

    Parameters
    ----------
    locations: (n_loc, 3) numpy.ndarray
        Receiver locations
    components: str or list of str
        Data component(s) measured at each receiver location. Use a ``str`` for a
        single component or a ``list`` of ``str`` if multiple components are simulated
        at each location. Component choices are:

        - "gx"   --> x-component of the gravity field
        - "gy"   --> y-component of the gravity field
        - "gz"   --> z-component of the gravity field (DEFAULT)
        - "gxx"  --> x-derivative of the x-component
        - "gxy"  --> y-derivative of the x-component (and visa versa)
        - "gxz"  --> z-derivative of the x-component (and visa versa)
        - "gyy"  --> y-derivative of the y-component
        - "gyz"  --> z-derivative of the y-component (and visa versa)
        - "gzz"  --> z-derivative of the z-component
        - "guv"  --> UV component

    See also
    --------
    simpeg.potential_fields.gravity.Simulation3DIntegral
    """

    def __init__(self, locations, components="gz", **kwargs):
        super(Point, self).__init__(locations, **kwargs)

        if isinstance(components, str):
            components = [components]

        for component in components:
            validate_string(
                "component",
                component,
                [
                    "gx",
                    "gy",
                    "gz",
                    "gxx",
                    "gxy",
                    "gxz",
                    "gyy",
                    "gyz",
                    "gzz",
                    "guv",
                ],
            )
        self.components = components

    @property
    def nD(self):
        """Number of data

        Returns
        -------
        int
            The number of data
        """
        return self.locations.shape[0] * len(self.components)
