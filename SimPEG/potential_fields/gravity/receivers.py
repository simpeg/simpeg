import numpy as np
from ... import survey


class Point(survey.BaseRx):
    """Point receiver class for gravity simulations

    The **Point** receiver defines the locations and the components of the gravity
    field that are simulated at each location. The length of the resulting data
    vector is *n_loc X n_comp*, and is organized by location then component.

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
    """

    def __init__(self, locations, components="gz", **kwargs):

        super(Point, self).__init__(locations, **kwargs)

        n_locations = self.locations.shape[0]

        if isinstance(components, str):
            components = [components]

        component_dict = {}
        for component in components:
            component_dict[component] = np.ones(n_locations, dtype="bool")

        assert np.all(
            [
                component
                in ["gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "guv"]
                for component in list(component_dict.keys())
            ]
        ), (
            "Components {0!s} not known. Components must be in "
            "'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz'"
            "'gyy', 'gyz', 'gzz', 'guv'"
            "Arbitrary orientations have not yet been "
            "implemented.".format(component)
        )
        self.components = component_dict

    def nD(self):
        """Number of data

        Returns
        -------
        int
            The number of data
        """

        if self.receiver_index is not None:
            return self.location_index.shape[0]
        elif self.locations is not None:
            return self.locations.shape[0]
        else:
            return None

    def receiver_index(self):
        """Receiver index

        Returns
        -------
        np.ndarray of int
            Receiver indices
        """
        return self.receiver_index
