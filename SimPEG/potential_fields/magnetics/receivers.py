import numpy as np
from ... import survey


class Point(survey.BaseRx):
    """Magnetic point receiver class for integral formulation

    Parameters
    ----------
    locations : (n, 3) numpy.ndarray
        Receiver locations.
    components : str or list of str, default: 'tmi'
        Use a ``str`` for a single component or a ``list`` of ``str`` if multiple
        components are simulated at each location. Component choices are:

        - "tmi"  --> total magnetic intensity data (DEFAULT)
        - "bx"   --> x-component of the magnetic field
        - "by"   --> y-component of the magnetic field
        - "bz"   --> z-component of the magnetic field
        - "bxx"  --> x-derivative of the x-component
        - "bxy"  --> y-derivative of the x-component (and visa versa)
        - "bxz"  --> z-derivative of the x-component (and visa versa)
        - "byy"  --> y-derivative of the y-component
        - "byz"  --> z-derivative of the y-component (and visa versa)
        - "bzz"  --> z-derivative of the z-component

    Notes
    -----
    If predicting amplitude data, you must set include 'bx', 'by', and 'bz' here, and
    set `is_amplitude_data` in the `magnetics.Simulation3DIntegral` to `True`.

    """

    def __init__(self, locations, components="tmi", **kwargs):

        super().__init__(locations, **kwargs)

        n_locations = self.locations.shape[0]

        if isinstance(components, str):
            components = [components]

        component_dict = {}
        for component in components:
            component_dict[component] = np.ones(n_locations, dtype="bool")

        assert np.all(
            [
                component
                in ["bxx", "bxy", "bxz", "byy", "byz", "bzz", "bx", "by", "bz", "tmi"]
                for component in list(component_dict.keys())
            ]
        ), (
            "Components {0!s} not known. Components must be in "
            "'bxx', 'bxy', 'bxz', 'byy',"
            "'byz', 'bzz', 'bx', 'by', 'bz', 'tmi'. "
            "Arbitrary orientations have not yet been "
            "implemented.".format(component)
        )
        self.components = component_dict

    def nD(self):
        """Number of data

        Returns
        -------
        int
            Number of data for the receiver (locations X components)
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
