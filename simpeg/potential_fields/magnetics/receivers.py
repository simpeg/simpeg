from ... import survey
from ...utils import validate_string


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
        - "tmi_x"--> x-derivative of the total magnetic intensity data
        - "tmi_y"--> y-derivative of the total magnetic intensity data
        - "tmi_z"--> z-derivative of the total magnetic intensity data

    Notes
    -----
    If predicting amplitude data, you must set include 'bx', 'by', and 'bz' here, and
    set `is_amplitude_data` in the `magnetics.Simulation3DIntegral` to `True`.

    """

    def __init__(self, locations, components="tmi", **kwargs):
        super().__init__(locations, **kwargs)

        if isinstance(components, str):
            components = [components]
        for component in components:
            validate_string(
                "component",
                component,
                [
                    "bxx",
                    "bxy",
                    "bxz",
                    "byy",
                    "byz",
                    "bzz",
                    "bx",
                    "by",
                    "bz",
                    "tmi",
                    "tmi_x",
                    "tmi_y",
                    "tmi_z",
                ],
            )
        self.components = components

    @property
    def nD(self):
        """Number of data.

        Returns
        -------
        int
            Number of data for the receiver (locations X components)
        """
        return self.locations.shape[0] * len(self.components)
