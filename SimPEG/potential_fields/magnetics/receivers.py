import properties
import numpy as np
from ... import survey
from ...utils.code_utils import deprecate_class


class Point(survey.BaseRx):
    """
    Magnetic point receiver class for integral formulation

    :param numpy.ndarray locs: receiver locations index (ie. :code:`np.c_[ind_1, ind_2, ...]`)
    :param string component: receiver component
         "bxx", "bxy", "bxz", "byy",
         "byz", "bzz", "bx", "by", "bz", "tmi" [default]
    """

    def __init__(self, locations, components="tmi", **kwargs):

        super(survey.BaseRx, self).__init__(locations=locations, **kwargs)

        n_locations = locations.shape[0]

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

        if self.receiver_index is not None:

            return self.location_index.shape[0]

        elif self.locations is not None:

            return self.locations.shape[0]

        else:

            return None

    def receiver_index(self):

        return self.receiver_index


@deprecate_class(removal_version="0.15.0")
class RxObs(Point):
    pass
