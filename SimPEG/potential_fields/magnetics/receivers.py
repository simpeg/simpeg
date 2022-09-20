from ... import survey


class Point(survey.BaseRx):
    """
    Magnetic point receiver class for integral formulation

    :param numpy.ndarray locs: receiver locations index (ie. :code:`np.c_[ind_1, ind_2, ...]`)
    :param string components: receiver component (string or list)
         "bxx", "bxy", "bxz", "byy",
         "byz", "bzz", "bx", "by", "bz", "tmi" [default]
    """

    def __init__(self, locations, components="tmi", **kwargs):

        super(survey.BaseRx, self).__init__(locations=locations, **kwargs)

        if isinstance(components, str):
            components = [components]
        for component in components:
            if component not in [
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
            ]:
                raise ValueError(
                    f"{component} not recognized. Must be "
                    "'bxx', 'bxy', 'bxz', 'byy',"
                    "'byz', 'bzz', 'bx', 'by', 'bz', or 'tmi'. "
                )
        self.components = components

    @property
    def nD(self):
        return self.locations.shape[0] * len(self.components)
