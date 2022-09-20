from ... import survey


class Point(survey.BaseRx):
    """
    Gravity point receiver class for integral formulation

    Parameters
    ----------
    locations : (n_loc, dim) array_like
    components : str or list of str

    :param numpy.ndarray locations: receiver locations index (ie. :code:`np.c_[ind_1, ind_2, ...]`)
    :param string component: receiver component
         "gx", "gy", "gz", "gxx", "gxy", "gxz",
         "gyy", "gyz", "gzz", "guv"
    """

    def __init__(self, locations, components="gz", **kwargs):
        super(survey.BaseRx, self).__init__(locations=locations, **kwargs)
        if isinstance(components, str):
            components = [components]

        for component in components:
            if component not in [
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
            ]:
                raise ValueError(
                    f"{component} not recognized, must be one of "
                    "'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz'"
                    "'gyy', 'gyz', 'gzz', or 'guv'"
                )
        self.components = components

    @property
    def nD(self):
        return self.locations.shape[0] * len(self.components)
