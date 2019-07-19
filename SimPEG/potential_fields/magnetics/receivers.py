import properties

from ... import survey


class point_receiver(survey.BaseRx):
    """
    Magnetic point receiver class for integral formulation

    :param numpy.ndarray locs: receiver locations index (ie. :code:`np.c_[ind_1, ind_2, ...]`)
    :param string component: receiver component
         "dbx_dx", "dbx_dy", "dbx_dz", "dby_dy",
         "dby_dz", "dbz_dz", "bx", "by", "bz", "tmi" [default]
    """

    receiver_index = None

    # component = properties.StringChoice(
    #     "Must be a magnetic component of the type",
    #     ["dbx_dx", "dbx_dy", "dbx_dz", "dby_dy",
    #      "dby_dz", "dbz_dz", "bx", "by", "bz", "tmi"
    #      ]
    # )

    component = [
        "dbx_dx", "dbx_dy", "dbx_dz", "dby_dy",
        "dby_dz", "dbz_dz", "bx", "by", "bz", "tmi"
    ]

    def __init__(self, component=["tmi"], **kwargs):

        self.component = component

        # super(point_receiver, self).__init__(location_index, **kwargs)

    def nD(self):

        if self.receiver_index is not None:

            return self.location_index.shape[0]

        elif self.locations is not None:

            return self.locations.shape[0]

        else:

            return None

    def receiver_index(self):

        return self.receiver_index
