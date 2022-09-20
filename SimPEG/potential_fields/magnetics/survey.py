import numpy as np
from ...survey import BaseSurvey


class Survey(BaseSurvey):
    """Base Magnetics Survey"""

    # source_field = properties.Instance(
    #     "The inducing field source for the survey",
    #     properties.Instance("A SimPEG source", SourceField),
    #     default=SourceField
    # )

    def __init__(self, source_field, **kwargs):
        self.source_field = source_field
        BaseSurvey.__init__(self, **kwargs)

    def eval(self, fields):
        return fields

    @property
    def nRx(self):
        return sum(rx.locations.shape[0] for rx in self.source_field.receiver_list)

    @property
    def receiver_locations(self):
        return np.concatenate([rx.locations for rx in self.source_field.receiver_list])

    @property
    def nD(self):
        return sum(rx.nD for rx in self.source_field.receiver_list)

    @property
    def components(self):
        comps = []
        for rx in self.source_field.receiver_list:
            comps += rx.components
        return comps

    def _location_component_iterator(self):
        for rx in self.source_field.receiver_list:
            for loc in rx.locations:
                yield loc, rx.components

    @property
    def vnD(self):
        """Vector number of data"""

        if getattr(self, "_vnD", None) is None:
            self._vnD = []
            for receiver in self.source_field.receiver_list:
                for component in receiver.components:

                    # If non-empty than logcial for empty entries
                    self._vnD.append(len(receiver.components))

            print(self._vnD)
            self._vnD = np.asarray(self._vnD)
        return self._vnD


# make this look like it lives in the below module
Survey.__module__ = "SimPEG.potential_fields.magnetics"
