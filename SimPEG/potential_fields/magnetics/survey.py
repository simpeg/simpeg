import numpy as np

from ...survey import BaseSurvey


class MagneticSurvey(BaseSurvey):
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

        return self.source_field.receiver_list[0].locations.shape[0]

    @property
    def receiver_locations(self):
        return self.source_field.receiver_list[0].locations

    @property
    def nD(self):
        return len(self.receiver_locations) * len(self.components)

    @property
    def components(self):
        return self.source_field.receiver_list[0].components

    @property
    def vnD(self):
        """Vector number of data"""

        if getattr(self, '_vnD', None) is None:
            self._vnD = []
            for receiver in self.source_field.receiver_list:

                for component in list(receiver.components.keys()):

                    # If non-empty than logcial for empty entries
                    self._vnD.append(int(receiver.components[component].sum()))

            print(self._vnD)
            self._vnD = np.asarray(self._vnD)
        return self._vnD

# make this look like it lives in the below module
MagneticSurvey.__module__ = 'SimPEG.potential_fields.magnetics'
