import numpy as np

from ....data import Data as BaseData


class Data(BaseData):
    """
    Data class for spectral induced polarization data
    """

    @property
    def index_dictionary(self):
        """
        Dictionary of data indices by sources and receivers. To set data using
        survey parameters:

        .. code::
            data = Data(survey)
            for src in survey.source_list:
                for rx in src.receiver_list:
                    for t in rx.times:
                        index = data.index_dictionary[src][rx][t]
                        data.dobs[index] = datum

        """
        if getattr(self, "_index_dictionary", None) is None:
            if self.survey is None:
                raise Exception(
                    "To set or get values by source-receiver pairs, a survey must "
                    "first be set. `data.survey = survey`"
                )

            # create an empty dict
            self._index_dictionary = {}

            # create an empty dict associated with each source
            for src in self.survey.source_list:
                self._index_dictionary[src] = {}

                for rx in src.receiver_list:
                    self._index_dictionary[src][rx] = {}

            # loop over sources and find the associated data indices
            indBot, indTop = 0, 0
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    for t in rx.times:
                        indTop += rx.nD
                        self._index_dictionary[src][rx][t] = np.arange(indBot, indTop)
                        indBot += rx.nD

        return self._index_dictionary

    ##########################
    # Methods
    ##########################

    def __setitem__(self, key, value):
        index = self.index_dictionary[key[0]][key[1]][key[2]]
        self.dobs[index] = mkvc(value)

    def __getitem__(self, key):
        index = self.index_dictionary[key[0]][key[1]][key[2]]
        return self.dobs[index]
