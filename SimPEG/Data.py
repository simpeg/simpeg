import properties

from .NewSurvey import BaseSurvey


class Data(properties.HasProperties):
    """
    Fancy data storage by Src and Rx
    """

    dobs = properties.Array(
        "observed data",
        shape=('*',),
        required=True
    )

    standard_deviation = properties.Float(
        "standard deviation of the data",
        min=0.0
    )

    noise_floor = properties.Float(
        "noise floor of the data",
        min=0.0
    )

    uncertainty = properties.Array(
        "data uncertainties",
        shape=('*',)
    )

    survey = properties.Instance(
        "a SimPEG survey object",
        BaseSurvey
    )

    uid = properties.Uuid("unique ID for the data")

    _data_dict = properties.Instance(
        "data dictionary so data can be accessed by [src, rx]. "
        "This stores the indicies in the data vector corresponding "
        "to each source receiver pair",
        dict,
    )

    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)

    @properties.observer('survey')
    def _create_data_dict(self, change):
        survey = change['value']
        self._data_dict = {}

        # create an empty dict associated with each source
        for src in survey.srcList:
            self._data_dict[src] = {}

        # loop over sources and find the associated data indices
        indBot, indTop = 0, 0
        for src in self.survey.srcList:
            for rx in src.rxList:
                indTop += rx.nD
                self[src, rx] = np.arange(indBot, indTop)
                indBot += rx.nD

    @property
    def nD(self):
        return len(dobs)

    def _ensureCorrectKey(self, key):
        if type(key) is tuplsee:
            if len(key) is not 2:
                raise KeyError('Key must be [Src, Rx]')
            if key[0] not in self.survey.srcList:
                raise KeyError('Src Key must be a source in the survey.')
            if key[1] not in key[0].rxList:
                raise KeyError('Rx Key must be a receiver for the source.')
            return key

        # TODO: I think this can go
        elif isinstance(key, self.survey.srcPair):
            if key not in self.survey.srcList:
                raise KeyError('Key must be a source in the survey.')
            return key, None

        else:
            raise KeyError('Key must be [Src] or [Src,Rx]')

    def __getitem__(self, key):
        src, rx = self._ensureCorrectKey(key)
        if rx is not None:
            if rx not in self._dataDict[src]:
                raise Exception('Data for receiver has not yet been set.')
            return self.dobs[self._dataDict[src][rx]]

        return np.concatenate([
            self.obs[self._dataDict[src, rx]] for rx in src.rxList
        ])
