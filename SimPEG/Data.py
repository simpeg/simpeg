import properties


class Data(object):
    """
    Fancy data storage by Src and Rx
    """

    dobs = properties.Array(
        "observed data",
        shape=('*',)
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

    srcList = properties.List(
        "source list",
        Source.BaseSrc
    )

    uid = properties.Uuid("unique ID for the data")

    def __init__(self, v=None):
        self._dataDict = {}
        for src in self.srcList:
            self._dataDict[src] = {}
        if v is not None:
            self.fromvec(v)

    @properties
    def nD(self):
        pass

    def _ensureCorrectKey(self, key):
        if type(key) is tuple:
            if len(key) is not 2:
                raise KeyError('Key must be [Src, Rx]')
            if key[0] not in self.survey.srcList:
                raise KeyError('Src Key must be a source in the survey.')
            if key[1] not in key[0].rxList:
                raise KeyError('Rx Key must be a receiver for the source.')
            return key
        elif isinstance(key, self.survey.srcPair):
            if key not in self.survey.srcList:
                raise KeyError('Key must be a source in the survey.')
            return key, None
        else:
            raise KeyError('Key must be [Src] or [Src,Rx]')

    def __setitem__(self, key, value):
        src, rx = self._ensureCorrectKey(key)
        assert rx is not None, 'set data using [Src, Rx]'
        assert isinstance(value, np.ndarray), 'value must by ndarray'
        assert value.size == rx.nD, (
            "value must have the same number of data as the source."
        )
        self._dataDict[src][rx] = Utils.mkvc(value)

    def __getitem__(self, key):
        src, rx = self._ensureCorrectKey(key)
        if rx is not None:
            if rx not in self._dataDict[src]:
                raise Exception('Data for receiver has not yet been set.')
            return self._dataDict[src][rx]

        return np.concatenate([self[src, rx] for rx in src.rxList])

    def tovec(self):
        return np.concatenate([self[src] for src in self.survey.srcList])

    def fromvec(self, v):
        v = Utils.mkvc(v)
        assert v.size == self.survey.nD, (
            'v must have the correct number of data.'
        )
        indBot, indTop = 0, 0
        for src in self.survey.srcList:
            for rx in src.rxList:
                indTop += rx.nD
                self[src, rx] = v[indBot:indTop]
                indBot += rx.nD
