from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG import Utils
from SimPEG.EM.Static.SIP.SrcSIP import BaseSrc
from SimPEG.EM.Static.SIP.RxSIP import BaseRx
import uuid


class Survey(BaseEMSurvey):
    rxPair = BaseRx
    srcPair = BaseSrc
    times = None

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        BaseEMSurvey.__init__(self, srcList, **kwargs)
        self.getUniqueTimes()

    def getUniqueTimes(self):
        time_rx = []
        for src in self.srcList:
            for rx in src.rxList:
                time_rx.append(rx.times)
        self.times = np.unique(np.hstack(time_rx))

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        return self.prob.forward(m, f=f)


class Data(SimPEG.Survey.Data):
    """Fancy data storage by Src and Rx"""

    def __init__(self, survey, v=None):
        self.uid = str(uuid.uuid4())
        self.survey = survey
        self._dataDict = {}
        for src in self.survey.srcList:
            self._dataDict[src] = {}
            for rx in src.rxList:
                self._dataDict[src][rx] = {}

        if v is not None:
            self.fromvec(v)

    def _ensureCorrectKey(self, key):
        if type(key) is tuple:
            if len(key) is not 3:
                raise KeyError('Key must be [Src, Rx, tInd]')
            if key[0] not in self.survey.srcList:
                raise KeyError('Src Key must be a source in the survey.')
            if key[1] not in key[0].rxList:
                raise KeyError('Rx Key must be a receiver for the source.')
            return key
        elif isinstance(key, self.survey.srcPair):
            if key not in self.survey.srcList:
                raise KeyError('Key must be a source in the survey.')
            return key, None, None
        else:
            raise KeyError('Key must be [Src] or [Src,Rx] or [Src, Rx, tInd]')

    def __setitem__(self, key, value):
        src, rx, t = self._ensureCorrectKey(key)
        assert rx is not None, 'set data using [Src, Rx]'
        assert isinstance(value, np.ndarray), 'value must by ndarray'
        assert value.size == rx.nD, "value must have the same number of data as the source."
        self._dataDict[src][rx][t] = Utils.mkvc(value)

    def __getitem__(self, key):
        src, rx, t = self._ensureCorrectKey(key)
        if rx is not None:
            if rx not in self._dataDict[src]:
                raise Exception('Data for receiver has not yet been set.')
            return self._dataDict[src][rx][t]

        return np.concatenate([self[src,rx, t] for rx in src.rxList])

    def tovec(self):
        val = []
        for src in self.survey.srcList:
            for rx in src.rxList:
                for t in rx.times:
                    val.append(self[src, rx, t])
        return np.concatenate(val)

    def fromvec(self, v):
        v = Utils.mkvc(v)
        assert v.size == self.survey.nD, 'v must have the correct number of data.'
        indBot, indTop = 0, 0
        for src in self.survey.srcList:
            for rx in src.rxList:
                for t in rx.times:
                    indTop += rx.nRx
                    self[src, rx, t] = v[indBot:indTop]
                    indBot += rx.nRx
