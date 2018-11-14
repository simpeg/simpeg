from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import SimPEG
from SimPEG.EM.Base import BaseEMSurvey
from SimPEG import Utils
from . import RxSIP as Rx
from . import SrcSIP as Src
from SimPEG.EM.Static import DC
import uuid


class Survey(BaseEMSurvey):
    rxPair = Rx.BaseRx
    srcPair = Src.BaseSrc
    times = None
    _pred = None
    n_pulse = 2
    T = 8.

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

    @property
    def n_locations(self):
        return int(self.nD/self.times.size)

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        if f is None:
            f = self.prob.fields(m)
        return self._pred
        # return self.prob.forward(m, f=f)


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
        assert value.size == rx.nD, ("value must have the same number of data as the source.")
        self._dataDict[src][rx][t] = Utils.mkvc(value)

    def __getitem__(self, key):
        src, rx, t = self._ensureCorrectKey(key)
        if rx is not None:
            if rx not in self._dataDict[src]:
                raise Exception('Data for receiver has not yet been set.')
            return self._dataDict[src][rx][t]

        return np.concatenate(
            [self[src, rx, t] for rx in src.rxList]
        )

    def tovec(self):
        val = []
        for src in self.survey.srcList:
            for rx in src.rxList:
                for t in rx.times:
                    val.append(self[src, rx, t])
        return np.concatenate(val)

    def fromvec(self, v):
        v = Utils.mkvc(v)
        assert v.size == self.survey.nD, (
            'v must have the correct number of data.'
        )
        indBot, indTop = 0, 0

        for t in self.survey.times:
            for src in self.survey.srcList:
                for rx in src.rxList:
                    indTop += rx.nRx
                    self[src, rx, t] = v[indBot:indTop]
                    indBot += rx.nRx


def from_dc_to_sip_survey(survey_dc, times):
    """
    Generate sip survey from dc survey
    """
    srcList = survey_dc.srcList

    srcList_sip = []
    for src in srcList:
        rxList_sip = []
        for rx in src.rxList:
            if isinstance(rx, DC.Rx.Pole_ky) or isinstance(rx, DC.Rx.Pole):
                rx_sip = Rx.Pole(rx.locs, times=times)
            elif isinstance(rx, DC.Rx.Dipole_ky) or isinstance(rx, DC.Rx.Dipole):
                rx_sip = Rx.Dipole(rx.locs[0], rx.locs[1], times=times)
            else:
                print (rx)
                raise NotImplementedError()
            rxList_sip.append(rx_sip)

        if isinstance(src, DC.Src.Pole):
            src_sip = Src.Pole(
                rxList_sip, src.loc
            )
        elif isinstance(src, DC.Src.Dipole):
            src_sip = Src.Dipole(
                rxList_sip, src.loc[0], src.loc[1]
            )
        else:
            raise NotImplementedError()
        srcList_sip.append(src_sip)

    survey_sip = Survey(srcList_sip)

    return survey_sip
