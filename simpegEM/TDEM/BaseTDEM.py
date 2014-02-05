from SimPEG import Utils
from SimPEG.Data import BaseData


class DataTDEM1D(BaseData):
    """
        docstring for DataTDEM1D
    """

    txLoc = None #: txLoc
    txType = None #: txType
    rxLoc = None #: rxLoc
    rxType = None #: rxType
    timeCh = None #: timeCh

    def __init__(self, **kwargs):
        BaseData.__init__(self, **kwargs)
        Utils.setKwargs(self, **kwargs)
