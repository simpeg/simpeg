import numpy as np
from SimPEG import Survey


class StraightRaySurvey(Survey.LinearSurvey):
    def __init__(self, txList):
        self.txList = txList

    @property
    def nD(self):
        n = 0
        for tx in self.txList:
            n += np.sum([rx.nD for rx in tx.rxList])
        return n

    def projectFields(self, u):
        return u

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.subplot(111)
        for tx in self.txList:
            ax.plot(tx.loc[0], tx.loc[1], 'ws', ms=8)

            for rx in tx.rxList:
                for loc_i in range(rx.locs.shape[0]):
                    ax.plot(rx.locs[loc_i, 0],rx.locs[loc_i, 1], 'w^', ms=8)
                    ax.plot(np.r_[tx.loc[0], rx.locs[loc_i, 0]], np.r_[tx.loc[1], rx.locs[loc_i, 1]], 'w-', lw=0.5, alpha=0.8)
