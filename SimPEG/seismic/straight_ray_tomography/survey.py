import numpy as np
from ...survey import BaseSurvey


class StraightRaySurvey(BaseSurvey):
    """Straight ray tomography survey    """

    @property
    def nD(self):
        """Number of data

        Returns
        -------
        int
            Number of data
        """
        n = 0
        for tx in self.source_list:
            n += np.sum([rx.nD for rx in tx.receiver_list])
        return n

    def projectFields(self, u):
        """Returns fields

        Parameters
        ----------
        u : numpy.ndarray
            Fields

        Returns
        -------
        numpy.ndarray
            Returns the input argument *u*
        """
        return u

    def plot(self, ax=None):
        """Plot straight ray topography

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes object

        Returns
        -------
        matplotlib.Axes
            The plot on the axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplot(111)
        for tx in self.source_list:
            ax.plot(tx.location[0], tx.location[1], "ws", ms=8)

            for rx in tx.receiver_list:
                for loc_i in range(rx.locations.shape[0]):
                    ax.plot(rx.locations[loc_i, 0], rx.locations[loc_i, 1], "w^", ms=8)
                    ax.plot(
                        np.r_[tx.location[0], rx.locations[loc_i, 0]],
                        np.r_[tx.location[1], rx.locations[loc_i, 1]],
                        "w-",
                        lw=0.5,
                        alpha=0.8,
                    )
