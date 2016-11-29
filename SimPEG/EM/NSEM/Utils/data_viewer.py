# Interactive NSEM map plot
import numpy as np
import matplotlib.pyplot as plt

class NSEM_data_viewer(object):

    def __init__(self, data, data_dict=None, backend='qt'):
        """
        An interactive SimPEG NSEM Data viewer


        """
        # Set data
        self._data = data
        self._data_dict = data_dict
        # Open the location figure
        self.location_fig, self._location_ax = self._data.plot_data_locations(
            picker=5
        )
        # Seth the onClick event function
        self.location_fig.canvas.mpl_connect('pick_event', self._onClick)
        # Add the selection highlight
        self._selecetd_point = None # Should use properties
        self._selected_loc = self._location_ax.plot(
            [], [], 'o', ms=12, alpha=0.4, color='yellow',
            visible=False, zorder=5
        )[0]
        # Set up the panels
        self.station_fig = None
        self.freqency_fig = None

        self.text = self._location_ax.text(
            0.05, 1.05, 'Selected: none',
            transform=self._location_ax.transAxes, va='top'
        )

    def view(self):
        self.location_fig.show()
        plt.draw()
        plt.show()

    def _onClick(self, event):

        # Get the event artist
        artist = event.artist
        ind = event.ind
        # Get the location
        x,y = artist.get_xdata()[ind][0],artist.get_ydata()[ind][0]
        self._selecetd_point = np.array([[x,y]])
        # Check if the figure exists
        if self.station_fig is None:
            self._setup_station_fig()
        else:
            self._clear_station_fig()
        # Draw
        self._draw_station_fig()

        # Highlight the point
        self._selected_loc.set_visible(True)
        self._selected_loc.set_data(x,y)
        # Set the text
        self.text.set_text('Selected: {:.1f} - {:.1f}'.format(x,y))
        self.location_fig.canvas.draw()
        self.station_fig.canvas.draw()
        self.station_fig.show()
        # plt.show()

    def _setup_station_fig(self):
        """
        Setup a station data plot figure.
        Hard coded for app_res/phs and imp amp/phs
        """
        self.station_fig, axT = plt.subplots(2, 2, sharex=True)
        axes = axT.ravel()
        self.station_fig.set_size_inches((13.5, 7.0))

        # Have to deal with axes
        # Set log
        for ax in axes.ravel():
            ax.set_xscale('log')

        axes[0].invert_xaxis()
        axes[0].set_yscale('log')
        axes[2].set_yscale('log')
        # Set labels
        axes[2].set_xlabel('Frequency [Hz]')
        axes[3].set_xlabel('Frequency [Hz]')
        axes[0].set_ylabel('Apperent resistivity [Ohm m]')
        axes[1].set_ylabel('Apperent phase [degrees]')
        axes[1].set_ylim(-180,180)
        axes[2].set_ylabel('Impedance amplitude [V/A]')
        axes[3].set_ylim(-180,180)
        axes[3].set_ylabel('Impedance angle [degrees]')

    def _draw_station_fig(self):
        """
        Draw the station figure

        """
        axes = self.station_fig.get_axes()

        # Apparent resistivity
        self._data.plot_station_component(
            self._selecetd_point, 'xy', 'app_res', ax=axes[0], color='b',
            marker='s', label='AppRes xy')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'xy', 'app_res', ax=axes[0], color='b')
        self._data.plot_station_component(
            self._selecetd_point, 'yx', 'app_res', ax=axes[0], color='r',
            marker='s', label='AppRes yx')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'yx', 'app_res', ax=axes[0], color='r')
        # Apparent phase
        self._data.plot_station_component(
            self._selecetd_point, 'xy', 'phase', ax=axes[1], color='b',
            marker='s', label='AppRes xy')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'xy', 'phase', ax=axes[1], color='b')
        self._data.plot_station_component(
            self._selecetd_point, 'yx', 'phase', ax=axes[1], color='r',
            marker='s', label='AppRes yx')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'yx', 'phase', ax=axes[1], color='r')
        # Impedamce amplitude
        self._data.plot_station_component(
            self._selecetd_point, 'xx', 'amplitude', ax=axes[2], color='g',
            marker='s', label='Amplitude xx')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'xx', 'amplitude', ax=axes[2], color='g')
        self._data.plot_station_component(
            self._selecetd_point, 'xy', 'amplitude', ax=axes[2], color='b',
            marker='s', label='Amplitude xy')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'xy', 'amplitude', ax=axes[2], color='b')
        self._data.plot_station_component(
            self._selecetd_point, 'yx', 'amplitude', ax=axes[2], color='r',
            marker='s', label='Amplitude yx')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'yx', 'amplitude', ax=axes[2], color='r')
        self._data.plot_station_component(
            self._selecetd_point, 'yy', 'amplitude', ax=axes[2], color='y',
            marker='s', label='Amplitude yy')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'yy', 'amplitude', ax=axes[2], color='y')
        # Impedance phase
        self._data.plot_station_component(
            self._selecetd_point, 'xx', 'phase', ax=axes[3], color='g',
            marker='s', label='Amplitude xx')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'xx', 'phase', ax=axes[3], color='g')
        self._data.plot_station_component(
            self._selecetd_point, 'xy', 'phase', ax=axes[3], color='b',
            marker='s', label='Amplitude xy')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'xy', 'phase', ax=axes[3], color='b')
        self._data.plot_station_component(
            self._selecetd_point, 'yx', 'phase', ax=axes[3], color='r',
            marker='s', label='Amplitude yx')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'yx', 'phase', ax=axes[3], color='r')
        self._data.plot_station_component(
            self._selecetd_point, 'yy', 'phase', ax=axes[3], color='y',
            marker='s', label='Amplitude yy')
        self._data.plot_station_errorbars(
            self._selecetd_point, 'yy', 'phase', ax=axes[3], color='y')


    def _clear_station_fig(self):
        """
        Clear all the axes of the station figure
        """

        for ax in self.station_fig.axes:
            while len(ax.lines) > 0:
                for line in ax.lines:
                    ax.lines.remove(line)

            while len(ax.collections) > 0:
                for item in ax.collections:
                    item.remove()
            # del ax.collections[:]
