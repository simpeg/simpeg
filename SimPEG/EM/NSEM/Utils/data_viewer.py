# Interactive NSEM map plot
import numpy as np
import matplotlib.pyplot as plt

from .. import RxNSEM
import properties

from plotUtils import ApperentResPhsStationPlot, TipperAmplitudeStationPlot

class NSEM_data_viewer(properties.HasProperties):
    """
    An interactive SimPEG NSEM Data viewer.

    Generates a clickble location map of the data, plotting data curves
    in a separate window.

    **Required**
    :param SimPEG.EM.NSEM.Data data: Data object, needs to have assigned
        standard_deviation and floor

    **Optional**
    :param dict data_dict: A dictionary of other NSEM Data objects
    :param string backend: Flag to control the backend used in matplotlib

    """

    _need_setup = True

    def __init__(self, data, data_dict=None, backend='qt'):
        """
        Initialize an object

        """
        # Set data
        self._data = data
        self._data_dict = data_dict

        # Set the default component
        if data_dict is not None:
            self.dict_comp = data_dict.keys()[0]
        else:
            self._dict_comp = None

        # Open the location figure
        self.location_fig, self._location_ax = self._data.map_data_locations(
            picker=5
        )
        # Seth the onClick event function
        self.location_fig.canvas.mpl_connect('pick_event', self._onClick)

        self.text = self._location_ax.text(
            0.05, 1.05, 'Selected: none',
            transform=self._location_ax.transAxes, va='top'
        )

        # Add the selection highlight
        self._selecetd_point = None # Should use properties
        self._selected_loc = self._location_ax.plot(
            [], [], 'o', ms=12, alpha=0.4, color='yellow',
            visible=False, zorder=5
        )[0]

        # Set up the plot containers
        self.station_figs = []
        unique_rx = np.unique(
            np.array([rx.__class__ for src in self._data.survey.srcList for rx in src.rxList]))
        if RxNSEM.Point_impedance3D in unique_rx:
            self.station_figs.append(ApperentResPhsStationPlot())
        if RxNSEM.Point_tipper3D in unique_rx:
            self.station_figs.append(TipperAmplitudeStationPlot())

        self.freqency_figs = []



    @property
    def dict_comp(self):
        return self._dict_comp

    @dict_comp.setter
    def dict_comp(self, value):
        if self._data_dict is None:
            raise Exception('No data dictionary connected to the object.')
        else:
            if value in self._data_dict.iterkeys():
                self._dict_comp = value
            else:
                raise Exception(
                    '{} is not a key in the connected dictionary.'.format(value))


    def view(self):
        """
        Excute the viewer and show the matplotlib window
        """
        self.location_fig.show()
        plt.draw()
        plt.show()

    def _onClick(self, event):
        """
        The click function that is executed when a location in
        plot is clicked the left-mouse button.
        """


        # Get the event artist
        artist = event.artist
        ind = event.ind
        # Get the location
        x,y = artist.get_xdata()[ind][0],artist.get_ydata()[ind][0]
        self._selecetd_point = np.array([[x,y]])
        # Check if the figure exists

        if self._need_setup:
            self._setup_station_figs()
            self._need_setup = False
        else:
            self._clear_station_figs()
        # Draw
        draw_list = [self._data]
        if self._data_dict is not None:
            draw_list.append(self._data_dict[self.dict_comp])
        self._draw_station_figs(draw_list, self._selecetd_point)

        # Highlight the point
        self._selected_loc.set_visible(True)
        self._selected_loc.set_data(x,y)

        # Set the text
        self.text.set_text('Selected: {:.1f} - {:.1f}'.format(x,y))
        self.location_fig.canvas.draw()

        for sta_fig in self.station_figs:
            sta_fig.fig.canvas.draw()
            sta_fig.fig.show()


    def _setup_station_figs(self):
        """
        Setup a station data plot figures


        """
        if len(self.station_figs) > 0:
            for sta_fig in self.station_figs:
                sta_fig.setup()

    def _draw_station_figs(self, data_list, location):
        """
        Draw the station figure

        """

        if len(self.station_figs) > 0:
            for sta_fig in self.station_figs:
                sta_fig.draw(data_list, location)

    def _clear_station_figs(self):
        """
        Clear all the axes of the station figure
        """

        if len(self.station_figs) > 0:
            for sta_fig in self.station_figs:
                sta_fig.clear_axes()

