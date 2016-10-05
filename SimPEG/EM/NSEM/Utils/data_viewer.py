# Interactive NSEM map plot


class NSEM_data_viewer(object):

    def __init__(self, data, pred_data_dict, backend='qt'):
        """
        An interactive SimPEG NSEM Data viewer


        """
        # Set data
        self._data = data
        # Open the location figure
        self.location_fig, self._location_ax = self._data.plot_data_locations()
        # Seth the onClick event function
        self.location_fig.canvas.mpl_connect('pick_event', self._onClick)
        # Add the selection highlight
        self._selected_loc = self._location_ax.plot(
            [], [], 'o', ms=12, alpha=0.4, color='yellow', visible=False, zorder=5)[0]
        # Set up the panels
        self.station_fig = None
        self.freqency_fig = None

        self.text = self._location_ax.text(0.05, 1.05, 'Selected: none',transform=self.mapax.transAxes, va='top')


    def _onClick(self, event):

        # Get the event artist
        artist = event.artist
        ind = event.ind

        x,y = artist.get_xdata()[ind][0],artist.get_ydata()[ind][0]
        if self.stafig is not None:
            # Clear the figure
            plt.close(self.stafig)
            del(self.stafig)
            self.stafig = None

        # Open the station figure
        self.stafig ,ax = MTpFunc.plotImpAppRes(self._data,np.array([x,y]))
        self._selected_loc.set_visible(True)
        self._selected_loc.set_data(x,y)
        self.text.set_text('Selected: {:.1f} - {:.1f}'.format(x,y))
        self.mapfig.canvas.draw()
        self.stafig.show()

