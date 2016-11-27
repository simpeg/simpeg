from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from SimPEG.EM.Utils.EMUtils import mu_0, omega

# Define the default component dictionaries
default_comp_dict = {
    'xx':{'color': 'green', 'label': 'Imp_xx', 'marker': '_', 'ls': 'None'},
    'xy':{'color': 'blue', 'label': 'Imp_xy', 'marker': '_', 'ls': 'None'},
    'yx':{'color': 'red', 'label': 'Imp_yx', 'marker': '_', 'ls': 'None'},
    'yy':{'color': 'yellow', 'label': 'Imp_yy', 'marker': '_', 'ls': 'None'},
    'zx':{'color': 'brown', 'label': 'Tip_zx', 'marker': '_', 'ls': 'None'},
    'zy':{'color': 'purple', 'label': 'Tip_zy', 'marker': '_', 'ls': 'None'},
}

_default_comp_dict_input = {
    'xx':None,
    'xy':None,
    'yx':None,
    'yy':None,
    'zx':None,
    'zy':None
}

_imp_comps = ['xx', 'xy', 'yx', 'yy']

def _validate_kwargs(input_dict, compare_dict):
    # Set the default plot kwargs
    for key, val in compare_dict.iteritems():
        # Make sure they aren't already assigned
        if input_dict is None:
            input_dict = compare_dict.copy()
        else:
            if key not in input_dict:
                input_dict[key] = val
    # Return
    return input_dict

class DataNSEM_plot_functions(object):
    """
    Class container for properties and methods for
    plotting of NSEM data.

    """

    def plot_app_res(self, location,
                     components=['xy', 'yx'], ax=None, errorbars=False,
                     comp_plot_dict=default_comp_dict
                    ):
        """
        Plot apperent resistivity curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param components: List of the components to plot.
            Default = ['xy','yx']
        :type components: list
        :param ax: The ax object to add the,  , Default: None
        :type ax: :class:`axes <matplotlib.axes>`
        :param errorbars: Controls if errorbars are plotted
            Default = True
        :type errorbars: boolean
        :param comp_plot_dict: Dictionary with additional kwargs
            for matplotlib.plot
        :type comp_plot_dict: dict

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.invert_xaxis()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Apperent resistivity [Ohm m]')
        else:
            fig = ax.get_figure()

        for comp in components:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], default_comp_dict[comp])

            self.plot_station_component(
                location, comp, 'app_res',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], default_comp_dict[comp])

                self.plot_station_errorbars(
                    location, comp, 'app_res',
                    ax=ax, **eb_kwargs)

        return ax

    def plot_app_phs(self, location,
                     components=['xy', 'yx'], ax=None, errorbars=False,
                     comp_plot_dict=default_comp_dict
                    ):
        """
        Plot apperent resistivity curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param components: List of the components to plot.
            Default = ['xy','yx']
        :type components: list
        :param ax: The ax object to add the,  , Default: None
        :type ax: :class:`axes <matplotlib.axes>`
        :param errorbars: Controls if errorbars are plotted
            Default = True
        :type errorbars: boolean
        :param comp_plot_dict: Dictionary with additional kwargs
            for matplotlib.plot settings
        :type comp_plot_dict: dict
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.invert_xaxis()
            ax.set_xscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Phase angle [Degrees]')
        else:
            fig = ax.get_figure()

        for comp in components:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], default_comp_dict[comp])

            self.plot_station_component(
                location, comp, 'phase',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], default_comp_dict[comp])

                self.plot_station_errorbars(
                    location, comp, 'phase',
                    ax=ax, **eb_kwargs)

        return ax

    def plot_imp_amp(self, location,
                     components=['xy', 'yx'], ax=None, errorbars=False,
                     comp_plot_dict=default_comp_dict
                    ):
        """
        Plot impedance amplitude curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param components: List of the components to plot.
            Default = ['xy','yx']
        :type components: list
        :param ax: The ax object to add the,  , Default: None
        :type ax: :class:`axes <matplotlib.axes>`
        :param errorbars: Controls if errorbars are plotted
            Default = True
        :type errorbars: boolean
        :param comp_plot_dict: Dictionary with additional kwargs
            for matplotlib.plot
        :type comp_plot_dict: dict
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.invert_xaxis()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Impedance amplitude [V/A]')
        else:
            fig = ax.get_figure()

        for comp in components:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], default_comp_dict[comp])

            self.plot_station_component(
                location, comp, 'amplitude',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], default_comp_dict[comp])

                self.plot_station_errorbars(
                    location, comp, 'amplitude',
                    ax=ax, **eb_kwargs)

        return ax

    def plot_tip_amp(self, location,
                     components=['zx', 'zy'], ax=None, errorbars=False,
                     comp_plot_dict=default_comp_dict
                    ):
        """
        Plot tipper amplitude curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param components: List of the components to plot.
            Default = ['xy','yx']
        :type components: list
        :param ax: The ax object to add the,  , Default: None
        :type ax: :class:`axes <matplotlib.axes>`
        :param errorbars: Controls if errorbars are plotted
            Default = True
        :type errorbars: boolean
        :param comp_plot_dict: Dictionary with additional kwargs
            for matplotlib.plot
        :type comp_plot_dict: dict
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.invert_xaxis()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Tipper magnitude [unitless]')
        else:
            fig = ax.get_figure()

        for comp in components:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], default_comp_dict[comp])

            self.plot_station_component(
                location, comp, 'amplitude',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], default_comp_dict[comp])

                self.plot_station_errorbars(
                    location, comp, 'amplitude',
                    ax=ax, **eb_kwargs)

        return ax

    def map_data_locations(self, ax=None, **plot_kwargs):
        """
        Function that plots all receiver locations of the data
            (all discreate data locations).

        :param ax: The ax object to add the,  , Default: None
        :type ax: :class:`axes <matplotlib.axes>`

        """
        # Default plot dict
        default_dict = {
            'marker': '+',
            'c': 'k',
            'ms': 10,
            'ls': 'None',
            'zorder': 4
        }
        # Set the default plot kwargs
        for key, val in default_dict.iteritems():
            # Make sure they aren't already assigned
            if key not in plot_kwargs:
                plot_kwargs[key] = val
        # Get unique locations
        unique_locations = _unique_rows(np.concatenate(
            [rx.locs for src in self.survey.srcList for rx in src.rxList])
        )
        # Make the figure and the axes
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        # Plot the locations
        ax.plot(unique_locations[:, 0], unique_locations[:, 1],
                   **plot_kwargs)

        return (fig, ax)

    def plot_station_component(self, location, orientation, component,
                               ax=None, **plot_kwargs):
        """

        :param numpy.ndarray location: Coordnaties of the station to plot
        :param str orientation: The orientation of the data
        :param str component: The data component to plot
        :param matplotlib.axes ax (optional):
        :param matplotlib.lines.Line2D keyword_arguments plot_kwargs)

        """

        # Sort the axes
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        # Plot the data
        freqs, plot_data = _get_plot_data(self,
                                          location, orientation,
                                          component)
        # Plot
        plot_obj = ax.plot(freqs, plot_data, **plot_kwargs)


        return (fig, ax, plot_obj)

    def plot_station_errorbars(self, location, orientation, component,
                               ax=None, **plot_kwargs):
        """

        :param numpy.ndarray location: Coordnaties of the station to plot
        :param str orientation: The orientation of the data
        :param str component: The data component to plot
        :param matplotlib.axes ax (optional):
        :param matplotlib.lines.Line2D keyword_arguments plot_kwargs)

        """

        # Sort the axes
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        # Plot the data
        freqs, plot_data, errorbars = _get_data_error(self, location,
                                           orientation, component)
        plot_obj = ax.errorbar(freqs, plot_data, yerr=errorbars,
                                   **plot_kwargs)
        return (fig, ax, plot_obj)

## Hidden utility functions
# Unique row function-should be moved to utils
def _unique_rows(array):
    """
    Finds and returns unique rows in an array
    """
    array = np.ascontiguousarray(array)
    unique_array = np.unique(array.view([('', array.dtype)] * array.shape[1]))
    return unique_array.view(
        array.dtype).reshape((unique_array.shape[0], array.shape[1]))

def _get_data_error(data, location, orientation, component):

    # Get the components
    if component in ['app_res', 'phase', 'amplitude']:
        real_tuple = _extract_location_data(data, location, orientation,
                                            'real', True)
        freqs, real_data, real_std, real_floor = real_tuple
        imag_tuple = _extract_location_data(data, location, orientation,
                                            'imag', True)
        freqs, imag_data, imag_std, imag_floor = imag_tuple
        # Add up the uncertainties
        real_uncert = real_std * np.abs(real_data) + real_floor
        imag_uncert = imag_std * np.abs(imag_data) + imag_floor

        if 'app_res' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = (1. / (mu_0 * omega(freqs))) * np.abs(comp_data) ** 2
            res_uncert = ((2. / (mu_0 * omega(freqs))) *
                          (real_data * real_uncert + imag_data * imag_uncert)
                          )
            errorbars = [res_uncert, res_uncert]
        elif 'phase' in component:
            plot_data = np.arctan2(imag_data, real_data) * (180. / np.pi)
            phs_uncert = ((1. / (real_data ** 2 + imag_data ** 2)) *
                                 ((real_data * real_uncert -
                                   imag_data * imag_uncert))
                                ) * (180. / np.pi)
            # Scale back the errorbars
            errorbars = [phs_uncert, phs_uncert]
        elif 'amplitude' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = np.abs(comp_data)
            amp_uncert = ((1. / plot_data) *
                          ((np.abs(real_data) * real_uncert) +
                          (np.abs(imag_data) * imag_uncert))
                          )
            errorbars = [amp_uncert, amp_uncert] #[low_unsert, up_unsert]
    else:
        freqs, plot_data, std_data, floor_data = _extract_location_data(
            data, location, orientation, component, return_uncert=True)
        attr_uncert = std_data * np.abs(plot_data) + floor_data
        errorbars = [attr_uncert, attr_uncert]
    return (freqs, plot_data, errorbars)

def _get_plot_data(data, location, orientation, component):

        if 'app_res' in component:
            freqs, dat_r = _extract_location_data(
                data, location, orientation, 'real')
            freqs, dat_i = _extract_location_data(
                data, location, orientation, 'imag')
            dat = dat_r + 1j * dat_i
            plot_data = 1. / (mu_0 * omega(freqs)) * np.abs(dat) ** 2
        elif 'phase' in component:
            freqs, dat_r = _extract_location_data(
                data, location, orientation, 'real')
            freqs, dat_i = _extract_location_data(
                data, location, orientation, 'imag')
            plot_data = np.arctan2(dat_i, dat_r) * (180. / np.pi)
        elif 'amplitude' in component:
            freqs, dat_r = _extract_location_data(
                data, location, orientation, 'real')
            freqs, dat_i = _extract_location_data(
                data, location, orientation, 'imag')
            dat_complex = dat_r + 1j * dat_i
            plot_data = np.abs(dat_complex)
        else:
            freqs, plot_data = _extract_location_data(
                data, location, orientation, component)
        return (freqs, plot_data)

def _extract_location_data(data, location,
                           orientation, component, return_uncert=False):
    """
    Function to extract data at given location
    """
    freq_list = []
    data_list = []
    std_list = []
    floor_list = []
    for src in data.survey.srcList:
        rx_list = [rx for rx in src.rxList
              if rx.orientation == orientation and rx.component == component]
        if len(rx_list) == 0:
            if return_uncert:
                return (np.array([]), np.array([]),
                        np.array([]), np.array([]))
            return (np.array([]), np.array([]))
        else:
            rx = rx_list[0]

        ind_loc = np.sqrt(np.sum((rx.locs[:, :2] - location) ** 2, axis=1)) < 0.1
        if np.any(ind_loc):
            freq_list.append(src.freq)
            data_list.append(data[src, rx][ind_loc])
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
            if return_uncert:
                std_list.append(data.standard_deviation[src, rx][ind_loc])
                floor_list.append(data.floor[src, rx][ind_loc])
    if return_uncert:
        return (np.array(freq_list), np.concatenate(data_list),
                np.concatenate(std_list), np.concatenate(floor_list))
    return (np.array(freq_list), np.concatenate(data_list))

