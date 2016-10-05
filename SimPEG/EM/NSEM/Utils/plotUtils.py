from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from SimPEG.EM.Utils.EMUtils import mu_0, omega


class DataNSEM_plot_functions(object):
    """
    Class container for properties and methods for
    plotting of NSEM data.


    """

    def plot_data_locatations(self, ax=None, **plot_kwargs):
        """
        Function that plots all receiver locations of the data
        (all discreate data locations).


        """
        # Default plot dict
        default_dict = {'marker': '+', 'c': 'k', 's': 50, 'zorder': 4}
        for key, val in default_dict.iteritems():
            if key not in plot_kwargs:
                plot_kwargs[key] = val
        unique_locations = _unique_rows(np.concatenate(
            [rx.locs for src in self.survey.srcList for rx in src.rxList])
        )
        # Make the figure and the axes
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        # Plot the locations
        ax.scatter(unique_locations[:, 0], unique_locations[:, 1],
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
        rx = [rx for rx in src.rxList
              if rx.orientation == orientation and rx.component == component][0]
        ind_loc = np.sqrt(np.sum((rx.locs - location) ** 2, axis=1)) < 0.1
        if np.any(ind_loc):
            freq_list.append(src.freq)
            data_list.append(data[src, rx][ind_loc])
            std_list.append(data.standard_deviation[src, rx][ind_loc])
            floor_list.append(data.floor[src, rx][ind_loc])
    if return_uncert:
        return (np.array(freq_list), np.concatenate(data_list),
                np.concatenate(std_list), np.concatenate(floor_list))
    return (np.array(freq_list), np.concatenate(data_list))

