from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import properties
from SimPEG.EM.Utils.EMUtils import mu_0, omega
from SimPEG.EM.NSEM.RxNSEM import (
    Point_tipper3D, Point_impedance3D, Point_horizontalmagvar3D)

# Define the default component dictionaries
DEFAULT_COMP_DICT = {
    'xx':{'color': 'green', 'label': 'Imp_xx', 'marker': '_', 'ls': 'None'},
    'xy':{'color': 'blue', 'label': 'Imp_xy', 'marker': '_', 'ls': 'None'},
    'yx':{'color': 'red', 'label': 'Imp_yx', 'marker': '_', 'ls': 'None'},
    'yy':{'color': 'darkorange', 'label': 'Imp_yy', 'marker': '_', 'ls': 'None'},
    'zx':{'color': 'brown', 'label': 'Tip_zx', 'marker': '_', 'ls': 'None'},
    'zy':{'color': 'purple', 'label': 'Tip_zy', 'marker': '_', 'ls': 'None'},
}

# Define some hidden attributes
_imp_comps = ['xx', 'xy', 'yx', 'yy']

def _validate_kwargs(input_dict, compare_dict):
    """
    Function to deal with keyword arguments.

    :param input_dict: matplotlib kwargs dictionary with custom arguments
    :type input_dict: :class: `dict`
    :param compare_dict: matplotlib kwargs of default to use arguments
    :type compare_dict: :class: `dict`
    """
    # Set the default plot kwargs
    for key, val in compare_dict.items():
        # Make sure they aren't already assigned
        if input_dict is None:
            input_dict = compare_dict.copy()
        else:
            if key not in input_dict:
                input_dict[key] = val
    # Return
    return input_dict


class Base_DataNSEM_plots(properties.HasProperties):
    """
    A class container of matplotlib panels for plotting
    NSEM data.

    """
    fig = properties.Instance(
        'Figure plotting',
        plt.Figure,
        required=False)
    axes = properties.List(
        'List of plot axes',
        properties.Instance('Axes to plot the on', plt.Axes),
        required=False)

    def setup(self):
        """
        Setup up the plot window.

        Should populate the
            self.fig and self.axes properties
        """
        raise NotImplementedError('Is required in subclasses of {}'.format(self.__class__))

    def draw(self):
        raise NotImplementedError('Is required in subclasses of {}'.format(self.__class__))


    def clear_axes(self):
        """
        Function to clear all of the axes
        """
        for ax in self.axes:
            while len(ax.lines) > 0:
                for line in ax.lines:
                    ax.lines.remove(line)

            while len(ax.collections) > 0:
                for item in ax.collections:
                    item.remove()


class tip_amp_station_plot(Base_DataNSEM_plots):
    """
    Class for setting up 2 axes figure with:
        tipper amplitudes | tipper phase
        setup.
    """

    def __init__(self):
        super(tip_amp_station_plot, self).__init__()


    def setup(self):
        """
        Setup a station data plot figure.
        """
        self.fig, axes_temp = plt.subplots(1, 2, sharex=True)
        self.axes = axes_temp.ravel().tolist()
        self.fig.set_size_inches((13.5, 4.0))

        for ax in self.axes:
            ax.set_xscale('log')

        self.axes[0].invert_xaxis()
        self.axes[0].set_yscale('log')
        # Set labels
        self.axes[0].set_xlabel('Frequency [Hz]')
        self.axes[1].set_xlabel('Frequency [Hz]')
        self.axes[0].set_ylabel('Tipper amplitude [V/A]')
        self.axes[1].set_ylim(-180, 180)
        self.axes[1].set_ylabel('Tipper angle [degrees]')

    def draw(self, data_list, location):
        """
        Function to draw on the axes

        :param data_list: List of NSEM data objects to plot.
            Has to be of length >= 1. First item is treat as a
            observed data (Hast to have standard_deviation and floor)
            assigned) and the others are plotted on top.
        :param location: Location of the station to plot
        """

        axes = self.axes

        # Set keyword arguments
        st_kwargs = {'marker': '_', 'ls': 'None'}
        eb_kwargs = {'ls': 'None'}
        # Pop the data from the list
        data = data_list[0]

        # Apparent resistivity
        data.plot_tip_amp(location, ['zx', 'zy'],
                          ax=axes[0], errorbars=True)

        # Apparent phase
        data.plot_app_phs(location, ['zx', 'zy'],
                          ax=axes[1], errorbars=True)

        # Plot the additional data
        for other_data in data_list[1::]:
            # Need add symbol generation
            dd_kwargs = {'zx': {'marker': '.', 'ls': '--'},
                         'zy': {'marker': '.', 'ls': '--'}}

            # Apparent resistivity
            other_data.plot_tip_amp(location, ['zx', 'zy'],
                                    ax=axes[0], errorbars=False,
                                    comp_plot_dict=dd_kwargs)

            # Apparent phase
            other_data.plot_app_phs(location, ['zx', 'zy'],
                                    ax=axes[1], errorbars=False,
                                    comp_plot_dict=dd_kwargs)


class tip_complex_station_plot(Base_DataNSEM_plots):
    """
    Class for setting up 2 axes figure with:
        tipper real | tipper imag
        setup.
    """

    def __init__(self):
        super(tip_complex_station_plot, self).__init__()


    def setup(self):
        """
        Setup a station data plot figure.
        """
        self.fig, axes_temp = plt.subplots(1, 2, sharex=True)
        self.axes = axes_temp.ravel().tolist()
        self.fig.set_size_inches((13.5, 4.0))

        for ax in self.axes:
            ax.set_xscale('log')
            ax.set_yscale('symlog', linthreshy=1e-3)

        self.axes[0].invert_xaxis()

        # Set labels
        self.axes[0].set_xlabel('Frequency [Hz]')
        self.axes[1].set_xlabel('Frequency [Hz]')
        self.axes[0].set_ylabel('Tipper real component')
        self.axes[1].set_ylabel('Tipper imaginary component')

    def draw(self, data_list, location):
        """
        Function to draw on the axes

        :param data_list: List of NSEM data objects to plot.
            Has to be of length >= 1. First item is treat as a
            observed data (Hast to have standard_deviation and floor)
            assigned) and the others are plotted on top.
        :param location: Location of the station to plot
        """

        axes = self.axes

        # Set keyword arguments
        st_kwargs = DEFAULT_COMP_DICT
        # Pop the data from the list
        data = data_list[0]

        # Apparent resistivity
        for comp in ['zx', 'zy']:
            data.station_errorbars(
                location, 'T', comp, 'real',
                ax=axes[0], **st_kwargs[comp])
            # Apparent phase
            data.station_errorbars(
                location, 'T', comp, 'imag',
                ax=axes[1], **st_kwargs[comp])

        # Plot the additional data
        for other_data in data_list[1::]:
            # Need add symbol generation
            comp_plot_dict = {
                'zx': {'marker': '.', 'ls': '--'},
                'zy': {'marker': '.', 'ls': '--'}}

            for comp in ['zx', 'zy']:
                dd_kwargs = _validate_kwargs(
                comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])
                # Real part
                other_data.station_component(
                    location, 'T', comp, 'real',
                    ax=axes[0], **dd_kwargs)

                # Imaginary part
                other_data.station_component(
                    location, 'T', comp, 'imag',
                    ax=axes[1], **dd_kwargs)

        axes[1].legend(
            loc='center left', bbox_to_anchor=(1, 0.5))


class app_res_phs_imp_station_plot(Base_DataNSEM_plots):
    """
    Class for setting up 4 axes figure with:
            apparent resistivity | phase
            --------------------------------
            impedance amplitudes | impedance phase
        setup.
    """

    def __init__(self):
        super(app_res_phs_imp_station_plot, self).__init__()

    def setup(self):
        """
        Setup a station data plot figure.
        """
        self.fig, axes_temp = plt.subplots(2, 2, sharex=True)
        self.axes = axes_temp.ravel().tolist()
        self.fig.set_size_inches((13.5, 7.0))

        # Have to deal with axes
        # Set log

        for ax in self.axes:
            ax.set_xscale('log')

        self.axes[0].invert_xaxis()
        self.axes[0].set_yscale('log')
        self.axes[2].set_yscale('log')
        # Set labels
        self.axes[2].set_xlabel('Frequency [Hz]')
        self.axes[3].set_xlabel('Frequency [Hz]')
        self.axes[0].set_ylabel('Apperent resistivity [Ohm m]')
        self.axes[1].set_ylabel('Apperent phase [degrees]')
        self.axes[1].set_ylim(-180, 180)
        self.axes[2].set_ylabel('Impedance amplitude [V/A]')
        self.axes[3].set_ylim(-180, 180)
        self.axes[3].set_ylabel('Impedance angle [degrees]')

    def draw(self, data_list, location):
        """
        Function to draw on the axes

        :param data_list: List of NSEM data objects to plot.
            Has to be of length >= 1. First item is treat as a
            observed data (Hast to have standard_deviation and floor)
            assigned) and the others are plotted on top.
        :param location: Location of the station to plot
        """

        axes = self.axes

        # Set keyword arguments
        st_kwargs = {'marker': '_', 'ls': 'None'}
        # Pop the data from the list
        data = data_list[0]

        # Apparent resistivity
        data.plot_app_res(location, ['xy', 'yx'],
                          ax=axes[0], errorbars=True)

        # Apparent phase
        data.plot_app_phs(location, ['xy', 'yx'],
                          ax=axes[1], errorbars=True)

        # Impedamce amplitude
        data.plot_imp_amp(location, ['xx', 'xy', 'yx', 'yy'],
                          ax=axes[2], errorbars=True)

        # Impedance phase
        data.plot_app_phs(location, ['xx', 'xy', 'yx', 'yy'],
                          ax=axes[3], errorbars=True)


        # Plot the additional data
        for other_data in data_list[1::]:
            # Need add symbol generation
            dd_kwargs = {'xx': {'marker': '.', 'ls': '--'},
                         'xy': {'marker': '.', 'ls': '--'},
                         'yx': {'marker': '.', 'ls': '--'},
                         'yy': {'marker': '.', 'ls': '--'}}

            # Apparent resistivity
            other_data.plot_app_res(location, ['xy', 'yx'],
                                    ax=axes[0], errorbars=False,
                                    comp_plot_dict=dd_kwargs)

            # Apparent phase
            other_data.plot_app_phs(location, ['xy', 'yx'],
                                    ax=axes[1], errorbars=False,
                                    comp_plot_dict=dd_kwargs)

            # Impedamce amplitude
            other_data.plot_imp_amp(location,
                                    ['xx', 'xy', 'yx', 'yy'],
                                    ax=axes[2], errorbars=False,
                                    comp_plot_dict=dd_kwargs)

            # Impedance phase
            other_data.plot_app_phs(location,
                                    ['xx', 'xy', 'yx', 'yy'],
                                    ax=axes[3], errorbars=False,
                                    comp_plot_dict=dd_kwargs)
        # axes[3].legend(
        #     loc='center left', bbox_to_anchor=(1, 1))


class hmv_complex_station_plot(Base_DataNSEM_plots):
    """
    Class for setting up 2 axes figure with:
        hmv real | hmv imag
        setup.
    """

    def __init__(self):
        super(hmv_complex_station_plot, self).__init__()


    def setup(self):
        """
        Setup a station data plot figure.
        """
        self.fig, axes_temp = plt.subplots(1, 2, sharex=True, sharey=True)
        self.axes = axes_temp.ravel().tolist()
        self.fig.set_size_inches((13.5, 4.0))

        for ax in self.axes:
            ax.set_xscale('log')
            ax.set_yscale('symlog', linthreshy=1e-3)
        self.axes[0].invert_xaxis()
        # Set labels
        self.axes[0].set_xlabel('Frequency [Hz]')
        self.axes[1].set_xlabel('Frequency [Hz]')
        self.axes[0].set_ylabel('HMV real component')
        self.axes[1].set_ylabel('HMV imaginary component')
        # Set the kwargs dict
        new_labels = {
            'xx': {'label': 'Imp_xx'},
            'xy': {'label': 'Imp_xy'},
            'yx': {'label': 'Imp_yx'},
            'yy': {'label': 'Imp_yy'}}
        hmv_kwargs = {}
        for comp in ['xx', 'xy', 'yx', 'yy']:
            hmv_kwargs[comp] = _validate_kwargs(
                new_labels[comp],
                DEFAULT_COMP_DICT[comp])
        self._st_kwargs = hmv_kwargs

    def draw(self, data_list, location):
        """
        Function to draw on the axes

        :param data_list: List of NSEM data objects to plot.
            Has to be of length >= 1. First item is treat as a
            observed data (Hast to have standard_deviation and floor)
            assigned) and the others are plotted on top.
        :param location: Location of the station to plot
        """

        axes = self.axes

        # Pop the data from the list
        data = data_list[0]

        # Apparent resistivity
        for comp in ['xx', 'xy', 'yx', 'yy']:
            data.station_errorbars(
                location, 'M', comp, 'real',
                ax=axes[0], **self._st_kwargs[comp])
            # Apparent phase
            data.station_errorbars(
                location, 'M', comp, 'imag',
                ax=axes[1], **self._st_kwargs[comp])

        # Plot the additional data
        for other_data in data_list[1::]:
            # Need add symbol generation
            comp_plot_dict = {'marker': '.', 'ls': '--'}

            for comp in ['xx', 'xy', 'yx', 'yy']:
                dd_kwargs = _validate_kwargs(
                    comp_plot_dict, self._st_kwargs[comp])
                # Real component
                other_data.station_component(
                    location, 'M', comp, 'real',
                    ax=axes[0], **dd_kwargs)

                # Imaginary component
                other_data.station_component(
                    location, 'M', comp, 'imag',
                    ax=axes[1], **dd_kwargs)

        axes[1].legend(
            loc='center left', bbox_to_anchor=(1, 0.5))


class DataNSEM_plot_functions(object):
    """
    Class container for properties and methods for
    plotting of NSEM data.

    """

    def __init__(self):
        """
        Just for reference
        """
        pass

    def plot_app_res(
        self, location, orientations=['xy', 'yx'], ax=None,
        errorbars=False, comp_plot_dict=DEFAULT_COMP_DICT):
        """
        Plot apperent resistivity curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param orientations: List of the orientations to plot.
            Default = ['xy','yx']
        :type orientations: list
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

        for comp in orientations:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

            self.station_component(
                location, 'Z', comp, 'app_res',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

                self.station_errorbars(
                    location, 'Z', comp, 'app_res',
                    ax=ax, **eb_kwargs)

        return ax

    def plot_app_phs(
        self, location, orientations=['xy', 'yx'], ax=None,
        errorbars=False, comp_plot_dict=DEFAULT_COMP_DICT):
        """
        Plot apperent resistivity curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param orientations: List of the orientations to plot.
            Default = ['xy','yx']
        :type orientations: list
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

        for comp in orientations:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

            self.station_component(
                location, 'Z', comp, 'phase',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

                self.station_errorbars(
                    location, 'Z', comp, 'phase',
                    ax=ax, **eb_kwargs)

        return ax

    def plot_imp_amp(
        self, location, orientations=['xy', 'yx'], ax=None,
        errorbars=False, comp_plot_dict=DEFAULT_COMP_DICT):
        """
        Plot impedance amplitude curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param orientations: List of the orientations to plot.
            Default = ['xy','yx']
        :type orientations: list
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

        for comp in orientations:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

            self.station_component(
                location, 'Z', comp, 'amplitude',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

                self.station_errorbars(
                    location, 'Z', comp, 'amplitude',
                    ax=ax, **eb_kwargs)

        return ax

    def plot_tip_amp(
        self, location, orientations=['zx', 'zy'], ax=None,
        errorbars=False, comp_plot_dict=DEFAULT_COMP_DICT):
        """
        Plot tipper amplitude curves at a given location

        :param location: Location of the data point
        :type location: :class:`axes <matplotlib.axes>`
        :param orientations: List of the orientations to plot.
            Default = ['xy','yx']
        :type orientations: list
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

        for comp in orientations:
            st_kwargs = _validate_kwargs(
                comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

            self.station_component(
                location, 'T', comp, 'amplitude',
                ax=ax, **st_kwargs)
            if errorbars:
                eb_kwargs = _validate_kwargs(
                    comp_plot_dict[comp], DEFAULT_COMP_DICT[comp])

                self.station_errorbars(
                    location, 'T', comp, 'amplitude',
                    ax=ax, **eb_kwargs)

        return ax

    def map_iso_frequency(
        self, freq, tensor, orientation, component='xy', ax=None,
        plot_error=False, **plot_kwargs):
        """
        Function that maps the data at an certain frequency.
        :param frequency: The frequency to map the data for
        :type frequency: :class:`float`
        :param ax: The ax object for mapping to. Default: None
        :type ax: :class:`axes <matplotlib.axes>`

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.invert_xaxis()
            ax.set_xlabel('Distance [Easting]')
            ax.set_ylabel('Distance [Northing]')
        else:
            fig = ax.get_figure()

        self.frequency_map(
            freq, tensor, orientation=orientation, component=component,
            ax=ax, plot_error=plot_error, **plot_kwargs)

    def map_data_locations(self, ax=None, **plot_kwargs):
        """
        Function that plots all receiver locations of the data
            (all discreate data locations).

        :param ax: The ax object for mapping to. Default: None
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
        for key, val in default_dict.items():
            # Make sure they aren't already assigned
            if key not in plot_kwargs:
                plot_kwargs[key] = val
        # Get unique locations
        unique_locations = _unique_rows(np.concatenate(
            [rx.locs if len(rx.locs.shape) == 2 else rx.locs[:,:,0]
            for src in self.survey.srcList for rx in src.rxList])
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

    def station_component(
        self, location, tensor, orientation, component,
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
                                          location, tensor, orientation,
                                          component)
        # Plot
        plot_obj = ax.plot(freqs, plot_data, **plot_kwargs)

        return (fig, ax, plot_obj)

    def station_errorbars(
        self, location, tensor, orientation, component,
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
        freqs, plot_data, errorbars = _get_station_data(
            self, location, tensor, orientation, component, plot_error=True)
        plot_obj = ax.errorbar(freqs, plot_data, yerr=errorbars,
                               **plot_kwargs)
        return (fig, ax, plot_obj)

    def frequency_map(
        self, frequency, tensor, orientation, component,
        ax=None, plot_error=True, **plot_kwargs):
        """
        Function to generate a iso-frequency map

        :param numpy.ndarray frequency: Frequency to be mapped
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
        info_set = _get_map_data(
            self, frequency, tensor, orientation, component, plot_error)
        if plot_error:
            locs, plot_data, errorbars = info_set
            plot_obj = plt.tricontourf(locs[:,0], locs[:,1], errorbars[0],
                                       **plot_kwargs)
        else:
            locs, plot_data = info_set
            plot_obj = plt.tricontourf(locs[:,0], locs[:,1], plot_data,
                                       **plot_kwargs)
        return (fig, ax, plot_obj)

    def section_plot(
        self, sectDict, tensor, orientation, component,
        ax=None, plot_error=False, **plot_kwargs):
        """
        Function to generate a section plot

        :param numpy.ndarray sectDict: Section dictionary to be mapped
            with keys of 'x' or 'y' and the coordinate
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
        info_set = _get_section_data(self,
            sectDict, tensor, orientation, component, plot_error)
        if plot_error:
            freqs, locs, plot_data, errorbars = info_set
        else:
            freqs, locs, plot_data = info_set
        # Find the location
        loc_arr = locs[:, 1 if 'x' in sectDict.keys() else 0]
        # Make the plot points
        xP, yP = loc_arr, freqs
        if plot_error:
            plot_obj = ax.tricontourf(
                xP.reshape(-1,), yP.reshape(-1,), errorbars[0], **plot_kwargs)
        else:
            plot_obj = ax.tricontourf(
                xP.reshape(-1,), yP.reshape(-1,), plot_data, **plot_kwargs)
        return (fig, ax, plot_obj)


# Hidden utils functions
def _get_section_data(
    data, sectDict, tensor, orientation, component,
    plot_error=False):
    """
    Function for getting frequency section
    """

    # Get the components
    if component in ['app_res', 'phase', 'amplitude']:
        real_tuple = _extract_section_data(
            data, sectDict, tensor, orientation, 'real', plot_error)
        imag_tuple = _extract_section_data(
            data, sectDict, tensor, orientation, 'imag', plot_error)
        if plot_error:
            freqs, locs, real_data, real_std, real_floor = real_tuple
            freqs, locs, imag_data, imag_std, imag_floor = imag_tuple
            # Add up the uncertainties
            real_uncert = real_std * np.abs(real_data) + real_floor
            imag_uncert = imag_std * np.abs(imag_data) + imag_floor
        else:
            freqs, locs, real_data = real_tuple
            freqs, locs, imag_data = imag_tuple

        if 'app_res' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = (
                1. / (mu_0 * omega(freqs))) * np.abs(comp_data) ** 2
            if plot_error:
                res_uncert = (
                    (2. / (mu_0 * omega(freqs))) *
                    (real_data * real_uncert + imag_data * imag_uncert)
                )
                error = res_uncert
        elif 'phase' in component:
            plot_data = np.arctan2(imag_data, real_data) * (180. / np.pi)
            if plot_error:
                phs_uncert = (
                    (1. / (real_data ** 2 + imag_data ** 2)) *
                    ((real_data * real_uncert - imag_data * imag_uncert))
                ) * (180. / np.pi)
                # Scale back the errorbars
                error = phs_uncert
        elif 'amplitude' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = np.abs(comp_data)
            if plot_error:
                amp_uncert = ((1. / plot_data) *
                              ((np.abs(real_data) * real_uncert) +
                              (np.abs(imag_data) * imag_uncert))
                              )
                error = amp_uncert #[low_unsert, up_unsert]
    else:
        if plot_error:
            freqs, locs, plot_data, std_data, floor_data = _extract_section_data(
                data, sectDict, tensor, orientation, component, return_uncert=plot_error)
            attr_uncert = std_data * np.abs(plot_data) + floor_data
            error =  attr_uncert
        else:
            freqs, locs, plot_data = _extract_section_data(
                data, sectDict, tensor, orientation, component, return_uncert=plot_error)
    if plot_error:
        return (freqs, locs, plot_data, error)
    else:
        return (freqs, locs, plot_data)


def _get_map_data(
    data, frequency, tensor, orientation, component,
    plot_error=False):
    """
    Function for getting frequency map data
    """

    # Get the components
    if component in ['app_res', 'phase', 'amplitude']:
        real_tuple = _extract_frequency_data(data, frequency, tensor, orientation,
                                            'real', plot_error)
        imag_tuple = _extract_frequency_data(data, frequency, tensor, orientation,
                                            'imag', plot_error)
        if plot_error:
            freqs, locs, real_data, real_std, real_floor = real_tuple
            freqs, locs, imag_data, imag_std, imag_floor = imag_tuple
            # Add up the uncertainties
            real_uncert = real_std * np.abs(real_data) + real_floor
            imag_uncert = imag_std * np.abs(imag_data) + imag_floor
        else:
            freqs, locs, real_data = real_tuple
            freqs, locs, imag_data = imag_tuple

        if 'app_res' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = (1. / (mu_0 * omega(freqs))) * np.abs(comp_data) ** 2
            if plot_error:
                res_uncert = (
                    (2. / (mu_0 * omega(freqs))) *
                    (real_data * real_uncert + imag_data * imag_uncert)
                )
                error = res_uncert
        elif 'phase' in component:
            plot_data = np.arctan2(imag_data, real_data) * (180. / np.pi)
            if plot_error:
                phs_uncert = (
                    (1. / (real_data ** 2 + imag_data ** 2)) *
                    ((real_data * real_uncert - imag_data * imag_uncert))
                ) * (180. / np.pi)
                # Scale back the errorbars
                error = phs_uncert
        elif 'amplitude' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = np.abs(comp_data)
            if plot_error:
                amp_uncert = ((1. / plot_data) *
                              ((np.abs(real_data) * real_uncert) +
                              (np.abs(imag_data) * imag_uncert))
                              )
                error =  amp_uncert #[low_unsert, up_unsert]
    else:

        if plot_error:
            freqs, locs, plot_data, std_data, floor_data = _extract_frequency_data(
                data, frequency, tensor, orientation, component, return_uncert=plot_error)
            attr_uncert = std_data * np.abs(plot_data) + floor_data
            error =  attr_uncert
        else:
            freqs, locs, plot_data = _extract_frequency_data(
                data, frequency, tensor, orientation, component, return_uncert=plot_error)
    # Adding this to make the ploting work for HMV
    if len(locs.shape) == 3:
        locs = locs[:, :, 0]
    if plot_error:
        return (locs, plot_data, error)
    else:
        return (locs, plot_data)


def _get_station_data(
    data, location, tensor, orientation, component,
    plot_error=False):

    # Get the components
    if component in ['app_res', 'phase', 'amplitude']:
        real_tuple = _extract_location_data(data, location, tensor, orientation,
                                            'real', plot_error)
        imag_tuple = _extract_location_data(data, location, tensor, orientation,
                                            'imag', plot_error)

        if plot_error:
            freqs, real_data, real_std, real_floor = real_tuple
            freqs, imag_data, imag_std, imag_floor = imag_tuple
            # Add up the uncertainties
            real_uncert = real_std * np.abs(real_data) + real_floor
            imag_uncert = imag_std * np.abs(imag_data) + imag_floor
        else:
            freqs, real_data = real_tuple
            freqs, imag_data = imag_tuple

        if 'app_res' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = (1. / (mu_0 * omega(freqs))) * np.abs(comp_data) ** 2
            if plot_error:
                res_uncert = (
                    (2. / (mu_0 * omega(freqs))) *
                    (real_data * real_uncert + imag_data * imag_uncert)
                )
                error = res_uncert
        elif 'phase' in component:
            plot_data = np.arctan2(imag_data, real_data) * (180. / np.pi)
            if plot_error:
                phs_uncert = (
                    (1. / (real_data ** 2 + imag_data ** 2)) *
                    ((real_data * real_uncert - imag_data * imag_uncert))
                ) * (180. / np.pi)
                # Scale back the errorbars
                error = phs_uncert
        elif 'amplitude' in component:
            comp_data = real_data + 1j * imag_data
            plot_data = np.abs(comp_data)
            if plot_error:
                amp_uncert = ((1. / plot_data) *
                              ((np.abs(real_data) * real_uncert) +
                              (np.abs(imag_data) * imag_uncert))
                              )
                error = amp_uncert #[low_unsert, up_unsert]
    else:
        if plot_error:
            freqs, plot_data, std_data, floor_data = _extract_location_data(
                data, location, tensor, orientation, component, return_uncert=plot_error)
            error = std_data * np.abs(plot_data) + floor_data
        else:
            freqs, plot_data = _extract_location_data(
                data, location, tensor, orientation, component, return_uncert=plot_error)
    if plot_error:
        return (freqs, plot_data, error)
    else:
        return (freqs, plot_data)


def _get_plot_data(
    data, location, tensor, orientation, component):

    return _get_station_data(data, location, tensor, orientation, component, plot_error=False)


def _extract_frequency_data(
    data, frequency, tensor, orientation, component,
    return_uncert=False):
    """
    Function to extract data at given frequency
    """

    src = data.survey.getSrcByFreq(frequency)[0]
    rx = _get_rx(src, tensor, orientation, component)
    if rx is None:
        if return_uncert:
            return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))
        return (np.array([]), np.array([]), np.array([]))

    loc_arr = rx.locs
    data_arr = data[src, rx]
    if return_uncert:
        std_arr = data._get_data_properties(
            (src, rx), 'standard_deviation')
        floor_arr = data._get_data_properties(
            (src, rx), 'noise_floor')
        return (frequency, loc_arr, data_arr, std_arr, floor_arr)
    return (frequency, loc_arr, data_arr)


def _extract_section_data(
    data, sectDict, tensor, orientation, component,
    return_uncert=False):
    """
    Function to extract data at given section, given
    sectDict {'x'|'y': coord_value}
    """

    freq_list = []
    loc_list = []
    data_list = []
    std_list = []
    floor_list = []
    for src in data.survey.srcList:
        rx = _get_rx(src, tensor, orientation, component)
        if rx is not None:
            # Fix for the magnetic
            if len(rx.locs.shape) == 3:
                locs = rx.locs[:,:,0]
            else:
                locs = rx.locs
            if 'x' in sectDict.keys():
                ind_loc = np.sqrt(
                    (locs[:, 0] - np.array([sectDict['x']])) ** 2) < 1.
            elif 'y' in sectDict.keys():
                ind_loc = np.sqrt(
                    (locs[:, 1] - np.array([sectDict['y']])) ** 2) < 1.

            if np.any(ind_loc):
                freq_list.append(np.ones((np.sum(ind_loc),), float) * src.freq)
                data_list.append(data[src, rx][ind_loc])
                loc_list.append(locs[ind_loc, :])

                if return_uncert:
                    std_list.append(
                        data._get_data_properties(
                            (src, rx),
                            'standard_deviation')[ind_loc])
                    floor_list.append(
                        data._get_data_properties(
                            (src, rx),
                            'noise_floor')[ind_loc])

    if len(data_list) == 0:
        if return_uncert:
            return (np.array([]), np.array([]),
                    np.array([]), np.array([]), np.array([]))
        return (np.array([]), np.array([]), np.array([]))

    if return_uncert:
        return (np.concatenate(freq_list), np.concatenate(loc_list),
            np.concatenate(data_list), np.concatenate(std_list),
            np.concatenate(floor_list))
    return (np.concatenate(freq_list), np.concatenate(loc_list),
        np.concatenate(data_list))



def _extract_location_data(
    data, location, tensor, orientation, component,
    return_uncert=False):
    """
    Function to extract data at given location
    """
    freq_list = []
    data_list = []
    std_list = []
    floor_list = []
    for src in data.survey.srcList:
        rx = _get_rx(src, tensor, orientation, component)
        if rx is None:
            if return_uncert:
                return (np.array([]), np.array([]),
                        np.array([]), np.array([]))
            return (np.array([]), np.array([]))
        if len(rx.locs.shape) == 2:
            ind_loc = np.sqrt(
                np.sum((rx.locs[:, :2] - location) ** 2, axis=1)) < 0.1
        elif len(rx.locs.shape) == 3:
            ind_loc = np.sqrt(
                np.sum((rx.locs[:, :2, 0] - location) ** 2, axis=1)) < 0.1
        if np.any(ind_loc):
            freq_list.append(src.freq)
            data_list.append(data[src, rx][ind_loc])

            if return_uncert:
                std_list.append(
                    data._get_data_properties(
                        (src, rx),
                        'standard_deviation')[ind_loc])
                floor_list.append(
                    data._get_data_properties(
                        (src, rx),
                        'noise_floor')[ind_loc])
    if return_uncert:
        return (np.array(freq_list), np.concatenate(data_list),
                np.concatenate(std_list), np.concatenate(floor_list))
    return (np.array(freq_list), np.concatenate(data_list))

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

def _get_rx(src, tensor_type, orientation, component):

    if 'Z' in tensor_type:
        class_type = Point_impedance3D
    elif 'T' in tensor_type:
        class_type = Point_tipper3D
    elif 'M' in tensor_type:
        class_type = Point_horizontalmagvar3D
    else:
        raise IOError('{} is a illegal tensor type. Has to be [Z,T,M]'.format(tensor_type))
    rx_list = [rx for rx in src.rxList
               if isinstance(rx, class_type) and
               rx.orientation == orientation and
               rx.component == component]
    # Check the number of the rx
    if len(rx_list) == 1:
        return rx_list[0]
    elif len(rx_list) == 0:
        return None
    else:
        # Should be a more specifice Exeption
        raise Exception(
            'To many Receivers of the same type, orientation and component')