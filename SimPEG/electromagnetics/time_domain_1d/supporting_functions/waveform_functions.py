import numpy as np
from scipy.integrate import fixed_quad
from scipy.special import roots_legendre



#######################################################################
#
#                  CURRENTS FOR BASIC WAVEFORMS
#
#######################################################################


def rectangular_waveform_current(times, start_time, end_time, peak_current_amplitude):

    out = np.zeros(times.size)
    out[(times>start_time) & (times<end_time)] = peak_current_amplitude
    
    return out

def triangular_waveform_current(times, start_time, peak_time, end_time, peak_current_amplitude):

    out = np.zeros(times.size)
    k = (times>=start_time) & (times<=peak_time)
    out[k] = (times[k] - start_time) * peak_current_amplitude / (peak_time - start_time) 
    k = (times>=peak_time) & (times<=end_time)
    out[k] = peak_current_amplitude * (1 - (times[k] - peak_time) / (end_time - peak_time))
            
    return out

def vtem_waveform_current(times, start_time, peak_time, end_time, decay_constant, peak_current_amplitude):

    out = np.zeros(times.size)

    k = (times>=start_time) & (times<=peak_time)
    out[k] = (
        peak_current_amplitude *
        (1 - np.exp(-decay_constant*(times[k] - start_time))) / 
        (1 - np.exp(-decay_constant*(peak_time - start_time)))
    )

    k = (times>=peak_time) & (times<=end_time)
    out[k] = peak_current_amplitude * (1 - (times[k] - peak_time) / (end_time - peak_time))
    
    return out







#######################################################################
#
#     TIMES, CURRENTS AND TIME CHANNELS FOR SPECIFIC SYSTEMS
#
#######################################################################

##########################
#       SKYTEM
##########################

def skytem_LM_waveform_times(off_time=1.250E-03):

    return np.array([
            -1.000E-03,
            -9.146E-04,
            -7.879E-04,
            -5.964E-04,
            0.000E+00,
            4.629E-07,
            8.751E-07,
            1.354E-06,
            2.540E-06,
            3.972E-06,
            5.404E-06,
            5.721E-06,
            6.113E-06,
            6.663E-06,
            8.068E-06,
            1.250E-03
        ]) - 1.250E-03 + off_time

def skytem_LM_waveform_current(peak_current_amplitude=1.):

    return np.array([
            0.000E+00,
            6.264E-01,
            9.132E-01,
            9.905E-01,
            1.000E+00,
            9.891E-01,
            9.426E-01,
            8.545E-01,
            6.053E-01,
            3.030E-01,
            4.077E-02,
            1.632E-02,
            4.419E-03,
            6.323E-04,
            0.000E+00,
            0.000E+00
        ]) * peak_current_amplitude



def skytem_HM_waveform_times(off_time=7.93939E-03):

    return np.array([
            7.53900E-05,
            9.63900E-05,
            1.22390E-04,
            1.54390E-04,
            1.96390E-04,
            2.47390E-04,
            3.12390E-04,
            3.94390E-04,
            4.97390E-04,
            6.27390E-04,
            7.90390E-04,
            9.96390E-04,
            1.25539E-03,
            1.58139E-03,
            1.99139E-03,
            2.50839E-03,
            3.15839E-03,
            3.97739E-03,
            5.00839E-03,
            6.30639E-03,
            7.93939E-03
        ]) - 7.93939E-03 + off_time

def skytem_HM_waveform_current(peak_current_amplitude=1.):

    return np.array([
            9.60000E-05,
            1.22000E-04,
            1.54000E-04,
            1.96000E-04,
            2.47000E-04,
            3.12000E-04,
            3.94000E-04,
            4.97000E-04,
            6.27000E-04,
            7.90000E-04,
            9.96000E-04,
            1.25500E-03,
            1.58100E-03,
            1.99100E-03,
            2.50800E-03,
            3.15800E-03,
            3.97700E-03,
            5.00800E-03,
            6.30600E-03,
            7.93900E-03,
            9.73900E-03
        ]) * peak_current_amplitude


##########################
#     SKYTEM 2015
##########################

def skytem_2015_LM_waveform_times(off_time=9.4274e-006):

    return np.array([
                -3.1810e-003,
                -3.1100e-003,
                -2.7860e-003,
                -2.5334e-003,
                -2.3820e-003,
                -2.3810e-003,
                -2.3798e-003,
                -2.3779e-003,
                -2.3762e-003,
                -2.3749e-003,
                -2.3733e-003,
                -2.3719e-003,
                -2.3716e-003,
                -8.0000e-004,
                -7.2902e-004,
                -4.0497e-004,
                -1.5238e-004,
                -1.0000e-006,
                0,
                1.1535e-006,
                3.0943e-006,
                4.7797e-006,
                6.1076e-006,
                7.7420e-006,
                9.0699e-006,
                9.4274e-006,
            ]) - 9.4274e-006 + off_time

def skytem_2015_LM_waveform_current(peak_current_amplitude=8.3):

    return np.array([
                0,
                -1.0078e-001,
                -4.5234e-001,
                -7.6328e-001,
                -1.0000e+000,
                -1.0000e+000,
                -8.6353e-001,
                -3.4002e-001,
                -1.1033e-001,
                -4.4709e-002,
                -1.3388e-002,
                -4.4389e-003,
                0,
                0,
                1.0078e-001,
                4.5234e-001,
                7.6328e-001,
                1.0000e+000,
                1.0000e+000,
                8.6353e-001,
                3.4002e-001,
                1.1033e-001,
                4.4709e-002,
                1.3388e-002,
                4.4389e-003,
                0
            ]) * peak_current_amplitude

def skytem_2015_LM_time_channels(first_channel_time=1.0215e-5):

    return np.array([
            10.215,
            12.715,
            16.215,
            20.715,
            26.215,
            33.215,
            42.215,
            53.715,
            68.215,
            86.215,
            108.715,
            136.715,
            172.215,
            217.715,
            274.715,
            346.715,
            437.715,
            551.715,
            695.715,
            877.215,
            1105.715,
            1394.215
        ]) * 1e-6 - 1.0215e-5 + first_channel_time


def skytem_2015_HM_waveform_times(off_time=1.96368E-04):

    return np.array([
            -2.06670E-02,
            -2.05770E-02,
            -2.04670E-02,
            -1.66670E-02,
            -1.64726E-02,
            -1.64720E-02,
            -1.64706E-02,
            -4.00000E-03,
            -3.91000E-03,
            -3.80000E-03,
            0.00000E+00,
            1.94367E-04,
            1.95038E-04,
            1.96368E-04
        ]) - 1.96368E-04 + off_time

def skytem_2015_HM_waveform_current(peak_current_amplitude=122.5):

    return np.array([
            0.00000E+00,
            -5.30000E-01,
            -9.73000E-01,
            -1.00000E+00,
            -2.81610E-03,
            -1.44356E-03,
            0.00000E+00,
            0.00000E+00,
            5.30000E-01,
            9.73000E-01,
            1.00000E+00,
            2.81610E-03,
            1.44356E-03,
            0.00000E+00
        ]) * peak_current_amplitude

def skytem_2015_HM_time_channels(first_channel_time=1.2715e-5):

    return np.array([
                12.715,
                16.215,
                20.715,
                26.215,
                33.215,
                42.215,
                53.715,
                68.215,
                86.215,
                108.715,
                136.715,
                172.215,
                217.715,
                274.715,
                346.715,
                437.715,
                551.715,
                695.715,
                877.215,
                1105.715,
                1394.215,
                1758.215,
                2216.715,
                2794.715,
                3523.715,
                4442.715,
                5601.215,
                7061.215,
                8902.215,
                11064.715
            ]) * 1e-6 - 1.2715e-5 + first_channel_time

##########################
#       VTEM PLUS
##########################

def vtem_plus_waveform_times(off_time=0.00734375):

    return np.array([
            0.,
            0.0014974,
            0.00299479,
            0.00449219,
            0.00598958,
            0.00632813,
            0.00666667,
            0.00700521,
            0.00734375
        ]) - 0.00734375 + off_time

def vtem_plus_waveform_current(peak_current_amplitude=1.):

    return np.array([
            0.00682522,
            0.68821963,
            0.88968217,
            0.95645264,
            1.,
            0.84188057,
            0.59605229,
            0.296009,
            0.
        ]) * peak_current_amplitude


def vtem_plus_time_channels(first_time_channel=2.050000e-05):

    return np.array([
                2.050000e-05,
                2.600000e-05,
                3.150000e-05,
                3.650000e-05,
                4.200000e-05,
                4.800000e-05,
                5.500000e-05,
                6.350000e-05,
                7.300000e-05,
                8.400000e-05,
                9.650000e-05,
                1.105000e-04,
                1.270000e-04,
                1.460000e-04,
                1.675000e-04,
                1.925000e-04,
                2.210000e-04,
                2.535000e-04,
                2.915000e-04,
                3.350000e-04,
                3.845000e-04,
                4.415000e-04,
                5.075000e-04,
                5.830000e-04,
                6.695000e-04,
                7.695000e-04,
                8.840000e-04,
                1.015500e-03,
                1.166500e-03,
                1.339500e-03,
                1.539000e-03,
                1.768500e-03,
                2.031500e-03,
                2.333500e-03,
                2.680000e-03,
                3.078500e-03,
                3.536500e-03,
                4.061000e-03,
                4.664000e-03,
                5.358000e-03,
                6.155000e-03,
                7.070500e-03,
                8.122500e-03,
                9.331000e-03,
                1.071745e-02
            ]) - 2.050000e-05 + first_time_channel


##################################################################################
#
#                  FUNCTIONS FOR FILTERING AND CONVOLUTION
#
##################################################################################

def piecewise_ramp(
    step_func, t_channels, t_currents, currents,
    n=20, eps=1e-10
):
    """
    Computes response from piecewise linear current waveform
    with a single pulse. This basically evaluates the convolution
    between dI/dt and step-off response.

    step_func: function handle to evaluate step-off response
    t_channels: time channels when the current is on or off
    currents: input source currents
    n: Gaussian quadrature order
    """
    dt = np.diff(t_currents)
    dI = np.diff(currents)
    dIdt = dI/dt
    nt = t_currents.size
    response = np.zeros(t_channels.size, dtype=float)
    pulse_time = t_currents.max()

    for i in range(1, nt):
        t_lag = pulse_time - t_currents[i]
        time = t_lag + t_channels
        t0 = dt[i-1]
        const = -dIdt[i-1]
        if abs(const) > eps:
            for j, t in enumerate(time):
                # on-time
                # TODO: this is only working when we have a single ramp...
                if t < 0.:
                    print (t+t0)
                    response[j] += (
                        fixed_quad(step_func, 0, t+t0, n=n)[0] * const
                    )
                # off-time
                else:
                    response[j] += (
                        fixed_quad(step_func, t, t+t0, n=n)[0] * const
                    )
    return response


def piecewise_ramp_fast(
    step_func, t_off, t_currents, currents, x, w,
    eps=1e-10
):
    """
    Computes response from piecewise linear current waveform
    with a single pulse. This basically evaluates the convolution
    between dI/dt and step-off response.

    step_func: function handle to evaluate step-off response
    t_off: time channels when the current is off
    t_shift: t_off + T/2
    currents: input source currents
    n: Gaussian quadrature order
    """
    n = x.size
    dt = np.diff(t_currents)
    dI = np.diff(currents)
    dIdt = dI/dt
    nt = t_currents.size
    pulse_time = t_currents.max()

    # Create a bunch of memory in C and use broadcasting
    t_lag = pulse_time - t_currents
    t_lag_expand = (np.repeat(t_lag[1:, np.newaxis], t_off.size, 1)).T
    t_lag_3D = np.repeat(t_lag_expand[:, :, np.newaxis], n, 2)
    t3D = t_lag_3D + t_off[:, np.newaxis, np.newaxis]
    # Gauss-Legendre part.
    # Expand time shifts and origin to 3D with G-L points
    inds = t3D[:,:,0] < 0.
    # Compute dt for both on-time and off-time
    # off-time f(t, t+t0)
    # on-time f(0, t+t0)
    dt_on_off = np.tile(dt, (t_off.size, 1))
    dt_on_off[inds] = (dt + t3D[:,:,0])[inds]
    t3D[inds,:] = 0.

    y = dt_on_off[:,:,np.newaxis] * (0.5 * (x + 1.0)) + t3D

    # Evaluate and weight G-L values with current waveform
    f = w * step_func(np.log10(y))
    s = f.sum(axis = 2) * 0.5 * dt_on_off

    response = np.sum(s * -dIdt, axis=1)

    return response


def piecewise_ramp_fast_diff(
    step_func, t_off, t_shift, t_currents, currents, x, w,
    eps=1e-10
):
    """
    Computes response from piecewise linear current waveform
    with a single pulse. This basically evaluates the convolution
    between dI/dt and step-off response.

    step_func: function handle to evaluate step-off response
    t_off: time channels when the current is off
    t_shift: t_off + T/2
    currents: input source currents
    n: Gaussian quadrature order
    """
    n = x.size
    dt = np.diff(t_currents)
    dI = np.diff(currents)
    dIdt = dI/dt
    nt = t_currents.size
    pulse_time = t_currents.max()

    # Create a bunch of memory in C and use broadcasting
    t_lag = pulse_time - t_currents
    t_lag_expand = (np.repeat(t_lag[1:, np.newaxis], t_off.size, 1)).T
    t_lag_3D = np.repeat(t_lag_expand[:, :, np.newaxis], n, 2)
    t3D = t_lag_3D + t_off[:,np.newaxis, np.newaxis]

    # Gauss-Legendre part.
    # Expand time shifts and origin to 3D with G-L points
    inds = t3D[:,:,0] < 0.
    # Compute dt for both on-time and off-time
    # off-time f(t, t+t0)
    # on-time f(0, t+t0)
    dt_on_off = np.tile(dt, (t_off.size, 1))
    dt_on_off[inds] = (dt + t3D[:,:,0])[inds]
    t3D[inds,:] = 0.

    y = dt_on_off[:,:,np.newaxis] * (0.5 * (x + 1.0)) + t3D
    # Evaluate and weight G-L values with current waveform
    f = w * step_func(np.log10(y))
    s = f.sum(axis = 2) * 0.5*dt_on_off

    response = np.sum(s * -dIdt, axis=1)

    t3D = t_lag_3D + t_shift[:,np.newaxis, np.newaxis]
    inds = t3D[:,:,0] < 0.
    # Compute dt for both on-time and off-time
    # off-time f(t, t+t0)
    # on-time f(0, t+t0)
    dt_on_off = np.tile(dt, (t_off.size, 1))
    dt_on_off[inds] = (dt + t3D[:,:,0])[inds]
    t3D[inds,:] = 0.

    # Gauss-Legendre part.
    # Expand time shifts and origin to 3D with G-L points
    y = dt_on_off[:,:,np.newaxis] * (0.5 * (x + 1.0)) + t3D
      # Evaluate and weight G-L values with current waveform
    f = w * step_func(np.log10(y))
    s = f.sum(axis = 2) * 0.5*dt_on_off

    response -= 0.5* np.sum(s * -dIdt, axis=1)

    return response



def piecewise_pulse(
    step_func, t_off, t_currents, currents, T, n=20, n_pulse=2
):
    """
    Computes response from double pulses (negative then positive)
    T: Period (e.g. 25 Hz base frequency, 0.04 s period)
    """
    if n_pulse == 1:
        response = piecewise_ramp(
                step_func, t_off, t_currents, currents, n=n
        )
    elif n_pulse == 2:
        response = (
            piecewise_ramp(
                step_func, t_off, t_currents, currents, n=n
            ) -
            piecewise_ramp(
                step_func, t_off+T/2., t_currents, currents, n=n
            ) * 0.5
        )
    else:
        raise NotImplementedError("n_pulse must be either 1 or 2")
    return response


def piecewise_pulse_fast(
    step_func, t_off, t_currents, currents, T, n=20, n_pulse=2
):
    """
    Computes response from double pulses (negative then positive)
    T: Period (e.g. 25 Hz base frequency, 0.04 s period)
    """

    # Use early out scheme for speed. Can turn assertions off with "python -O"
    assert (n_pulse == 1 or n_pulse == 2), NotImplementedError("n_pulse must be either 1 or 2")

    # Get gauss-legendre points and weights early since n never changes inside here
    # x, w = _cached_roots_legendre(n)
    x, w = roots_legendre(n)

    if n_pulse == 1:
        response = piecewise_ramp_fast(
                step_func, t_off, t_currents, currents, x, w
        )
    elif n_pulse == 2:
        response = piecewise_ramp_fast_diff(
                step_func, t_off, t_off+0.5*T, t_currents, currents, x, w
            )
    return response


def butter_lowpass_filter(highcut_frequency, fs=1e6, period=0.04, order=1):
    """
    Butterworth low pass filter

    Parameters
    ----------

    highcut_frequency: float
        high-cut frequency for the low pass filter
    fs: float
        sampling rate, 1./ dt, (default = 1MHz)
    period:
        period of the signal (e.g. 25Hz base frequency, 0.04s)
    order: int
        The order of the butterworth filter

    Returns
    -------

    frequency, h: ndarray, ndarray
        Filter values (`h`) at frequencies (`frequency`) are provided.
    """

    # Nyquist frequency
    nyq = 0.5 * fs
    n_samples = period * fs
    high = highcut_frequency / nyq
    b, a = butter(order, high, btype='low')
    w, h = freqz(b, a, worN=int(n_samples))
    frequency = (fs * 0.5 / np.pi) * w

    return frequency, h


def butterworth_type_filter(frequency, highcut_frequency, order=2):
    """
    Butterworth low pass filter

    Parameters
    ----------

    highcut_frequency: float
        high-cut frequency for the low pass filter
    fs: float
        sampling rate, 1./ dt, (default = 1MHz)
    period:
        period of the signal (e.g. 25Hz base frequency, 0.04s)
    order: int
        The order of the butterworth filter

    Returns
    -------

    frequency, h: ndarray, ndarray
        Filter values (`h`) at frequencies (`frequency`) are provided.
    """

    # Nyquist frequency
    h = 1./(1+1j*(frequency/highcut_frequency))**order
    highcut_frequency = 300*1e3
    h *= 1./(1+1j*(frequency/highcut_frequency))**1
    return h


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    xx = xy[:, 0] * np.cos(radians) + xy[:, 1] * np.sin(radians)
    yy = -xy[:, 0] * np.sin(radians) + xy[:, 1] * np.cos(radians)
    return np.c_[xx, yy]


def rotate_to_x_axis(xy, observation_point):
    """
    Moves each of two sequential points to origin (0,0)
    then rotate to x-axis
    """
    n_pts = xy.shape[0]
    dxy = np.empty((n_pts-1, 2), dtype=float, order='F')
    dxy_obs = np.empty((n_pts-1, 2), dtype=float, order='F')
    angle = np.empty(n_pts-1, dtype=float)
    dxy = np.diff(xy, axis=0)
    print (dxy)
    dxy_obs = observation_point.repeat(n_pts-1).reshape((2, n_pts-1)).T
    angle = np.arctan2(dxy[:, 1], dxy[:, 0])
    # this is for self-check
    xy_rot = rotate_origin_only(dxy, angle)
    xy_obs_rot = rotate_origin_only(dxy_obs, angle)

    # For computation of Hx, Hy angle is required
    # Hx: hx cos theta + hy sin theta
    # Hy: hx sin theta - hy cos theta
    # Hz: hz
    return xy_rot, xy_obs_rot, angle


def get_geotem_wave(pulse_period, n_segment=3):
    t = 0.
    T = pulse_period
    time_input_currents = np.r_[0., T/2., np.linspace(3*T/4., T, n_segment)]
    input_currents = np.sin(np.pi/T*time_input_currents)
    return time_input_currents, input_currents


def get_nanotem_wave(ramp_off_time):
    time_input_currents = np.r_[0, ramp_off_time]
    input_currents = np.r_[1, 0.]
    return time_input_currents, input_currents


def get_flight_direction_from_fiducial(fiducial, lines, easting, northing):
    lines_unique = np.unique(lines)
    n_line = lines_unique.size
    flight_direction = np.empty(fiducial.size, dtype=float)
    i_start = 0
    for i_line, line_unique in enumerate(lines_unique):
        ind_line = lines == line_unique
        x0, x1 = easting[ind_line][0], easting[ind_line][-1]
        y0, y1 = northing[ind_line][0], northing[ind_line][-1]
        dx = x1-x0
        dy = y1-y0
        n = ind_line.sum()
        flight_direction[i_start:i_start+n] = np.arctan2(dy, dx)
        i_start += n
    return flight_direction


def get_rx_locations_from_flight_direction(
    flight_direction, src_locations, offset=120, dz=-45
):
    dx = offset*np.cos(np.deg2rad(flight_direction))
    dy = offset*np.sin(np.deg2rad(flight_direction))
    rx_locations = np.c_[
        src_locations[:, 0]-dx, src_locations[:, 1]-dy, src_locations[:, 2]+dz
    ]
    return rx_locations
