import numpy as np
import scipy.special as spec
from ...utils import validate_ndarray_with_shape, validate_float

# import properties

from ...survey import BaseSrc
from .waveforms import BaseVRMWaveform

#########################################
# BASE VRM SOURCE CLASS
#########################################


class BaseSrcVRM(BaseSrc):
    """Base VRM source class

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.viscous_remanent_magnetization.receivers.Point
        A list of VRM receivers
    location : (3) array_like
        Source location
    waveform : SimPEG.electromagnetics.viscous_remanent_magnetization.waveforms.BaseVRMWaveform
        A VRM waveform
    """

    def __init__(self, receiver_list, location=None, waveform=None, **kwargs):

        if not isinstance(waveform, BaseVRMWaveform):
            AttributeError(
                "Waveform must be an instance of a VRM waveform class: StepOff, SquarePulse or Arbitrary"
            )

        super(BaseSrcVRM, self).__init__(receiver_list, location, **kwargs)
        self.waveform = waveform

    @property
    def nRx(self):
        """Total number of receiver locations

        Returns
        -------
        int
            Number of receiver locations
        """
        return np.sum(
            np.array([np.shape(rx.locations)[0] for rx in self.receiver_list])
        )

    @property
    def vnRx(self):
        """Vector number of receiver locations

        Returns
        -------
        list of int
            Number of receivers per source
        """
        return np.array([np.shape(rx.locations)[0] for rx in self.receiver_list])


#########################################
# MAGNETIC DIPOLE VRM SOURCE CLASS
#########################################


class MagDipole(BaseSrcVRM):
    """Magnetic dipole source

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.viscous_remanent_magnetization.receivers.Point
        VRM receivers
    location : (3) array_like
        source location
    moment : (3) array_like
        dipole moment (mx, my, mz)
    waveform : SimPEG.electromagnetics.viscous_remanent_magnetization.waveforms.BaseVRMWaveform
        VRM waveform
    """

    def __init__(self, receiver_list, location, moment, waveform, **kwargs):

        if len(location) != 3:
            raise ValueError(
                "Tx location (x,y,z) must be given as a column vector of length 3."
            )

        if len(moment) != 3:
            raise ValueError(
                "Dipole moment (mx,my,mz) must be given as a column vector of length 3."
            )

        super(MagDipole, self).__init__(receiver_list, location, waveform, **kwargs)

        self.moment = moment

    @property
    def moment(self):
        """Dipole moment (mx, my, mz)

        Returns
        -------
        (3) numpy.ndarray
            Dipole moment (mx, my, mz)
        """
        return self._moment

    @moment.setter
    def moment(self, val):
        self._moment = validate_ndarray_with_shape(
            "moment", val, shape=("*",), dtype=float
        )

    def getH0(self, xyz):
        """Compute inducing field at locations xyz

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            locations at which primary field components are computed

        Returns
        -------
        (n, 3) numpy.ndarray
            Primary magnetic field at all xyz locations; organized as columns [Hx0,Hy0,Hz0]
        """

        m = self.moment
        r0 = self.location

        r = np.sqrt(
            (xyz[:, 0] - r0[0]) ** 2
            + (xyz[:, 1] - r0[1]) ** 2
            + (xyz[:, 2] - r0[2]) ** 2
        )
        mdotr = (
            m[0] * (xyz[:, 0] - r0[0])
            + m[1] * (xyz[:, 1] - r0[1])
            + m[2] * (xyz[:, 2] - r0[2])
        )

        hx0 = (1 / (4 * np.pi)) * (
            3 * (xyz[:, 0] - r0[0]) * mdotr / r ** 5 - m[0] / r ** 3
        )
        hy0 = (1 / (4 * np.pi)) * (
            3 * (xyz[:, 1] - r0[1]) * mdotr / r ** 5 - m[1] / r ** 3
        )
        hz0 = (1 / (4 * np.pi)) * (
            3 * (xyz[:, 2] - r0[2]) * mdotr / r ** 5 - m[2] / r ** 3
        )

        return np.c_[hx0, hy0, hz0]

    def _getRefineFlags(self, xyzc, refinement_factor, refinement_distance):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        refinement_factor -- Refinement factors

        refinement_distance -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        refFlag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        r = np.sqrt(
            (xyzc[:, 0] - self.location[0]) ** 2
            + (xyzc[:, 1] - self.location[1]) ** 2
            + (xyzc[:, 2] - self.location[2]) ** 2
        )

        for nn in range(0, refinement_factor):

            k = (r < refinement_distance[nn] + 1e-5) & (
                refFlag < refinement_factor - nn + 1
            )
            refFlag[k] = refinement_factor - nn

        return refFlag


#########################################
# CIRCULAR LOOP VRM SOURCE CLASS
#########################################


class CircLoop(BaseSrcVRM):
    """Circular loop source

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.viscous_remanent_magnetization.receivers.Point
        VRM receivers
    location : (3) array_like
        source location
    radius : float
        loop radius
    orientation : (2) array_like
        Circular loop normal azimuth and declination
    Imax : float
        Maximum current amplitude
    waveform : SimPEG.electromagnetics.viscous_remanent_magnetization.waveforms.BaseVRMWaveform
        VRM waveform
    """

    def __init__(
        self, receiver_list, location, radius, orientation, Imax, waveform, **kwargs
    ):

        if len(location) != 3:
            raise ValueError(
                "Tx location (x,y,z) must be given as a column vector of length 3."
            )

        if len(orientation) != 2:
            raise ValueError(
                "Circular loop transmitter orientation orientation defined by two angles (theta, alpha)."
            )

        super(CircLoop, self).__init__(receiver_list, location, waveform, **kwargs)

        self.orientation = validate_ndarray_with_shape(
            "orientation", orientation, shape=(2,), dtype=float
        )
        self.radius = validate_float("radius", radius, min_val=0.0, inclusive_min=False)
        self.Imax = validate_float("Imax", Imax)

    def getH0(self, xyz):
        """Compute inducing field at locations xyz

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            locations at which primary field components are computed

        Returns
        -------
        (n, 3) numpy.ndarray
            Primary magnetic field at all xyz locations; organized as columns [Hx0,Hy0,Hz0]
        """

        r0 = self.location
        theta = self.orientation[0]  # Azimuthal
        alpha = self.orientation[1]  # Declination
        a = self.radius
        I = self.Imax

        # Rotate x,y,z into coordinate axis of transmitter loop
        rot_x = np.r_[
            np.c_[1, 0, 0],
            np.c_[0, np.cos(np.pi * theta / 180), -np.sin(np.pi * theta / 180)],
            np.c_[0, np.sin(np.pi * theta / 180), np.cos(np.pi * theta / 180)],
        ]  # CCW ROTATION OF THETA AROUND X-AXIS

        rot_z = np.r_[
            np.c_[np.cos(np.pi * alpha / 180), -np.sin(np.pi * alpha / 180), 0],
            np.c_[np.sin(np.pi * alpha / 180), np.cos(np.pi * alpha / 180), 0],
            np.c_[0, 0, 1],
        ]  # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS

        rot_mat = np.dot(rot_x, rot_z)  # THE ORDER MATTERS

        x1p = np.dot(
            np.c_[xyz[:, 0] - r0[0], xyz[:, 1] - r0[1], xyz[:, 2] - r0[2]],
            rot_mat[0, :].T,
        )
        x2p = np.dot(
            np.c_[xyz[:, 0] - r0[0], xyz[:, 1] - r0[1], xyz[:, 2] - r0[2]],
            rot_mat[1, :].T,
        )
        x3p = np.dot(
            np.c_[xyz[:, 0] - r0[0], xyz[:, 1] - r0[1], xyz[:, 2] - r0[2]],
            rot_mat[2, :].T,
        )

        s = np.sqrt(x1p ** 2 + x2p ** 2) + 1e-10  # Radial distance
        k = 4 * a * s / (x3p ** 2 + (a + s) ** 2)

        hxp = (
            (x1p / s)
            * (x3p * I / (2 * np.pi * s * np.sqrt(x3p ** 2 + (a + s) ** 2)))
            * (
                ((a ** 2 + x3p ** 2 + s ** 2) / (x3p ** 2 + (s - a) ** 2))
                * spec.ellipe(k)
                - spec.ellipk(k)
            )
        )
        hyp = (
            (x2p / s)
            * (x3p * I / (2 * np.pi * s * np.sqrt(x3p ** 2 + (a + s) ** 2)))
            * (
                ((a ** 2 + x3p ** 2 + s ** 2) / (x3p ** 2 + (s - a) ** 2))
                * spec.ellipe(k)
                - spec.ellipk(k)
            )
        )
        hzp = (I / (2 * np.pi * np.sqrt(x3p ** 2 + (a + s) ** 2))) * (
            ((a ** 2 - x3p ** 2 - s ** 2) / (x3p ** 2 + (s - a) ** 2)) * spec.ellipe(k)
            + spec.ellipk(k)
        )

        # Rotate the other way to get back into original coordinates
        rot_mat_t = rot_mat.T
        hx0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[0, :].T)
        hy0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[1, :].T)
        hz0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[2, :].T)

        return np.c_[hx0, hy0, hz0]

    def _getRefineFlags(self, xyzc, refinement_factor, refinement_distance):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        refinement_factor -- Refinement factors

        refinement_distance -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        refFlag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        r0 = self.location
        a = self.radius
        theta = self.orientation[0]  # Azimuthal
        alpha = self.orientation[1]  # Declination

        # Rotate x,y,z into coordinate axis of transmitter loop
        rot_x = np.r_[
            np.c_[1, 0, 0],
            np.c_[0, np.cos(np.pi * theta / 180), -np.sin(np.pi * theta / 180)],
            np.c_[0, np.sin(np.pi * theta / 180), np.cos(np.pi * theta / 180)],
        ]  # CCW ROTATION OF THETA AROUND X-AXIS

        rot_z = np.r_[
            np.c_[np.cos(np.pi * alpha / 180), -np.sin(np.pi * alpha / 180), 0],
            np.c_[np.sin(np.pi * alpha / 180), np.cos(np.pi * alpha / 180), 0],
            np.c_[0, 0, 1],
        ]  # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS

        rot_mat = np.dot(rot_x, rot_z)  # THE ORDER MATTERS

        x1p = np.dot(
            np.c_[xyzc[:, 0] - r0[0], xyzc[:, 1] - r0[1], xyzc[:, 2] - r0[2]],
            rot_mat[0, :].T,
        )
        x2p = np.dot(
            np.c_[xyzc[:, 0] - r0[0], xyzc[:, 1] - r0[1], xyzc[:, 2] - r0[2]],
            rot_mat[1, :].T,
        )
        x3p = np.dot(
            np.c_[xyzc[:, 0] - r0[0], xyzc[:, 1] - r0[1], xyzc[:, 2] - r0[2]],
            rot_mat[2, :].T,
        )
        r = np.sqrt(x1p ** 2 + x2p ** 2 + x3p ** 2)
        cosA = np.sqrt(x1p ** 2 + x2p ** 2) / r
        d = np.sqrt(a ** 2 + r ** 2 - 2 * a * r * cosA)

        for nn in range(0, refinement_factor):

            k = (d < refinement_distance[nn] + 1e-3) & (
                refFlag < refinement_factor - nn + 1
            )
            refFlag[k] = refinement_factor - nn

        return refFlag


#########################################
# LINE CURRENT VRM SOURCE CLASS
#########################################


class LineCurrent(BaseSrcVRM):
    """Line current source.

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.time_domain.receivers.BaseRx
        List of TDEM receivers
    location : (n, 3) numpy.ndarray
        Array defining the node locations for the wire path. For inductive sources,
        you must close the loop.
    Imax : float
        Maximum current amplitude
    waveform : SimPEG.electromagnetics.viscous_remanent_magnetization.waveforms.BaseVRMWaveform
        VRM waveform
    """

    # location = properties.Array("location of the source wire points", shape=("*", 3))

    def __init__(self, receiver_list, location, Imax, waveform, **kwargs):

        super(LineCurrent, self).__init__(receiver_list, location, waveform, **kwargs)

        self.Imax = validate_float("Imax", Imax)

    @property
    def location(self):
        """Line current nodes locations

        Returns
        -------
        (n, 3) numpy.ndarray
            Line current node locations.
        """
        return self._location

    @location.setter
    def location(self, loc):
        self._location = loc = validate_ndarray_with_shape(
            "location", loc, shape=("*", 3)
        )

    def getH0(self, xyz):
        """Compute inducing field at locations xyz

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray
            locations at which primary field components are computed

        Returns
        -------
        (n, 3) numpy.ndarray
            Primary magnetic field at all xyz locations; organized as columns [Hx0,Hy0,Hz0]
        """

        # TRANSMITTER NODES
        I = self.Imax
        tx_nodes = self.location
        x1tr = tx_nodes[:, 0]
        x2tr = tx_nodes[:, 1]
        x3tr = tx_nodes[:, 2]

        nLoc = np.shape(xyz)[0]
        nSeg = np.size(x1tr) - 1

        hx0 = np.zeros(nLoc)
        hy0 = np.zeros(nLoc)
        hz0 = np.zeros(nLoc)

        for pp in range(0, nSeg):

            # Wire ends for transmitter wire pp
            x1a = x1tr[pp]
            x2a = x2tr[pp]
            x3a = x3tr[pp]
            x1b = x1tr[pp + 1]
            x2b = x2tr[pp + 1]
            x3b = x3tr[pp + 1]

            # Vector Lengths between points
            vab = np.sqrt((x1b - x1a) ** 2 + (x2b - x2a) ** 2 + (x3b - x3a) ** 2)
            vap = np.sqrt(
                (xyz[:, 0] - x1a) ** 2 + (xyz[:, 1] - x2a) ** 2 + (xyz[:, 2] - x3a) ** 2
            )
            vbp = np.sqrt(
                (xyz[:, 0] - x1b) ** 2 + (xyz[:, 1] - x2b) ** 2 + (xyz[:, 2] - x3b) ** 2
            )

            # Cosines from cos()=<v1,v2>/(|v1||v2|)
            cos_alpha = (
                (xyz[:, 0] - x1a) * (x1b - x1a)
                + (xyz[:, 1] - x2a) * (x2b - x2a)
                + (xyz[:, 2] - x3a) * (x3b - x3a)
            ) / (vap * vab)
            cos_beta = (
                (xyz[:, 0] - x1b) * (x1a - x1b)
                + (xyz[:, 1] - x2b) * (x2a - x2b)
                + (xyz[:, 2] - x3b) * (x3a - x3b)
            ) / (vbp * vab)

            # Determining Radial Vector From Wire
            dot_temp = (
                (x1a - xyz[:, 0]) * (x1b - x1a)
                + (x2a - xyz[:, 1]) * (x2b - x2a)
                + (x3a - xyz[:, 2]) * (x3b - x3a)
            )

            rx1 = (x1a - xyz[:, 0]) - dot_temp * (x1b - x1a) / vab ** 2
            rx2 = (x2a - xyz[:, 1]) - dot_temp * (x2b - x2a) / vab ** 2
            rx3 = (x3a - xyz[:, 2]) - dot_temp * (x3b - x3a) / vab ** 2

            r = np.sqrt(rx1 ** 2 + rx2 ** 2 + rx3 ** 2)

            phi = (cos_alpha + cos_beta) / r

            # I/4*pi in each direction
            ix1 = I * (x1b - x1a) / (4 * np.pi * vab)
            ix2 = I * (x2b - x2a) / (4 * np.pi * vab)
            ix3 = I * (x3b - x3a) / (4 * np.pi * vab)

            # Add contribution from wire pp into array
            hx0 = hx0 + phi * (-ix2 * rx3 + ix3 * rx2) / r
            hy0 = hy0 + phi * (ix1 * rx3 - ix3 * rx1) / r
            hz0 = hz0 + phi * (-ix1 * rx2 + ix2 * rx1) / r

        return np.c_[hx0, hy0, hz0]

    def _getRefineFlags(self, xyzc, refinement_factor, refinement_distance):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        refinement_factor -- Refinement factors

        refinement_distance -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        ref_flag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        nSeg = np.shape(self.location)[0] - 1

        for tt in range(0, nSeg):

            ref_flag_tt = np.zeros(np.shape(xyzc)[0], dtype=np.int)
            tx0 = self.location[tt, :]
            tx1 = self.location[tt + 1, :]
            a = (tx1[0] - tx0[0]) ** 2 + (tx1[1] - tx0[1]) ** 2 + (tx1[2] - tx0[2]) ** 2
            b = (
                2 * (tx1[0] - tx0[0]) * (tx0[0] - xyzc[:, 0])
                + 2 * (tx1[1] - tx0[1]) * (tx0[1] - xyzc[:, 1])
                + 2 * (tx1[2] - tx0[2]) * (tx0[2] - xyzc[:, 2])
            )

            for nn in range(0, refinement_factor):

                d = refinement_distance[nn] + 1e-3
                c = (
                    (tx0[0] - xyzc[:, 0]) ** 2
                    + (tx0[1] - xyzc[:, 1]) ** 2
                    + (tx0[2] - xyzc[:, 2]) ** 2
                    - d ** 2
                )
                e = np.array(b ** 2 - 4 * a * c, dtype=np.complex)

                q_pos = (-b + np.sqrt(e)) / (2 * a)
                q_neg = (-b - np.sqrt(e)) / (2 * a)

                k_pos = (
                    (np.abs(np.imag(q_pos)) > 1e-12)
                    | ((np.real(q_pos) < 0.0) & (np.real(q_neg) < 0.0))
                    | ((np.real(q_pos) > 1.0) & (np.real(q_neg) > 1.0))
                )
                k_neg = (
                    (np.abs(np.imag(q_pos)) > 1e-12)
                    | ((np.real(q_pos) < 0.0) & (np.real(q_neg) < 0.0))
                    | ((np.real(q_pos) > 1.0) & (np.real(q_neg) > 1.0))
                    | (k_pos)
                )

                ind = (
                    (k_pos == False)
                    & (k_neg == False)
                    & (ref_flag_tt < refinement_factor + 1 - nn)
                )
                ref_flag_tt[ind] = refinement_factor - nn

            ref_flag = np.maximum(ref_flag, ref_flag_tt)

        return ref_flag
