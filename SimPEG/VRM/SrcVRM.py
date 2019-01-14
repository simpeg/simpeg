import numpy as np
import scipy.special as spec
from SimPEG import Survey
from .RxVRM import BaseRxVRM
from .WaveformVRM import StepOff, SquarePulse, ArbitraryDiscrete, ArbitraryPiecewise

#########################################
# BASE VRM SOURCE CLASS
#########################################


class BaseSrcVRM(Survey.BaseSrc):
    """SimPEG Source Object"""

    def __init__(self, rxList, waveform, **kwargs):

        if isinstance(waveform, (StepOff, SquarePulse, ArbitraryDiscrete, ArbitraryPiecewise)) == False:
            AttributeError("Waveform must be an instance of a VRM waveform class: StepOff, SquarePulse or Arbitrary")

        super(BaseSrcVRM, self).__init__(rxList, **kwargs)
        self.waveform = waveform
        self.rxPair = BaseRxVRM  # Links base Src class to acceptable Rx class?

    @property
    def nRx(self):
        """Total number of receiver locations"""
        return np.sum(np.array([np.shape(rx.locs)[0] for rx in self.rxList]))

    @property
    def vnRx(self):
        """Vector number of receiver locations"""
        return np.array([np.shape(rx.locs)[0] for rx in self.rxList])

    @property
    def vnD(self):
        """Vector number of data"""
        return np.array([rx.nD for rx in self.rxList])


#########################################
# MAGNETIC DIPOLE VRM SOURCE CLASS
#########################################

class MagDipole(BaseSrcVRM):

    """

    """

    def __init__(self, rxList, loc, moment, waveform, **kwargs):

        if len(loc) != 3:
            raise ValueError('Tx location (x,y,z) must be given as a column vector of length 3.')

        if len(moment) != 3:
            raise ValueError('Dipole moment (mx,my,mz) must be given as a column vector of length 3.')

        super(MagDipole, self).__init__(rxList, waveform, **kwargs)

        self.loc = loc
        self.moment = moment

    def getH0(self, xyz):

        """
        Computes inducing field at locations xyz

        REQUIRED ARGUMENTS:

        xyz -- N X 3 array of locations at which primary field components
        are computed

        OUTPUTS:

        H0 -- N X 3 array containing [Hx0,Hy0,Hz0] at all xyz locations

        """

        m = self.moment
        r0 = self.loc

        r = np.sqrt((xyz[:, 0]-r0[0])**2 + (xyz[:, 1]-r0[1])**2 + (xyz[:, 2]-r0[2])**2)
        mdotr = m[0]*(xyz[:, 0]-r0[0]) + m[1]*(xyz[:, 1]-r0[1]) + m[2]*(xyz[:, 2]-r0[2])

        hx0 = (1/(4*np.pi))*(3*(xyz[:, 0]-r0[0])*mdotr/r**5 - m[0]/r**3)
        hy0 = (1/(4*np.pi))*(3*(xyz[:, 1]-r0[1])*mdotr/r**5 - m[1]/r**3)
        hz0 = (1/(4*np.pi))*(3*(xyz[:, 2]-r0[2])*mdotr/r**5 - m[2]/r**3)

        return np.c_[hx0, hy0, hz0]

    def _getRefineFlags(self, xyzc, ref_factor, ref_radius):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        ref_factor -- Refinement factors

        ref_radius -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        refFlag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        r = np.sqrt(
            (xyzc[:, 0] - self.loc[0])**2 +
            (xyzc[:, 1] - self.loc[1])**2 +
            (xyzc[:, 2] - self.loc[2])**2
        )

        for nn in range(0, ref_factor):

            k = (r < ref_radius[nn]+1e-5) & (refFlag < ref_factor-nn+1)
            refFlag[k] = ref_factor - nn

        return refFlag


#########################################
# CIRCULAR LOOP VRM SOURCE CLASS
#########################################

class CircLoop(BaseSrcVRM):

    """

    """

    def __init__(self, rxList, loc, radius, orientation, Imax, waveform, **kwargs):

        if len(loc) != 3:
            raise ValueError('Tx location (x,y,z) must be given as a column vector of length 3.')

        if len(orientation) != 2:
            raise ValueError('Circular loop transmitter orientation orientation defined by two angles (theta, alpha).')

        super(CircLoop, self).__init__(rxList, waveform, **kwargs)

        self.loc = loc
        self.orientation = orientation
        self.radius = radius
        self.Imax = Imax

    def getH0(self, xyz):

        """
        Computes inducing field at locations xyz

        REQUIRED ARGUMENTS:

        xyz -- N X 3 array of locations at which primary field components
        are computed

        OUTPUTS:

        H0 -- N X 3 array containing [hx0,hy0,hz0] at all xyz locations

        """

        r0 = self.loc
        theta = self.orientation[0]  # Azimuthal
        alpha = self.orientation[1]  # Declination
        a = self.radius
        I = self.Imax

        # Rotate x,y,z into coordinate axis of transmitter loop
        rot_x = np.r_[
            np.c_[1, 0, 0],
            np.c_[0, np.cos(np.pi*theta/180), -np.sin(np.pi*theta/180)],
            np.c_[0, np.sin(np.pi*theta/180), np.cos(np.pi*theta/180)]
        ]     # CCW ROTATION OF THETA AROUND X-AXIS

        rot_z = np.r_[
            np.c_[np.cos(np.pi*alpha/180), -np.sin(np.pi*alpha/180), 0],
            np.c_[np.sin(np.pi*alpha/180), np.cos(np.pi*alpha/180), 0],
            np.c_[0, 0, 1]
        ]     # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS

        rot_mat = np.dot(rot_x, rot_z)            # THE ORDER MATTERS

        x1p = np.dot(np.c_[xyz[:, 0]-r0[0], xyz[:, 1]-r0[1], xyz[:, 2]-r0[2]], rot_mat[0, :].T)
        x2p = np.dot(np.c_[xyz[:, 0]-r0[0], xyz[:, 1]-r0[1], xyz[:, 2]-r0[2]], rot_mat[1, :].T)
        x3p = np.dot(np.c_[xyz[:, 0]-r0[0], xyz[:, 1]-r0[1], xyz[:, 2]-r0[2]], rot_mat[2, :].T)

        s = np.sqrt(x1p**2 + x2p**2) + 1e-10     # Radial distance
        k = 4*a*s/(x3p**2 + (a+s)**2)

        hxp = (x1p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        hyp = (x2p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        hzp =         (    I/(2*np.pi*  np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 - x3p**2 - s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) + spec.ellipk(k))

        # Rotate the other way to get back into original coordinates
        rot_mat_t = rot_mat.T
        hx0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[0, :].T)
        hy0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[1, :].T)
        hz0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[2, :].T)

        return np.c_[hx0, hy0, hz0]

    def _getRefineFlags(self, xyzc, ref_factor, ref_radius):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        ref_factor -- Refinement factors

        ref_radius -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        refFlag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        r0 = self.loc
        a = self.radius
        theta = self.orientation[0]  # Azimuthal
        alpha = self.orientation[1]  # Declination

        # Rotate x,y,z into coordinate axis of transmitter loop
        rot_x = np.r_[
            np.c_[1, 0, 0],
            np.c_[0, np.cos(np.pi*theta/180), -np.sin(np.pi*theta/180)],
            np.c_[0, np.sin(np.pi*theta/180), np.cos(np.pi*theta/180)]
        ]     # CCW ROTATION OF THETA AROUND X-AXIS

        rot_z = np.r_[
            np.c_[np.cos(np.pi*alpha/180), -np.sin(np.pi*alpha/180), 0],
            np.c_[np.sin(np.pi*alpha/180), np.cos(np.pi*alpha/180), 0],
            np.c_[0, 0, 1]
        ]     # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS

        rot_mat = np.dot(rot_x, rot_z)            # THE ORDER MATTERS

        x1p = np.dot(np.c_[xyzc[:, 0]-r0[0], xyzc[:, 1]-r0[1], xyzc[:, 2]-r0[2]], rot_mat[0, :].T)
        x2p = np.dot(np.c_[xyzc[:, 0]-r0[0], xyzc[:, 1]-r0[1], xyzc[:, 2]-r0[2]], rot_mat[1, :].T)
        x3p = np.dot(np.c_[xyzc[:, 0]-r0[0], xyzc[:, 1]-r0[1], xyzc[:, 2]-r0[2]], rot_mat[2, :].T)
        r = np.sqrt(x1p**2 + x2p**2 + x3p**2)
        cosA = np.sqrt(x1p**2 + x2p**2)/r
        d = np.sqrt(a**2 + r**2 - 2*a*r*cosA)

        for nn in range(0, ref_factor):

            k = (d < ref_radius[nn]+1e-3) & (refFlag < ref_factor-nn+1)
            refFlag[k] = ref_factor - nn

        return refFlag


#########################################
# LINE CURRENT VRM SOURCE CLASS
#########################################

class LineCurrent(BaseSrcVRM):

    """

    """

    def __init__(self, rxList, loc, Imax, waveform, **kwargs):

        if np.shape(loc)[1] != 3:
            raise ValueError("Attribute 'loc' must be np.array(N+1,3) where N is the number of line segments.")

        if np.shape(loc)[0] < 4:
            raise ValueError("Attribute 'loc' must be np.array(N+1,3) where N is the number of line segments.")

        self.loc = loc
        self.Imax = Imax

        super(LineCurrent, self).__init__(rxList, waveform, **kwargs)

    def getH0(self, xyz):

        """
        Computes inducing field at locations xyz

        REQUIRED ARGUMENTS:

        xyz -- N X 3 array of locations at which primary field components
        are computed

        OUTPUTS:

        H0 -- N X 3 array containing [Hx0,Hy0,Hz0] at all xyz locations

        """

        # TRANSMITTER NODES
        I = self.Imax
        tx_nodes = self.loc
        x1tr = tx_nodes[:, 0]
        x2tr = tx_nodes[:, 1]
        x3tr = tx_nodes[:, 2]

        nLoc = np.shape(xyz)[0]
        nSeg = np.size(x1tr)-1

        hx0 = np.zeros(nLoc)
        hy0 = np.zeros(nLoc)
        hz0 = np.zeros(nLoc)

        for pp in range(0, nSeg):

            # Wire ends for transmitter wire pp
            x1a = x1tr[pp]
            x2a = x2tr[pp]
            x3a = x3tr[pp]
            x1b = x1tr[pp+1]
            x2b = x2tr[pp+1]
            x3b = x3tr[pp+1]

            # Vector Lengths between points
            vab = np.sqrt((x1b - x1a)**2 + (x2b - x2a)**2 + (x3b - x3a)**2)
            vap = np.sqrt((xyz[:, 0] - x1a)**2 + (xyz[:, 1] - x2a)**2 + (xyz[:, 2] - x3a)**2)
            vbp = np.sqrt((xyz[:, 0] - x1b)**2 + (xyz[:, 1] - x2b)**2 + (xyz[:, 2] - x3b)**2)

            # Cosines from cos()=<v1,v2>/(|v1||v2|)
            cos_alpha = ((xyz[:, 0]-x1a)*(x1b - x1a) + (xyz[:, 1]-x2a)*(x2b - x2a) + (xyz[:, 2]-x3a)*(x3b - x3a))/(vap*vab)
            cos_beta  = ((xyz[:, 0]-x1b)*(x1a - x1b) + (xyz[:, 1]-x2b)*(x2a - x2b) + (xyz[:, 2]-x3b)*(x3a - x3b))/(vbp*vab)

            # Determining Radial Vector From Wire
            dot_temp = (
                (x1a - xyz[:, 0])*(x1b - x1a) +
                (x2a - xyz[:, 1])*(x2b - x2a) +
                (x3a - xyz[:, 2])*(x3b - x3a)
            )

            rx1 = (x1a - xyz[:, 0]) - dot_temp*(x1b - x1a)/vab**2
            rx2 = (x2a - xyz[:, 1]) - dot_temp*(x2b - x2a)/vab**2
            rx3 = (x3a - xyz[:, 2]) - dot_temp*(x3b - x3a)/vab**2

            r = np.sqrt(rx1**2 + rx2**2 + rx3**2)

            phi = (cos_alpha + cos_beta)/r

            # I/4*pi in each direction
            ix1 = I*(x1b - x1a)/(4*np.pi*vab)
            ix2 = I*(x2b - x2a)/(4*np.pi*vab)
            ix3 = I*(x3b - x3a)/(4*np.pi*vab)

            # Add contribution from wire pp into array
            hx0 = hx0 + phi*(-ix2*rx3 + ix3*rx2)/r
            hy0 = hy0 + phi*( ix1*rx3 - ix3*rx1)/r
            hz0 = hz0 + phi*(-ix1*rx2 + ix2*rx1)/r

        return np.c_[hx0, hy0, hz0]

    def _getRefineFlags(self, xyzc, ref_factor, ref_radius):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        ref_factor -- Refinement factors

        ref_radius -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        ref_flag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        nSeg = np.shape(self.loc)[0] - 1

        for tt in range(0, nSeg):

            ref_flag_tt = np.zeros(np.shape(xyzc)[0], dtype=np.int)
            tx0 = self.loc[tt, :]
            tx1 = self.loc[tt+1, :]
            a = (tx1[0] - tx0[0])**2 + (tx1[1] - tx0[1])**2 + (tx1[2] - tx0[2])**2
            b = (
                2*(tx1[0] - tx0[0])*(tx0[0] - xyzc[:, 0]) +
                2*(tx1[1] - tx0[1])*(tx0[1] - xyzc[:, 1]) +
                2*(tx1[2] - tx0[2])*(tx0[2] - xyzc[:, 2])
            )

            for nn in range(0, ref_factor):

                d = ref_radius[nn] + 1e-3
                c = (
                    (tx0[0] - xyzc[:, 0])**2 +
                    (tx0[1] - xyzc[:, 1])**2 +
                    (tx0[2] - xyzc[:, 2])**2 -
                    d**2
                )
                e = np.array(b**2 - 4*a*c, dtype=np.complex)

                q_pos = (-b + np.sqrt(e))/(2*a)
                q_neg = (-b - np.sqrt(e))/(2*a)

                k_pos = (
                    (np.abs(np.imag(q_pos)) > 1e-12) |
                    ((np.real(q_pos) < 0.) & (np.real(q_neg) < 0.)) |
                    ((np.real(q_pos) > 1.) & (np.real(q_neg) > 1.))
                )
                k_neg = (
                    (np.abs(np.imag(q_pos)) > 1e-12) |
                    ((np.real(q_pos) < 0.) & (np.real(q_neg) < 0.)) |
                    ((np.real(q_pos) > 1.) & (np.real(q_neg) > 1.)) |
                    (k_pos == True)
                )

                ind = (k_pos == False) & (k_neg == False) & (ref_flag_tt < ref_factor+1-nn)
                ref_flag_tt[ind] = ref_factor - nn

            ref_flag = np.maximum(ref_flag, ref_flag_tt)

        return ref_flag
