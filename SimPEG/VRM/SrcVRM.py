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

        assert isinstance(waveform, (StepOff, SquarePulse, ArbitraryDiscrete, ArbitraryPiecewise)), "waveform must be an instance of a VRM waveform class: StepOff, SquarePulse or Arbitrary"

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

        assert len(loc) is 3, 'Tx location must be given as a column vector np.r_[x,y,z]'
        assert len(moment) is 3, 'Dipole moment given as column vector np.r_[mx,my,mz]'
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

        R = np.sqrt((xyz[:, 0]-r0[0])**2 + (xyz[:, 1]-r0[1])**2 + (xyz[:, 2]-r0[2])**2)
        mdotr = m[0]*(xyz[:, 0]-r0[0]) + m[1]*(xyz[:, 1]-r0[1]) + m[2]*(xyz[:, 2]-r0[2])

        Hx0 = (1/(4*np.pi))*(3*(xyz[:, 0]-r0[0])*mdotr/R**5 - m[0]/R**3)
        Hy0 = (1/(4*np.pi))*(3*(xyz[:, 1]-r0[1])*mdotr/R**5 - m[1]/R**3)
        Hz0 = (1/(4*np.pi))*(3*(xyz[:, 2]-r0[2])*mdotr/R**5 - m[2]/R**3)

        return np.c_[Hx0, Hy0, Hz0]

    def _getRefineFlags(self, xyzc, refFact, refRadius):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        refFact -- Refinement factors

        refRadius -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        refFlag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        R = np.sqrt((xyzc[:, 0] - self.loc[0])**2 + (xyzc[:, 1] - self.loc[1])**2 + (xyzc[:, 2] - self.loc[2])**2)

        for nn in range(0, refFact):

            k = (R < refRadius[nn]+1e-5) & (refFlag < refFact-nn+1)
            refFlag[k] = refFact - nn

        return refFlag


#########################################
# CIRCULAR LOOP VRM SOURCE CLASS
#########################################

class CircLoop(BaseSrcVRM):

    """

    """

    def __init__(self, rxList, loc, radius, orientation, Imax, waveform, **kwargs):

        assert len(loc) is 3, 'Tx location must be given as a column vector np.r[x,y,z]'
        assert len(orientation) is 2, 'Two angles (theta, alpha) required to define orientation'
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

        H0 -- N X 3 array containing [Hx0,Hy0,Hz0] at all xyz locations

        """

        r0 = self.loc
        theta = self.orientation[0]  # Azimuthal
        alpha = self.orientation[1]  # Declination
        a = self.radius
        I = self.Imax

        # Rotate x,y,z into coordinate axis of transmitter loop
        Rx = np.r_[np.c_[1, 0, 0], np.c_[0, np.cos(np.pi*theta/180), -np.sin(np.pi*theta/180)], np.c_[0, np.sin(np.pi*theta/180), np.cos(np.pi*theta/180)]]     # CCW ROTATION OF THETA AROUND X-AXIS
        Rz = np.r_[np.c_[np.cos(np.pi*alpha/180), -np.sin(np.pi*alpha/180), 0], np.c_[np.sin(np.pi*alpha/180), np.cos(np.pi*alpha/180), 0], np.c_[0, 0, 1]]     # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS
        R = np.dot(Rx, Rz)            # THE ORDER MATTERS

        x1p = np.dot(np.c_[xyz[:, 0]-r0[0], xyz[:, 1]-r0[1], xyz[:, 2]-r0[2]],R[0, :].T)
        x2p = np.dot(np.c_[xyz[:, 0]-r0[0], xyz[:, 1]-r0[1], xyz[:, 2]-r0[2]],R[1, :].T)
        x3p = np.dot(np.c_[xyz[:, 0]-r0[0], xyz[:, 1]-r0[1], xyz[:, 2]-r0[2]],R[2, :].T)

        s = np.sqrt(x1p**2 + x2p**2) + 1e-10     # Radial distance
        k = 4*a*s/(x3p**2 + (a+s)**2)

        Hxp = (x1p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        Hyp = (x2p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        Hzp =         (    I/(2*np.pi*  np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 - x3p**2 - s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) + spec.ellipk(k))

        # Rotate the other way to get back into original coordinates
        Rp = R.T
        Hx0 = np.dot(np.c_[Hxp, Hyp, Hzp], Rp[0, :].T)
        Hy0 = np.dot(np.c_[Hxp, Hyp, Hzp], Rp[1, :].T)
        Hz0 = np.dot(np.c_[Hxp, Hyp, Hzp], Rp[2, :].T)

        return np.c_[Hx0, Hy0, Hz0]

    def _getRefineFlags(self, xyzc, refFact, refRadius):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        refFact -- Refinement factors

        refRadius -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        refFlag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        r0 = self.loc
        a = self.radius
        theta = self.orientation[0]  # Azimuthal
        alpha = self.orientation[1]  # Declination

        # Rotate x,y,z into coordinate axis of transmitter loop
        Rx = np.r_[np.c_[1, 0, 0], np.c_[0, np.cos(np.pi*theta/180), -np.sin(np.pi*theta/180)], np.c_[0, np.sin(np.pi*theta/180), np.cos(np.pi*theta/180)]]     # CCW ROTATION OF THETA AROUND X-AXIS
        Rz = np.r_[np.c_[np.cos(np.pi*alpha/180), -np.sin(np.pi*alpha/180), 0], np.c_[np.sin(np.pi*alpha/180), np.cos(np.pi*alpha/180), 0], np.c_[0, 0, 1]]     # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS
        R = np.dot(Rx, Rz)            # THE ORDER MATTERS

        x1p = np.dot(np.c_[xyzc[:, 0]-r0[0], xyzc[:, 1]-r0[1], xyzc[:, 2]-r0[2]], R[0, :].T)
        x2p = np.dot(np.c_[xyzc[:, 0]-r0[0], xyzc[:, 1]-r0[1], xyzc[:, 2]-r0[2]], R[1, :].T)
        x3p = np.dot(np.c_[xyzc[:, 0]-r0[0], xyzc[:, 1]-r0[1], xyzc[:, 2]-r0[2]], R[2, :].T)
        R = np.sqrt(x1p**2 + x2p**2 + x3p**2)
        cosA = np.sqrt(x1p**2 + x2p**2)/R
        D = np.sqrt(a**2 + R**2 - 2*a*R*cosA)

        for nn in range(0, refFact):

            k = (D < refRadius[nn]+1e-3) & (refFlag < refFact-nn+1)
            refFlag[k] = refFact - nn

        return refFlag


#########################################
# LINE CURRENT VRM SOURCE CLASS
#########################################

class LineCurrent(BaseSrcVRM):

    """

    """

    def __init__(self, rxList, loc, Imax, waveform, **kwargs):

        assert np.shape(loc)[1] == 3 and np.shape(loc)[0] > 1, 'locs is a N+1 by 3 array where N is the number of transmitter segments'

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
        TxNodes = self.loc
        x1tr = TxNodes[:, 0]
        x2tr = TxNodes[:, 1]
        x3tr = TxNodes[:, 2]

        M = np.shape(xyz)[0]
        N = np.size(x1tr)-1

        Hx0 = np.zeros(M)
        Hy0 = np.zeros(M)
        Hz0 = np.zeros(M)

        for pp in range(0, N):

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
            CosAlpha = ((xyz[:,0]-x1a)*(x1b - x1a) + (xyz[:, 1]-x2a)*(x2b - x2a) + (xyz[:, 2]-x3a)*(x3b - x3a))/(vap*vab)
            CosBeta  = ((xyz[:,0]-x1b)*(x1a - x1b) + (xyz[:, 1]-x2b)*(x2a - x2b) + (xyz[:, 2]-x3b)*(x3a - x3b))/(vbp*vab)

            # Determining Radial Vector From Wire
            DotTemp = (x1a - xyz[:, 0])*(x1b - x1a) + (x2a - xyz[:, 1])*(x2b - x2a) + (x3a - xyz[:, 2])*(x3b - x3a)

            Rx1 = (x1a - xyz[:, 0]) - DotTemp*(x1b - x1a)/vab**2
            Rx2 = (x2a - xyz[:, 1]) - DotTemp*(x2b - x2a)/vab**2
            Rx3 = (x3a - xyz[:, 2]) - DotTemp*(x3b - x3a)/vab**2

            R = np.sqrt(Rx1**2 + Rx2**2 + Rx3**2)

            Phi = (CosAlpha + CosBeta)/R

            # I/4*pi in each direction
            Ix1 = I*(x1b - x1a)/(4*np.pi*vab)
            Ix2 = I*(x2b - x2a)/(4*np.pi*vab)
            Ix3 = I*(x3b - x3a)/(4*np.pi*vab)

            # Add contribution from wire pp into array
            Hx0 = Hx0 + Phi*(-Ix2*Rx3 + Ix3*Rx2)/R
            Hy0 = Hy0 + Phi*( Ix1*Rx3 - Ix3*Rx1)/R
            Hz0 = Hz0 + Phi*(-Ix1*Rx2 + Ix2*Rx1)/R

        return np.c_[Hx0, Hy0, Hz0]

    def _getRefineFlags(self, xyzc, refFact, refRadius):

        """
        This function finds the refinement factor to be assigned to each cell

        REQUIRED ARGUMENTS:

        xyzc -- Cell-center locations as NX3 array

        refFact -- Refinement factors

        refRadius -- Refinement radii

        OUTPUTS:

        refFlag -- Vector of length N with the refinement factor for each cell

        """

        refFlag = np.zeros(np.shape(xyzc)[0], dtype=np.int)

        nSeg = np.shape(self.loc)[0] - 1

        for tt in range(0, nSeg):

            refFlagtt = np.zeros(np.shape(xyzc)[0], dtype=np.int)
            Tx0 = self.loc[tt, :]
            Tx1 = self.loc[tt+1, :]
            A = (Tx1[0] - Tx0[0])**2 + (Tx1[1] - Tx0[1])**2 + (Tx1[2] - Tx0[2])**2
            B = 2*(Tx1[0] - Tx0[0])*(Tx0[0] - xyzc[:, 0]) + 2*(Tx1[1] - Tx0[1])*(Tx0[1] - xyzc[:, 1]) + 2*(Tx1[2] - Tx0[2])*(Tx0[2] - xyzc[:, 2])

            for nn in range(0, refFact):

                D = refRadius[nn] + 1e-3
                C = (Tx0[0] - xyzc[:, 0])**2 + (Tx0[1] - xyzc[:, 1])**2 + (Tx0[2] - xyzc[:, 2])**2 - D**2
                E = np.array(B**2 - 4*A*C, dtype=np.complex)

                Qpos = (-B + np.sqrt(E))/(2*A)
                Qneg = (-B - np.sqrt(E))/(2*A)

                kpos = (np.abs(np.imag(Qpos)) > 1e-12) | ((np.real(Qpos) < 0.) & (np.real(Qneg) < 0.)) | ((np.real(Qpos) > 1.) & (np.real(Qneg) > 1.))
                kneg = (np.abs(np.imag(Qpos)) > 1e-12) | ((np.real(Qpos) < 0.) & (np.real(Qneg) < 0.)) | ((np.real(Qpos) > 1.) & (np.real(Qneg) > 1.)) | (kpos == True)

                refFlagtt[(kpos == False) & (kneg == False) & (refFlagtt < refFact+1-nn)] = refFact - nn

            refFlag = np.maximum(refFlag, refFlagtt)

        return refFlag
