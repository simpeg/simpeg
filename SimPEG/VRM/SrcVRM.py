import numpy as np
import scipy.sparse as sp
import scipy.special as spec
from SimPEG import Props, Utils, Survey
from RxVRM import BaseRxVRM







#########################################
# VRM WAVEFORM CLASS
#########################################

# class waveformVRM():











#########################################
# BASE VRM SOURCE CLASS
#########################################

class BaseSrcVRM(Survey.BaseSrc):
    """SimPEG Source Object"""

    def __init__(self, rxList, **kwargs):
        super(BaseSrcVRM, self).__init__(rxList, **kwargs)
        self.rxPair = BaseRxVRM # Links base Src class to acceptable Rx class?

    @property
    def nRx(self):
        """Number of data"""
        return len(self.rxList)

    @property
    def nD(self):
        """Vector number of receivers"""
        return np.array([rx.nD for rx in self.rxList])

    


#########################################
# MAGNETIC DIPOLE VRM SOURCE CLASS
#########################################

class MagDipole(BaseSrcVRM):

    def __init__(self, rxList, loc, moment, **kwargs):

        waveform = 'StepOff'

        assert len(loc) is 3, 'Tx location must be given as a column vector np.r[x,y,z]'
        assert len(moment) is 3, 'Dipole moment given as column vector np.r_[mx,my,mz]'
        super(MagDipole, self).__init__(rxList, **kwargs)
        
        self.loc = loc
        self.waveform = waveform
        self.moment = moment

    # COMPUTE INDUCING FIELD OPERATOR FROM SOURCE
    def getH0(self, xyz):

        # INPUTS
        # xyz: [x,y,z] locations as a list at which primary field is computed

        m = self.moment
        r0 = self.loc

        R = np.sqrt((xyz[0]-r0[0])**2 + (xyz[1]-r0[1])**2 + (xyz[2]-r0[2])**2)
        mdotr = m[0]*(xyz[0]-r0[0]) + m[1]*(xyz[1]-r0[1]) + m[2]*(xyz[2]-r0[2])

        Hx0 = (1/(4*np.pi))*(3*(xyz[0]-r0[0])*mdotr/R**5 - m[0]/R**3)
        Hy0 = (1/(4*np.pi))*(3*(xyz[1]-r0[1])*mdotr/R**5 - m[1]/R**3)
        Hz0 = (1/(4*np.pi))*(3*(xyz[2]-r0[2])*mdotr/R**5 - m[2]/R**3)

        Hx0 = sp.diags(Hx0)
        Hy0 = sp.diags(Hy0)
        Hz0 = sp.diags(Hz0)

        H0 = sp.vstack([Hx0,Hy0,Hz0])

        return H0




#########################################
# CIRCULAR LOOP VRM SOURCE CLASS
#########################################

class CircLoop(BaseSrcVRM):

    def __init__(self, rxList, loc, radius, orientation, **kwargs):

        waveform = 'StepOff'
        Imax = 1.

        assert len(loc) is 3, 'Tx location must be given as a column vector np.r[x,y,z]'
        assert len(orientation) is 2, 'Two angles (theta, alpha) required to define orientation'
        super(CircLoop, self).__init__(rxList, **kwargs)
        
        self.loc = loc
        self.waveform = waveform
        self.orientation = orientation
        self.radius = radius
        self.Imax = Imax

    # COMPUTE INDUCING FIELD OPERATOR FROM SOURCE
    def getH0(self, xyz):

        # INPUTS
        # xyz: [x,y,z] locations as a list at which primary field is computed

        r0 = self.loc
        theta = self.orientation[0] # Azimuthal
        alpha = self.orientation[1] # Declination
        a = self.radius
        I = self.Imax

        # Rotate x,y,z into coordinate axis of transmitter loop
        Rx = np.r_[np.c_[1,0,0], np.c_[0,np.cos(np.pi*theta/180), -np.sin(np.pi*theta/180)], np.c_[0,np.sin(np.pi*theta/180),np.cos(np.pi*theta/180)]]     # CCW ROTATION OF THETA AROUND X-AXIS
        Rz = np.r_[np.c_[np.cos(np.pi*alpha/180),-np.sin(np.pi*alpha/180), 0], np.c_[np.sin(np.pi*alpha/180),np.cos(np.pi*alpha/180),0], np.c_[0,0,1]]     # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS
        R = np.dot(Rx,Rz)            # THE ORDER MATTERS
        
        x1p = np.dot(np.c_[xyz[0]-r0[0], xyz[1]-r0[1], xyz[2]-r0[2]],R[0,:].T)
        x2p = np.dot(np.c_[xyz[0]-r0[0], xyz[1]-r0[1], xyz[2]-r0[2]],R[1,:].T)
        x3p = np.dot(np.c_[xyz[0]-r0[0], xyz[1]-r0[1], xyz[2]-r0[2]],R[2,:].T)
        
        s = np.sqrt(x1p**2 + x2p**2)     # Radial distance
        k = 4*a*s/(x3p**2 + (a+s)**2)

        Hxp = (x1p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        Hyp = (x2p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        Hzp =         (    I/(2*np.pi*  np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 - x3p**2 - s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) + spec.ellipk(k))  

        # Rotate the other way to get back into original coordinates
        Rp = R.T 
        Hx0 = np.dot(np.c_[Hxp, Hyp, Hzp],Rp[0,:].T)
        Hy0 = np.dot(np.c_[Hxp, Hyp, Hzp],Rp[1,:].T)
        Hz0 = np.dot(np.c_[Hxp, Hyp, Hzp],Rp[2,:].T)

        Hx0 = sp.diags(Hx0)
        Hy0 = sp.diags(Hy0)
        Hz0 = sp.diags(Hz0)

        H0 = sp.vstack([Hx0,Hy0,Hz0])

        return H0




#########################################
# LINE CURRENT VRM SOURCE CLASS
#########################################

class LineCurrent(BaseSrcVRM):

    def __init__(self, rxList, loc, **kwargs):

        Imax = 1.
        waveform = 'StepOff'

        assert np.shape(loc)[1] == 3 and np.shape(loc)[0] > 1, 'locs is a N+1 by 3 array where N is the number of transmitter segments'
        if waveform is not 'StepOff':
            assert np.shape(waveform)[1] is 2, 'For custom waveforms, must have times and current (N X 2 array)'
        self.loc = loc
        self.Imax = Imax
        self.waveform = waveform
        super(LineCurrent, self).__init__(rxList, **kwargs)


    # COMPUTE INDUCING FIELD OPERATOR FROM SOURCE
    def getH0(self, xyz):

        # INPUTS
        # xyz: [x,y,z] locations as a list at which primary field is computed

        # TRANSMITTER NODES
        I = self.Imax
        TxNodes = self.loc
        x1tr = TxNodes[:,0]
        x2tr = TxNodes[:,1]
        x3tr = TxNodes[:,2]

        M = np.size(xyz[0])
        N = np.size(x1tr)-1

        Hx0 = np.zeros(M)
        Hy0 = np.zeros(M)
        Hz0 = np.zeros(M)

        for pp in range(0,N):

            # Wire ends for transmitter wire pp
            x1a = x1tr[pp]
            x2a = x2tr[pp]
            x3a = x3tr[pp]
            x1b = x1tr[pp+1]
            x2b = x2tr[pp+1]
            x3b = x3tr[pp+1]

            # Vector Lengths between points
            vab = np.sqrt((x1b - x1a)**2 + (x2b - x2a)**2 + (x3b - x3a)**2)
            vap = np.sqrt((xyz[0] - x1a)**2 + (xyz[1] - x2a)**2 + (xyz[2] - x3a)**2)
            vbp = np.sqrt((xyz[0] - x1b)**2 + (xyz[1] - x2b)**2 + (xyz[2] - x3b)**2)

            # Cosines from cos()=<v1,v2>/(|v1||v2|)
            CosAlpha = ((xyz[0]-x1a)*(x1b - x1a) + (xyz[1]-x2a)*(x2b - x2a) + (xyz[2]-x3a)*(x3b - x3a))/(vap*vab)
            CosBeta  = ((xyz[0]-x1b)*(x1a - x1b) + (xyz[1]-x2b)*(x2a - x2b) + (xyz[2]-x3b)*(x3a - x3b))/(vbp*vab)

            # Determining Radial Vector From Wire
            DotTemp = (x1a - xyz[0])*(x1b - x1a) + (x2a - xyz[1])*(x2b - x2a) + (x3a - xyz[2])*(x3b - x3a)

            Rx1 = (x1a - xyz[0]) - DotTemp*(x1b - x1a)/vab**2
            Rx2 = (x2a - xyz[1]) - DotTemp*(x2b - x2a)/vab**2
            Rx3 = (x3a - xyz[2]) - DotTemp*(x3b - x3a)/vab**2

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

        Hx0 = sp.diags(Hx0)
        Hy0 = sp.diags(Hy0)
        Hz0 = sp.diags(Hz0)

        H0 = sp.vstack([Hx0,Hy0,Hz0])

        return H0








