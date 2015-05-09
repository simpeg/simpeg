from SimPEG import Survey, Utils, Problem, np, sp, mkvc
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc

class RxMT(Survey.BaseRx):

    knownRxTypes = {
                    'zxxr':[['e', 'Ex'],['b','Fx'], 'real'],
                    'zxyr':[['e', 'Ex'],['b','Fy'], 'real'],
                    'zyxr':[['e', 'Ey'],['b','Fx'], 'real'],
                    'zyyr':[['e', 'Ey'],['b','Fy'], 'real'],
                    'zxxi':[['e', 'Ex'],['b','Fx'], 'imag'],
                    'zxyi':[['e', 'Ex'],['b','Fy'], 'imag'],
                    'zyxi':[['e', 'Ey'],['b','Fx'], 'imag'],
                    'zyyi':[['e', 'Ey'],['b','Fy'], 'imag'],

                    #TODO: Add tipper fractions as well. Bz/B(x|y)
                    # 'exi':['e', 'Ex', 'imag'],
                    # 'eyi':['e', 'Ey', 'imag'],
                    # 'ezi':['e', 'Ez', 'imag'],

                    # 'bxr':['b', 'Fx', 'real'],
                    # 'byr':['b', 'Fy', 'real'],
                    # 'bzr':['b', 'Fz', 'real'],
                    # 'bxi':['b', 'Fx', 'imag'],
                    # 'byi':['b', 'Fy', 'imag'],
                    # 'bzi':['b', 'Fz', 'imag'],
                   }
    # TODO: Have locs as single or double coordinates for both or numerator and denominator separately, respectively.
    def __init__(self, locs, rxType):
        Survey.BaseRx.__init__(self, locs, rxType)

    @property
    def projField(self,fracPos):
        """
        Field Type projection (e.g. e b ...)
        :param str fracPos: Position of the field in the data ratio
        
        """
        if 'numerator' in fracPos:
            return self.knownRxTypes[self.rxType][0][0]
        elif 'denominator' in fracPos:
            return self.knownRxTypes[self.rxType][1][0]
        else:
            raise Exception('{s} is an unknown option. Use numerator or denominator.')

    @property
    def projGLoc(self,fracPos):
        """
        Grid Location projection (e.g. Ex Fy ...)
        :param str fracPos: Position of the field in the data ratio
        
        """
        if 'numerator' in fracPos:
            return self.knownRxTypes[self.rxType][0][1]
        elif 'denominator' in fracPos:
            return self.knownRxTypes[self.rxType][0][1]
        else:
            raise Exception('{s} is an unknown option. Use numerator or denominator.')

    @property
    def projComp(self):
        """Component projection (real/imag)"""
        return self.knownRxTypes[self.rxType][2]

    def projectFields(self, src, mesh, u):
        '''
        Project the fields and return the 
        '''

        # Get the projection
        # Pex = self.getP(mesh,'Ex')
        # Pey = self.getP(mesh,'Ey')
        # Pbx = self.getP(mesh,'Fx')
        # Pby = self.getP(mesh,'Fy')
        Pex = mesh.getInterpolationMat(self.locs,'Ex')
        Pey = mesh.getInterpolationMat(self.locs,'Ey')
        Pbx = mesh.getInterpolationMat(self.locs,'Fx')
        Pby = mesh.getInterpolationMat(self.locs,'Fy')
        # Get the fields at location
        ex_px = Pex*u[src,'e_px']
        ey_px = Pey*u[src,'e_px']
        ex_py = Pex*u[src,'e_py']
        ey_py = Pey*u[src,'e_py']
        hx_px = Pbx*u[src,'b_px']/mu_0
        hy_px = Pby*u[src,'b_px']/mu_0
        hx_py = Pbx*u[src,'b_py']/mu_0
        hy_py = Pby*u[src,'b_py']/mu_0
        if 'zxx' in self.rxType:
            f_part_complex = (ex_px*hy_py - ex_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
        elif 'zxy' in self.rxType:
            f_part_complex  = (-ex_px*hx_py + ex_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)
        elif 'zyx' in self.rxType:
            f_part_complex  = (ey_px*hy_py - ey_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
        elif 'zyy' in self.rxType:
            f_part_complex  = (-ey_px*hx_py + ey_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)

        # P_num = self.getP(mesh,self.projGLoc('numerator'))
        # u_num_complex = u[src, self.projField('numerator')]
        # # Get the denominator information
        # P_den = self.getP(mesh,self.projGLoc('denominator'))
        # u_den_complex = u[src, self.projField('denominator')]
        # # Calculate the fraction
        # f_part_complex = (P_num*u_num_complex)/(P_den*u_den_complex)
        # get the real or imag component
        real_or_imag = self.projComp
        f_part = getattr(f_part_complex, real_or_imag)
        return f_part

    def projectFieldsDeriv(self, src, mesh, u, v, adjoint=False):
        P = self.getP(mesh)

        if not adjoint:
            Pv_complex = P * v
            real_or_imag = self.projComp
            Pv = getattr(Pv_complex, real_or_imag)
        elif adjoint:
            Pv_real = P.T * v

            real_or_imag = self.projComp
            if real_or_imag == 'imag':
                Pv = 1j*Pv_real
            elif real_or_imag == 'real':
                Pv = Pv_real.astype(complex)
            else:
                raise NotImplementedError('must be real or imag')

        return Pv


# Note: Might need to add tests to make sure that both polarization have the same rxList. 
class srcMT(Survey.BaseSrc):
    '''
    Sources for the MT problem. 
    Use the SimPEG BaseSrc, since the source fields share properties with the transmitters.

    :param float freq: The frequency of the source
    :param list rxList: A list of receivers associated with the source
    :param str srcPol: The polarization of the source
    '''

    freq = None #: Frequency (float)

    rxPair = RxMT

    knownSrcTypes = ['pol_xy','pol_x','pol_y'] # ORThogonal POLarization

    def __init__(self, freq, rxList, srcPol = 'pol_xy'): # remove rxType? hardcode to one thing. always polarizations
        self.freq = float(freq)
        Survey.BaseSrc.__init__(self, None, srcPol, rxList)



class FieldsMT(Problem.Fields):
    """Fancy Field Storage for a MT survey."""
    knownFields = {'b_px': 'F','b_py': 'F', 'e_px': 'E','e_py': 'E'}
    dtype = complex


class SurveyMT(Survey.BaseSurvey):
    """
        Survey class for MT. Contains all the sources associated with the survey.

        :param list srcList: List of sources associated with the survey

    """

    srcPair = srcMT

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)

        _freqDict = {}
        for src in srcList:
            if src.freq not in _freqDict:
                _freqDict[src.freq] = []
            _freqDict[src.freq] += [src]

        self._freqDict = _freqDict
        self._freqs = sorted([f for f in self._freqDict])

    @property
    def freqs(self):
        """Frequencies"""
        return self._freqs

    @property
    def nFreq(self):
        """Number of frequencies"""
        return len(self._freqDict)

    # TODO: Rename to getSources
    def getSources(self, freq):
        """Returns the sources associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def projectFields(self, u):
        data = DataMT(self)
        for src in self.srcList:
            print 'Project at freq: {:.3e}'.format(src.freq)
            sys.stdout.flush()
            for rx in src.rxList:
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Transmitters to project fields deriv.')

class DataMT(Survey.Data):
    '''
    Data class for MTdata 

    :param SimPEG survey object survey: 
    :param v vector with data

    '''
    def __init__(self, survey, v=None):
        # Pass the variables to the "parent" method
        Survey.Data.__init__(self, survey, v)

    def toRecArray(self,returnType='RealImag'):
        '''
        Function that returns a numpy.recarray for a SimpegMT impedance data object.

        :param str returnType: Switches between returning a rec array where the impedance is split to real and imaginary ('RealImag') or is a complex ('Complex')

        '''

        def rec2ndarr(x,dt=float):
            return x.view((dt, len(x.dtype.names)))
        # Define the record fields
        dtRI = [('freq',float),('x',float),('y',float),('z',float),('zxxr',float),('zxxi',float),('zxyr',float),('zxyi',float),('zyxr',float),('zyxi',float),('zyyr',float),('zyyi',float)]
        dtCP = [('freq',float),('x',float),('y',float),('z',float),('zxx',complex),('zxy',complex),('zyx',complex),('zyy',complex)]
        impList = ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']
        for src in self.survey.srcList:
            # Temp array for all the receivers of the source.
            # Note: needs to be written more generally, using diffterent rxTypes and not all the data at the locaitons
            # Assume the same locs for all RX
            locs = src.rxList[0].locs
            tArrRec = np.concatenate((src.freq*np.ones((locs.shape[0],1)),locs,np.nan*np.ones((locs.shape[0],8))),axis=1).view(dtRI)
            # np.array([(src.freq,rx.locs[0,0],rx.locs[0,1],rx.locs[0,2],np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ) for rx in src.rxList],dtype=dtRI)
            # Get the type and the value for the DataMT object as a list
            typeList = [[rx.rxType,self[src,rx]] for rx in src.rxList]
            # Insert the values to the temp array
            for nr,(key,val) in enumerate(typeList):
                tArrRec[key] = mkvc(val,2)
            # Masked array 
            mArrRec = np.ma.MaskedArray(rec2ndarr(tArrRec),mask=np.isnan(rec2ndarr(tArrRec))).view(dtype=tArrRec.dtype)
            # Unique freq and loc of the masked array
            uniFLmarr = np.unique(mArrRec[['freq','x','y','z']])

            try:
                outTemp = recFunc.stack_arrays((outTemp,mArrRec))
                #outTemp = np.concatenate((outTemp,dataBlock),axis=0)
            except NameError as e:
                outTemp = mArrRec

            if 'RealImag' in returnType:
                outArr = outTemp
            if 'Complex' in returnType:
                # Add the real and imaginary to a complex number

                outArr = np.empty(outTemp.shape,dtype=dtCP)
                for comp in ['freq','x','y','z']:
                    outArr[comp] = outTemp[comp].copy()
                for comp in ['zxx','zxy','zyx','zyy']:
                    outArr[comp] = outTemp[comp+'r'].copy() + 1j*outTemp[comp+'i'].copy()

        # Return 
        return outArr