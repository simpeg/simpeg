from SimPEG import Survey, Utils, Problem, Maps, np, sp, mkvc
from simpegEM.FDEM.SurveyFDEM import SrcFDEM
from simpegEM.Utils.EMUtils import omega
from scipy.constants import mu_0
import sys
from numpy.lib import recfunctions as recFunc
from DataMT import DataMT
from simpegMT.Sources import homo1DModelSource
#################
### Receivers ###
#################

class RxMT(Survey.BaseRx):

    knownRxTypes = {
                    # 3D impedance
                    'zxxr':['Z3D', 'real'],
                    'zxyr':['Z3D', 'real'],
                    'zyxr':['Z3D', 'real'],
                    'zyyr':['Z3D', 'real'],
                    'zxxi':['Z3D', 'imag'],
                    'zxyi':['Z3D', 'imag'],
                    'zyxi':['Z3D', 'imag'],
                    'zyyi':['Z3D', 'imag'],
                    # 2D impedance
                    # TODO:
                    # 1D impedance
                    'z1dr':['Z1D', 'real'],
                    'z1di':['Z1D', 'imag'],
                    # Tipper
                    'tzxr':['T3D','real'],
                    'tzxi':['T3D','imag'],
                    'tzyr':['T3D','real'],
                    'tzyi':['T3D','imag']
                   }
    # TODO: Have locs as single or double coordinates for both or numerator and denominator separately, respectively.
    def __init__(self, locs, rxType):
        Survey.BaseRx.__init__(self, locs, rxType)

    @property
    def projField(self):
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
    def projGLoc(self):
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
    def projType(self):
        """
        Receiver type for projection.

        """
        return self.knownRxTypes[self.rxType][0]

    @property
    def projComp(self):
        """Component projection (real/imag)"""
        return self.knownRxTypes[self.rxType][1]

    def projectFields(self, src, mesh, f):
        '''
        Project the fields and return the correct data.
        '''

        if self.projType is 'Z1D':
            Pex = mesh.getInterpolationMat(self.locs[:,-1],'Fx')
            Pbx = mesh.getInterpolationMat(self.locs[:,-1],'Ex')
            ex = Pex*mkvc(f[src,'e_1d'],2)
            bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
            # Note: Has a minus sign in front, to comply with quadrant calculations.
            # Can be derived from zyx case for the 3D case.
            f_part_complex = -ex/bx
        # elif self.projType is 'Z2D':
        elif self.projType is 'Z3D':
            if self.locs.ndim == 3:
                eFLocs = self.locs[:,:,0]
                bFLocs = self.locs[:,:,1]
            else:
                eFLocs = self.locs
                bFLocs = self.locs
            # Get the projection
            Pex = mesh.getInterpolationMat(eFLocs,'Ex')
            Pey = mesh.getInterpolationMat(eFLocs,'Ey')
            Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
            Pby = mesh.getInterpolationMat(bFLocs,'Fy')
            # Get the fields at location
            # px: x-polaration and py: y-polaration.
            ex_px = Pex*f[src,'e_px']
            ey_px = Pey*f[src,'e_px']
            ex_py = Pex*f[src,'e_py']
            ey_py = Pey*f[src,'e_py']
            hx_px = Pbx*f[src,'b_px']/mu_0
            hy_px = Pby*f[src,'b_px']/mu_0
            hx_py = Pbx*f[src,'b_py']/mu_0
            hy_py = Pby*f[src,'b_py']/mu_0
            # Make the complex data
            if 'zxx' in self.rxType:
                f_part_complex = ( ex_px*hy_py - ex_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zxy' in self.rxType:
                f_part_complex = (-ex_px*hx_py + ex_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zyx' in self.rxType:
                f_part_complex = ( ey_px*hy_py - ey_py*hy_px)/(hx_px*hy_py - hx_py*hy_px)
            elif 'zyy' in self.rxType:
                f_part_complex = (-ey_px*hx_py + ey_py*hx_px)/(hx_px*hy_py - hx_py*hy_px)
        elif self.projType is 'T3D':
            if self.locs.ndim == 3:
                horLoc = self.locs[:,:,0]
                vertLoc = self.locs[:,:,1]
            else:
                horLoc = self.locs
                vertLoc = self.locs
            Pbx = mesh.getInterpolationMat(horLoc,'Fx')
            Pby = mesh.getInterpolationMat(horLoc,'Fy')
            Pbz = mesh.getInterpolationMat(vertLoc,'Fz')
            bx_px = Pbx*f[src,'b_px']
            by_px = Pby*f[src,'b_px']
            bz_px = Pbz*f[src,'b_px']
            bx_py = Pbx*f[src,'b_py']
            by_py = Pby*f[src,'b_py']
            bz_py = Pbz*f[src,'b_py']
            if 'tzx' in self.rxType:
                f_part_complex = (- by_px*bz_py + by_py*bz_px)/(bx_px*by_py - bx_py*by_px)
            if 'tzy' in self.rxType:
                f_part_complex = (  bx_px*bz_py - bx_py*bz_px)/(bx_px*by_py - bx_py*by_px)

        else:
            NotImplementedError('Projection of {:s} receiver type is not implemented.'.format(self.rxType))
        # Get the real or imag component
        real_or_imag = self.projComp
        f_part = getattr(f_part_complex, real_or_imag)
        # print f_part
        return f_part

    def projectFieldsDeriv(self, src, mesh, f, v, adjoint=False):
        """
        The derivative of the projection wrt u

        :param MTsrc src: MT source
        :param TensorMesh mesh: Mesh defining the topology of the problem
        :param MTfields f: MT fields object of the source
        :param numpy.ndarray v: Random vector of size
        """

        real_or_imag = self.projComp

        if not adjoint:
            if self.projType is 'Z1D':
                Pex = mesh.getInterpolationMat(self.locs[:,-1],'Fx')
                Pbx = mesh.getInterpolationMat(self.locs[:,-1],'Ex')
                # ex = Pex*mkvc(f[src,'e_1d'],2)
                # bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
                dP_de = -mkvc(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))*(Pex*v),2)
                dP_db = mkvc( Utils.sdiag(Pex*mkvc(f[src,'e_1d'],2))*(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)).T*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)))*(Pbx*f._bDeriv_u(src,v)/mu_0),2)
                PDeriv_complex = np.sum(np.hstack((dP_de,dP_db)),1)
            elif self.projType is 'Z2D':
                raise NotImplementedError('Has not been implement for 2D impedance tensor')
            elif self.projType is 'Z3D':
                if self.locs.ndim == 3:
                    eFLocs = self.locs[:,:,0]
                    bFLocs = self.locs[:,:,1]
                else:
                    eFLocs = self.locs
                    bFLocs = self.locs
                # Get the projection
                Pex = mesh.getInterpolationMat(eFLocs,'Ex')
                Pey = mesh.getInterpolationMat(eFLocs,'Ey')
                Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
                Pby = mesh.getInterpolationMat(bFLocs,'Fy')
                # Get the fields at location
                # px: x-polaration and py: y-polaration.
                ex_px = Pex*f[src,'e_px']
                ey_px = Pey*f[src,'e_px']
                ex_py = Pex*f[src,'e_py']
                ey_py = Pey*f[src,'e_py']
                hx_px = Pbx*f[src,'b_px']/mu_0
                hy_px = Pby*f[src,'b_px']/mu_0
                hx_py = Pbx*f[src,'b_py']/mu_0
                hy_py = Pby*f[src,'b_py']/mu_0
                # Derivatives as lambda functions
                # The size of the diratives should be nD,nU
                ex_px_u = lambda vec: Pex*f._e_pxDeriv_u(src,vec)
                ey_px_u = lambda vec: Pey*f._e_pxDeriv_u(src,vec)
                ex_py_u = lambda vec: Pex*f._e_pyDeriv_u(src,vec)
                ey_py_u = lambda vec: Pey*f._e_pyDeriv_u(src,vec)
                # NOTE: Think b_p?Deriv_u should return a 2*nF size matrix
                hx_px_u = lambda vec: Pbx*f._b_pxDeriv_u(src,vec)/mu_0
                hy_px_u = lambda vec: Pby*f._b_pxDeriv_u(src,vec)/mu_0
                hx_py_u = lambda vec: Pbx*f._b_pyDeriv_u(src,vec)/mu_0
                hy_py_u = lambda vec: Pby*f._b_pyDeriv_u(src,vec)/mu_0
                # Update the input vector
                sDiag = lambda t: Utils.sdiag(mkvc(t,2))
                # Define the components of the derivative
                Hd = sDiag(1./(sDiag(hx_px)*hy_py - sDiag(hx_py)*hy_px))
                Hd_uV = sDiag(hy_py)*hx_px_u(v) + sDiag(hx_px)*hy_py_u(v) - sDiag(hx_py)*hy_px_u(v) - sDiag(hy_px)*hx_py_u(v)
                # Calculate components
                if 'zxx' in self.rxType:
                    Zij = sDiag(Hd*( sDiag(ex_px)*hy_py - sDiag(ex_py)*hy_px ))
                    ZijN_uV =  sDiag(hy_py)*ex_px_u(v) + sDiag(ex_px)*hy_py_u(v) - sDiag(ex_py)*hy_px_u(v) - sDiag(hy_px)*ex_py_u(v)
                elif 'zxy' in self.rxType:
                    Zij = sDiag(Hd*(-sDiag(ex_px)*hx_py + sDiag(ex_py)*hx_px ))
                    ZijN_uV = -sDiag(hx_py)*ex_px_u(v) - sDiag(ex_px)*hx_py_u(v) + sDiag(ex_py)*hx_px_u(v) + sDiag(hx_px)*ex_py_u(v)
                elif 'zyx' in self.rxType:
                    Zij = sDiag(Hd*( sDiag(ey_px)*hy_py - sDiag(ey_py)*hy_px ))
                    ZijN_uV =  sDiag(hy_py)*ey_px_u(v) + sDiag(ey_px)*hy_py_u(v) - sDiag(ey_py)*hy_px_u(v) - sDiag(hy_px)*ey_py_u(v)
                elif 'zyy' in self.rxType:
                    Zij = sDiag(Hd*(-sDiag(ey_px)*hx_py + sDiag(ey_py)*hx_px ))
                    ZijN_uV = -sDiag(hx_py)*ey_px_u(v) - sDiag(ey_px)*hx_py_u(v) + sDiag(ey_py)*hx_px_u(v) + sDiag(hx_px)*ey_py_u(v)

                # Calculate the complex derivative
                PDeriv_complex = Hd * (ZijN_uV - Zij * Hd_uV )
            # Extract the real number for the real/imag components.
            Pv = np.array(getattr(PDeriv_complex, real_or_imag))
        elif adjoint:
            # Note: The v vector is real and the return should be complex
            if self.projType is 'Z1D':
                Pex = mesh.getInterpolationMat(self.locs[:,-1],'Fx')
                Pbx = mesh.getInterpolationMat(self.locs[:,-1],'Ex')
                # ex = Pex*mkvc(f[src,'e_1d'],2)
                # bx = Pbx*mkvc(f[src,'b_1d'],2)/mu_0
                dP_deTv = -mkvc(Pex.T*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0)).T*v,2)
                db_duv = Pbx.T/mu_0*Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))*(Utils.sdiag(1./(Pbx*mkvc(f[src,'b_1d'],2)/mu_0))).T*Utils.sdiag(Pex*mkvc(f[src,'e_1d'],2)).T*v
                dP_dbTv = mkvc(f._bDeriv_u(src,db_duv,adjoint=True),2)
                PDeriv_real = np.sum(np.hstack((dP_deTv,dP_dbTv)),1)
            elif self.projType is 'Z2D':
                raise NotImplementedError('Has not be implement for 2D impedance tensor')
            elif self.projType is 'Z3D':
                if self.locs.ndim == 3:
                    eFLocs = self.locs[:,:,0]
                    bFLocs = self.locs[:,:,1]
                else:
                    eFLocs = self.locs
                    bFLocs = self.locs
                # Get the projection
                Pex = mesh.getInterpolationMat(eFLocs,'Ex')
                Pey = mesh.getInterpolationMat(eFLocs,'Ey')
                Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
                Pby = mesh.getInterpolationMat(bFLocs,'Fy')
                # Get the fields at location
                # px: x-polaration and py: y-polaration.
                aex_px = mkvc(mkvc(f[src,'e_px'],2).T*Pex.T)
                aey_px = mkvc(mkvc(f[src,'e_px'],2).T*Pey.T)
                aex_py = mkvc(mkvc(f[src,'e_py'],2).T*Pex.T)
                aey_py = mkvc(mkvc(f[src,'e_py'],2).T*Pey.T)
                ahx_px = mkvc(mkvc(f[src,'b_px'],2).T/mu_0*Pbx.T)
                ahy_px = mkvc(mkvc(f[src,'b_px'],2).T/mu_0*Pby.T)
                ahx_py = mkvc(mkvc(f[src,'b_py'],2).T/mu_0*Pbx.T)
                ahy_py = mkvc(mkvc(f[src,'b_py'],2).T/mu_0*Pby.T)
                # Derivatives as lambda functions
                aex_px_u = lambda vec: f._e_pxDeriv_u(src,Pex.T*vec,adjoint=True)
                aey_px_u = lambda vec: f._e_pxDeriv_u(src,Pey.T*vec,adjoint=True)
                aex_py_u = lambda vec: f._e_pyDeriv_u(src,Pex.T*vec,adjoint=True)
                aey_py_u = lambda vec: f._e_pyDeriv_u(src,Pey.T*vec,adjoint=True)
                ahx_px_u = lambda vec: f._b_pxDeriv_u(src,Pbx.T*vec,adjoint=True)/mu_0
                ahy_px_u = lambda vec: f._b_pxDeriv_u(src,Pby.T*vec,adjoint=True)/mu_0
                ahx_py_u = lambda vec: f._b_pyDeriv_u(src,Pbx.T*vec,adjoint=True)/mu_0
                ahy_py_u = lambda vec: f._b_pyDeriv_u(src,Pby.T*vec,adjoint=True)/mu_0

                # Update the input vector
                # Define shortcuts
                sDiag = lambda t: Utils.sdiag(mkvc(t,2))
                sVec = lambda t: Utils.sp.csr_matrix(mkvc(t,2))
                # Define the components of the derivative
                aHd = sDiag(1./(sDiag(ahx_px)*ahy_py - sDiag(ahx_py)*ahy_px))
                aHd_uV = lambda x: ahx_px_u(sDiag(ahy_py)*x) + ahx_px_u(sDiag(ahy_py)*x) - ahy_px_u(sDiag(ahx_py)*x) - ahx_py_u(sDiag(ahy_px)*x)
                # Need to fix this to reflect the adjoint
                if 'zxx' in self.rxType:
                    Zij = sDiag(aHd*( sDiag(ahy_py)*aex_px - sDiag(ahy_px)*aex_py))
                    ZijN_uV = lambda x: aex_px_u(sDiag(ahy_py)*x) + ahy_py_u(sDiag(aex_px)*x) - ahy_px_u(sDiag(aex_py)*x) - aex_py_u(sDiag(ahy_px)*x)
                elif 'zxy' in self.rxType:
                    Zij = sDiag(aHd*(-sDiag(ahx_py)*aex_px + sDiag(ahx_px)*aex_py))
                    ZijN_uV = lambda x:-aex_px_u(sDiag(ahx_py)*x) - ahx_py_u(sDiag(aex_px)*x) + ahx_px_u(sDiag(aex_py)*x) + aex_py_u(sDiag(ahx_px)*x)
                elif 'zyx' in self.rxType:
                    Zij = sDiag(aHd*( sDiag(ahy_py)*aey_px - sDiag(ahy_px)*aey_py))
                    ZijN_uV = lambda x: aey_px_u(sDiag(ahy_py)*x) + ahy_py_u(sDiag(aey_px)*x) - ahy_px_u(sDiag(aey_py)*x) - aey_py_u(sDiag(ahy_px)*x)
                elif 'zyy' in self.rxType:
                    Zij = sDiag(aHd*(-sDiag(ahx_py)*aey_px + sDiag(ahx_px)*aey_py))
                    ZijN_uV = lambda x:-aey_px_u(sDiag(ahx_py)*x) - ahx_py_u(sDiag(aey_px)*x) + ahx_px_u(sDiag(aey_py)*x) + aey_py_u(sDiag(ahx_px)*x)

                # Calculate the complex derivative
                PDeriv_real = ZijN_uV(aHd*v) - aHd_uV(Zij.T*aHd*v)#
                # NOTE: Need to reshape the output to go from 2*nU array to a (nU,2) matrix for each polarization
                PDeriv_real = PDeriv_real.reshape((mesh.nE,2))
            # Extract the data
            if real_or_imag == 'imag':
                Pv = 1j*PDeriv_real
            elif real_or_imag == 'real':
                Pv = PDeriv_real.astype(complex)


        return Pv


###############
### Sources ###
###############

class srcMT(SrcFDEM): # Survey.BaseSrc):
    '''
    Sources for the MT problem.
    Use the SimPEG BaseSrc, since the source fields share properties with the transmitters.

    :param float freq: The frequency of the source
    :param list rxList: A list of receivers associated with the source
    '''

    freq = None #: Frequency (float)
    rxPair = RxMT

    def __init__(self, rxList, freq):
        self.freq = float(freq)
        Survey.BaseSrc.__init__(self, rxList)

# 1D sources
class srcMT_polxy_1DhomotD(srcMT):
    """
    MT source for both polarizations (x and y) for the total Domain. It calculates fields calculated based on conditions on the boundary of the domain.
    """
    def __init__(self, rxList, freq):
        srcMT.__init__(self, rxList, freq)


    # TODO: need to add the  primary fields calc and source terms into the problem.


# Need to implement such that it works for all dims.
class srcMT_polxy_1Dprimary(srcMT):
    """
    MT source for both polarizations (x and y) given a 1D primary models. It assigns fields calculated from the 1D model
    as fields in the full space of the problem.
    """
    def __init__(self, rxList, freq):
        # assert mkvc(self.mesh.hz.shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigma1d = None
        srcMT.__init__(self, rxList, freq)
        # Hidden property of the ePrimary
        self._ePrimary = None

    def ePrimary(self,problem):
        # Get primary fields for both polarizations
        if self.sigma1d is None:
            # Set the sigma1d as the 1st column in the background model
            if len(problem._sigmaPrimary) == problem.mesh.nC:
                if problem.mesh.dim == 1:
                    self.sigma1d = problem.mesh.r(problem._sigmaPrimary,'CC','CC','M')[:]
                elif problem.mesh.dim == 3:
                    self.sigma1d = problem.mesh.r(problem._sigmaPrimary,'CC','CC','M')[0,0,:]
            # Or as the 1D model that matches the vertical cell number
            elif len(problem._sigmaPrimary) == problem.mesh.nCz:
                self.sigma1d = problem._sigmaPrimary

        if self._ePrimary is None:
            self._ePrimary = homo1DModelSource(problem.mesh,self.freq,self.sigma1d)
        return self._ePrimary

    def bPrimary(self,problem):
        # Project ePrimary to bPrimary
        # Satisfies the primary(background) field conditions
        if problem.mesh.dim == 1:
            C = problem.mesh.nodalGrad
        elif problem.mesh.dim == 3:
            C = problem.mesh.edgeCurl
        bBG_bp = (- C * self.ePrimary(problem) )*(1/( 1j*omega(self.freq) ))
        return bBG_bp

    def S_e(self,problem):
        """
        Get the electrical field source
        """
        e_p = self.ePrimary(problem)
        Map_sigma_p = Maps.Vertical1DMap(problem.mesh)
        sigma_p = Map_sigma_p._transform(self.sigma1d)
        # Make mass matrix
        # Note: M(sig) - M(sig_p) = M(sig - sig_p)
        # Need to deal with the edge/face discrepencies between 1d/2d/3d
        if problem.mesh.dim == 1:
            Mesigma = problem.mesh.getFaceInnerProduct(problem.curModel.sigma)
            Mesigma_p = problem.mesh.getFaceInnerProduct(sigma_p)
        if problem.mesh.dim == 2:
            pass
        if problem.mesh.dim == 3:
            Mesigma = problem.MeSigma
            Mesigma_p = problem.mesh.getEdgeInnerProduct(sigma_p)
        return (Mesigma - Mesigma_p) * e_p

    def S_eDeriv_m(self, problem, v, adjoint = False):
        '''
        Get the derivative of S_e wrt to sigma (m)
        '''
        # Need to deal with
        if problem.mesh.dim == 1:
            # Need to use the faceInnerProduct
            MsigmaDeriv = problem.mesh.getFaceInnerProductDeriv(problem.curModel.sigma)(self.ePrimary(problem)[:,1]) * problem.curModel.sigmaDeriv
            # MsigmaDeriv = ( MsigmaDeriv * MsigmaDeriv.T)**2
        if problem.mesh.dim == 2:
            pass
        if problem.mesh.dim == 3:
            # Need to take the derivative of both u_px and u_py
            ePri = self.ePrimary(problem)
            # MsigmaDeriv = problem.MeSigmaDeriv(ePri[:,0]) + problem.MeSigmaDeriv(ePri[:,1])
            # MsigmaDeriv = problem.MeSigmaDeriv(np.sum(ePri,axis=1))
            if adjoint:
                return sp.hstack(( problem.MeSigmaDeriv(ePri[:,0]).T, problem.MeSigmaDeriv(ePri[:,1]).T ))*v
            else:
                return np.hstack(( mkvc(problem.MeSigmaDeriv(ePri[:,0]) * v,2), mkvc(problem.MeSigmaDeriv(ePri[:,1])*v,2) ))
        if adjoint:
            #
            return MsigmaDeriv.T * v
        else:
            # v should be nC size
            return MsigmaDeriv * v


##############
### Survey ###
##############
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
    def getSrcByFreq(self, freq):
        """Returns the sources associated with a specific frequency."""
        assert freq in self._freqDict, "The requested frequency is not in this survey."
        return self._freqDict[freq]

    def projectFields(self, u):
        data = DataMT(self)
        for src in self.srcList:
            sys.stdout.flush()
            for rx in src.rxList:
                data[src, rx] = rx.projectFields(src, self.mesh, u)
        return data

    def projectFieldsDeriv(self, u):
        raise Exception('Use Transmitters to project fields deriv.')

