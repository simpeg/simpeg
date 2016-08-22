from __future__ import print_function
from SimPEG import Survey as SimPEGsurvey, Utils, Problem, Maps, np, sp, mkvc
from SimPEG.EM.FDEM.SrcFDEM import BaseSrc as FDEMBaseSrc
from SimPEG.EM.Utils import omega
from scipy.constants import mu_0
from numpy.lib import recfunctions as recFunc
from .Utils import rec2ndarr
from . import SrcMT
import sys

#################
### Receivers ###
#################
class Rx(SimPEGsurvey.BaseRx):
    """
        Class that defines natural source receivers.

        See knownRxTypes for types of allowed receivers.

        :param ndArray locs: Locations of the receivers
        :param str rxType: The type of receiver

    """

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
        SimPEGsurvey.BaseRx.__init__(self, locs, rxType)

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

    def eval(self, src, mesh, f):
        '''
        Project the fields to natural source data.

            :param SrcMT src: The source of the fields to project
            :param SimPEG.Mesh mesh:
            :param FieldsMT f: Natural source fields object to project
        '''

        ## NOTE: Assumes that e is on t
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
            ## NOTE: Assumes that e is on edges and b on the faces. Need to generalize that or use a prop of fields to determine that.
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
        # print(f_part)
        return f_part

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
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
            elif self.projType is 'T3D':
                if self.locs.ndim == 3:
                    eFLocs = self.locs[:,:,0]
                    bFLocs = self.locs[:,:,1]
                else:
                    eFLocs = self.locs
                    bFLocs = self.locs
                # Get the projection
                Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
                Pby = mesh.getInterpolationMat(bFLocs,'Fy')
                Pbz = mesh.getInterpolationMat(bFLocs,'Fz')

                # Get the fields at location
                # px: x-polaration and py: y-polaration.
                bx_px = Pbx*f[src,'b_px']
                by_px = Pby*f[src,'b_px']
                bz_px = Pbz*f[src,'b_px']
                bx_py = Pbx*f[src,'b_py']
                by_py = Pby*f[src,'b_py']
                bz_py = Pbz*f[src,'b_py']
                # Derivatives as lambda functions
                # NOTE: Think b_p?Deriv_u should return a 2*nF size matrix
                bx_px_u = lambda vec: Pbx*f._b_pxDeriv_u(src,vec)
                by_px_u = lambda vec: Pby*f._b_pxDeriv_u(src,vec)
                bz_px_u = lambda vec: Pbz*f._b_pxDeriv_u(src,vec)
                bx_py_u = lambda vec: Pbx*f._b_pyDeriv_u(src,vec)
                by_py_u = lambda vec: Pby*f._b_pyDeriv_u(src,vec)
                bz_py_u = lambda vec: Pbz*f._b_pyDeriv_u(src,vec)
                # Update the input vector
                sDiag = lambda t: Utils.sdiag(mkvc(t,2))
                # Define the components of the derivative
                Hd = sDiag(1./(sDiag(bx_px)*by_py - sDiag(bx_py)*by_px))
                Hd_uV = sDiag(by_py)*bx_px_u(v) + sDiag(bx_px)*by_py_u(v) - sDiag(bx_py)*by_px_u(v) - sDiag(by_px)*bx_py_u(v)
                if 'tzx' in self.rxType:
                    Tij = sDiag(Hd*( - sDiag(by_px)*bz_py + sDiag(by_py)*bz_px ))
                    TijN_uV = -sDiag(by_px)*bz_py_u(v) - sDiag(bz_py)*by_px_u(v) + sDiag(by_py)*bz_px_u(v) + sDiag(bz_px)*by_py_u(v)
                elif 'tzy' in self.rxType:
                    Tij = sDiag(Hd*( sDiag(bx_px)*bz_py - sDiag(bx_py)*bz_px ))
                    TijN_uV =  sDiag(bz_py)*bx_px_u(v) + sDiag(bx_px)*bz_py_u(v) - sDiag(bx_py)*bz_px_u(v) - sDiag(bz_px)*bx_py_u(v)
                # Calculate the complex derivative
                PDeriv_complex = Hd * (TijN_uV - Tij * Hd_uV )

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
                # PDeriv_real = np.hstack((mkvc(PDeriv_real[:len(PDeriv_real)/2],2),mkvc(PDeriv_real[len(PDeriv_real)/2::],2)))
                PDeriv_real = PDeriv_real.reshape((2,mesh.nE)).T

            elif self.projType is 'T3D':
                if self.locs.ndim == 3:
                    bFLocs = self.locs[:,:,1]
                else:
                    bFLocs = self.locs
                # Get the projection
                Pbx = mesh.getInterpolationMat(bFLocs,'Fx')
                Pby = mesh.getInterpolationMat(bFLocs,'Fy')
                Pbz = mesh.getInterpolationMat(bFLocs,'Fz')
                # Get the fields at location
                # px: x-polaration and py: y-polaration.
                abx_px = mkvc(mkvc(f[src,'b_px'],2).T*Pbx.T)
                aby_px = mkvc(mkvc(f[src,'b_px'],2).T*Pby.T)
                abz_px = mkvc(mkvc(f[src,'b_px'],2).T*Pbz.T)
                abx_py = mkvc(mkvc(f[src,'b_py'],2).T*Pbx.T)
                aby_py = mkvc(mkvc(f[src,'b_py'],2).T*Pby.T)
                abz_py = mkvc(mkvc(f[src,'b_py'],2).T*Pbz.T)
                # Derivatives as lambda functions
                abx_px_u = lambda vec: f._b_pxDeriv_u(src,Pbx.T*vec,adjoint=True)
                aby_px_u = lambda vec: f._b_pxDeriv_u(src,Pby.T*vec,adjoint=True)
                abz_px_u = lambda vec: f._b_pxDeriv_u(src,Pbz.T*vec,adjoint=True)
                abx_py_u = lambda vec: f._b_pyDeriv_u(src,Pbx.T*vec,adjoint=True)
                aby_py_u = lambda vec: f._b_pyDeriv_u(src,Pby.T*vec,adjoint=True)
                abz_py_u = lambda vec: f._b_pyDeriv_u(src,Pbz.T*vec,adjoint=True)

                # Update the input vector
                # Define shortcuts
                sDiag = lambda t: Utils.sdiag(mkvc(t,2))
                sVec = lambda t: Utils.sp.csr_matrix(mkvc(t,2))
                # Define the components of the derivative
                aHd = sDiag(1./(sDiag(abx_px)*aby_py - sDiag(abx_py)*aby_px))
                aHd_uV = lambda x: abx_px_u(sDiag(aby_py)*x) + abx_px_u(sDiag(aby_py)*x) - aby_px_u(sDiag(abx_py)*x) - abx_py_u(sDiag(aby_px)*x)
                # Need to fix this to reflect the adjoint
                if 'tzx' in self.rxType:
                    Tij = sDiag(aHd*( -sDiag(abz_py)*aby_px + sDiag(abz_px)*aby_py))
                    TijN_uV = lambda x: -abz_py_u(sDiag(aby_px)*x) - aby_px_u(sDiag(abz_py)*x) + aby_py_u(sDiag(abz_px)*x) + abz_px_u(sDiag(aby_py)*x)
                elif 'tzy' in self.rxType:
                    Tij = sDiag(aHd*( sDiag(abz_py)*abx_px - sDiag(abz_px)*abx_py))
                    TijN_uV = lambda x: abx_px_u(sDiag(abz_py)*x) + abz_py_u(sDiag(abx_px)*x) - abx_py_u(sDiag(abz_px)*x) - abz_px_u(sDiag(abx_py)*x)
                # Calculate the complex derivative
                PDeriv_real = TijN_uV(aHd*v) - aHd_uV(Tij.T*aHd*v)#
                # NOTE: Need to reshape the output to go from 2*nU array to a (nU,2) matrix for each polarization
                # PDeriv_real = np.hstack((mkvc(PDeriv_real[:len(PDeriv_real)/2],2),mkvc(PDeriv_real[len(PDeriv_real)/2::],2)))
                PDeriv_real = PDeriv_real.reshape((2,mesh.nE)).T
            # Extract the data
            if real_or_imag == 'imag':
                Pv = 1j*PDeriv_real
            elif real_or_imag == 'real':
                Pv = PDeriv_real.astype(complex)


        return Pv

#################
###  Survey   ###
#################
class Survey(SimPEGsurvey.BaseSurvey):
    """
        Survey class for MT. Contains all the sources associated with the survey.

        :param list srcList: List of sources associated with the survey

    """
    srcPair = SrcMT.BaseMTSrc

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        SimPEGsurvey.BaseSurvey.__init__(self, **kwargs)

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

    def eval(self, f):
        data = Data(self)
        for src in self.srcList:
            sys.stdout.flush()
            for rx in src.rxList:
                data[src, rx] = rx.eval(src, self.mesh, f)
        return data

    def evalDeriv(self, f):
        raise Exception('Use Transmitters to project fields deriv.')

#################
###   Data    ###
#################
class Data(SimPEGsurvey.Data):
    '''
    Data class for MTdata. Stores the data vector indexed by the survey.

    :param SimPEG survey object survey:
    :param v vector of the data in order matching of the survey


    '''
    def __init__(self, survey, v=None):
        # Pass the variables to the "parent" method
        SimPEGsurvey.Data.__init__(self, survey, v)

    # # Import data
    # @classmethod
    # def fromEDIFiles():
    #     pass

    def toRecArray(self,returnType='RealImag'):
        '''
        Function that returns a numpy.recarray for a SimpegMT impedance data object.

        :param str returnType: Switches between returning a rec array where the impedance is split to real and imaginary ('RealImag') or is a complex ('Complex')

        '''

        # Define the record fields
        dtRI = [('freq',float),('x',float),('y',float),('z',float),('zxxr',float),('zxxi',float),('zxyr',float),('zxyi',float),
        ('zyxr',float),('zyxi',float),('zyyr',float),('zyyi',float),('tzxr',float),('tzxi',float),('tzyr',float),('tzyi',float)]
        dtCP = [('freq',float),('x',float),('y',float),('z',float),('zxx',complex),('zxy',complex),('zyx',complex),('zyy',complex),('tzx',complex),('tzy',complex)]
        impList = ['zxxr','zxxi','zxyr','zxyi','zyxr','zyxi','zyyr','zyyi']
        for src in self.survey.srcList:
            # Temp array for all the receivers of the source.
            # Note: needs to be written more generally, using diffterent rxTypes and not all the data at the locaitons
            # Assume the same locs for all RX
            locs = src.rxList[0].locs
            if locs.shape[1] == 1:
                locs = np.hstack((np.array([[0.0,0.0]]),locs))
            elif locs.shape[1] == 2:
                locs = np.hstack((np.array([[0.0]]),locs))
            tArrRec = np.concatenate((src.freq*np.ones((locs.shape[0],1)),locs,np.nan*np.ones((locs.shape[0],12))),axis=1).view(dtRI)
            # np.array([(src.freq,rx.locs[0,0],rx.locs[0,1],rx.locs[0,2],np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ) for rx in src.rxList],dtype=dtRI)
            # Get the type and the value for the DataMT object as a list
            typeList = [[rx.rxType.replace('z1d','zyx'),self[src,rx]] for rx in src.rxList]
            # Insert the values to the temp array
            for nr,(key,val) in enumerate(typeList):
                tArrRec[key] = mkvc(val,2)
            # Masked array
            mArrRec = np.ma.MaskedArray(rec2ndarr(tArrRec),mask=np.isnan(rec2ndarr(tArrRec))).view(dtype=tArrRec.dtype)
            # Unique freq and loc of the masked array
            uniFLmarr = np.unique(mArrRec[['freq','x','y','z']]).copy()

            try:
                outTemp = recFunc.stack_arrays((outTemp,mArrRec))
                #outTemp = np.concatenate((outTemp,dataBlock),axis=0)
            except NameError:
                outTemp = mArrRec

            if 'RealImag' in returnType:
                outArr = outTemp
            elif 'Complex' in returnType:
                # Add the real and imaginary to a complex number
                outArr = np.empty(outTemp.shape,dtype=dtCP)
                for comp in ['freq','x','y','z']:
                    outArr[comp] = outTemp[comp].copy()
                for comp in ['zxx','zxy','zyx','zyy','tzx','tzy']:
                    outArr[comp] = outTemp[comp+'r'].copy() + 1j*outTemp[comp+'i'].copy()
            else:
                raise NotImplementedError('{:s} is not implemented, as to be RealImag or Complex.')

        # Return
        return outArr

    @classmethod
    def fromRecArray(cls, recArray, srcType='primary'):
        """
        Class method that reads in a numpy record array to MTdata object.

        Only imports the impedance data.

        """
        if srcType=='primary':
            src = SrcMT.polxy_1Dprimary
        elif srcType=='total':
            src = SrcMT.polxy_1DhomotD
        else:
            raise NotImplementedError('{:s} is not a valid source type for MTdata')

        # Find all the frequencies in recArray
        uniFreq = np.unique(recArray['freq'])
        srcList = []
        dataList = []
        for freq in uniFreq:
            # Initiate rxList
            rxList = []
            # Find that data for freq
            dFreq = recArray[recArray['freq'] == freq].copy()
            # Find the impedance rxTypes in the recArray.
            rxTypes = [ comp for comp in recArray.dtype.names if (len(comp)==4 or len(comp)==3) and 'z' in comp]
            for rxType in rxTypes:
                # Find index of not nan values in rxType
                notNaNind = ~np.isnan(dFreq[rxType])
                if np.any(notNaNind): # Make sure that there is any data to add.
                    locs = rec2ndarr(dFreq[['x','y','z']][notNaNind].copy())
                    if dFreq[rxType].dtype.name in 'complex128':
                        rxList.append(Rx(locs,rxType+'r'))
                        dataList.append(dFreq[rxType][notNaNind].real.copy())
                        rxList.append(Rx(locs,rxType+'i'))
                        dataList.append(dFreq[rxType][notNaNind].imag.copy())
                    else:
                        rxList.append(Rx(locs,rxType))
                        dataList.append(dFreq[rxType][notNaNind].copy())
            srcList.append(src(rxList,freq))

        # Make a survey
        survey = Survey(srcList)
        dataVec = np.hstack(dataList)
        return cls(survey,dataVec)

