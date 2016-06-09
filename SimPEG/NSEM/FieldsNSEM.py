from SimPEG import Survey, Utils, Problem, np, sp, mkvc
from scipy.constants import mu_0
import numpy as np
import scipy.sparse as sp
from SimPEG.Utils import Zero, Identity
from SimPEG.EM.Utils import omega


##############
### Fields ###
##############
class BaseNSEMFields(Problem.Fields):
    """Field Storage for a NSEM method."""
    knownFields = {}
    dtype = complex

###########
# 1D Fields
###########
class Fields1D_ePrimSec(BaseNSEMFields):
    """
    Fields storage for the 1D NSEM solution.

    Solving for e fields, using primary/secondary formulation
    """
    knownFields = {'e_1dSolution':'F'}
    aliasFields = {
                    'e_1d' : ['e_1dSolution','F','_e'],
                    'e_1dPrimary' : ['e_1dSolution','F','_ePrimary'],
                    'e_1dSecondary' : ['e_1dSolution','F','_eSecondary'],
                    'b_1d' : ['e_1dSolution','E','_b'],
                    'b_1dPrimary' : ['e_1dSolution','E','_bPrimary'],
                    'b_1dSecondary' : ['e_1dSolution','E','_bSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        BaseNSEMFields.__init__(self,mesh,survey,**kwargs)

    def _ePrimary(self, eSolution, srcList):
        """
        Primary electric field from source

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """
        ePrimary = np.zeros_like(eSolution)
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                ePrimary[:,i] = ep[:,-1]
        return ePrimary

    def _eSecondary(self, eSolution, srcList):
        """
        Secondary electric field is the thing we solved for

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field
        """
        return eSolution

    # Overwriting a base FDEM method, could use it.
    def _e(self, eSolution, srcList):
        """
        Total electric field is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total electric field
        """
        return self._ePrimary(eSolution,srcList) + self._eSecondary(eSolution,srcList)

    def _eDeriv_u(self, src, v, adjoint = False):
        """
        Partial derivative of the total electric field with respect to the solution.

        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect to the field we solved for with a vector
        """

        return Identity()*v

    def _eDeriv_m(self, src, v, adjoint = False):
        """
        Partial derivative of the total electric field with respect to the inversion model. Here, we assume that the primary does not depend on the model. Note that this also includes derivative contributions from the sources.

        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the electric field derivative with respect to the inversion model with a vector
        """

        # assuming primary does not depend on the model
        return Zero()

    def _bPrimary(self, eSolution, srcList):
        bPrimary = np.zeros([self.survey.mesh.nE,eSolution.shape[1]], dtype = complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                bPrimary[:,i] += bp[:,-1]
        return bPrimary

    def _bSecondary(self, eSolution, srcList):
        """
        Primary magnetic flux density from source

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic flux density as defined by the sources
        """
        C = self.mesh.nodalGrad
        b = (C * eSolution)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
        return b

    def _b(self, eSolution, srcList):
        """
        Total magnetic field is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic field
        """
        return self._bPrimary(eSolution, srcList) + self._bSecondary(eSolution, srcList)

    def _bDeriv_u(self, src, v, adjoint = False):
        """
        Derivative of the magnetic flux density with respect to the solution

        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with respect to the field we solved for with a vector
        """
        # bPrimary: no model depenency
        C = self.mesh.nodalGrad
        if adjoint:
            bSecondaryDeriv_u = - 1./(1j*omega(src.freq)) * (C.T * v)
        else:
            bSecondaryDeriv_u = - 1./(1j*omega(src.freq)) * (C * v)
        return bSecondaryDeriv_u

    def _bDeriv_m(self, src, v, adjoint = False):
        """
        Derivative of the magnetic flux density with respect to the inversion model.

        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic flux density derivative with respect to the inversion model with a vector
        """
        # Neither bPrimary nor bSeconary have model dependency => return Zero
        return Zero()


class Fields1D_eTotal(BaseNSEMFields):
    """
    Fields storage for the 1D NSEM solution solved with for a total domain formulation.

    Used in conjuction with Problem1D_eTotal.
    """
    knownFields = {'e_1dSolution':'F'}
    aliasFields = {
                    'e_1d' : ['e_1dSolution','F','_e'],
                    'e_1dPrimary' : ['e_1dSolution','F','_ePrimary'],
                    'e_1dSecondary' : ['e_1dSolution','F','_eSecondary'],
                    'b_1d' : ['e_1dSolution','E','_b'],
                    'b_1dPrimary' : ['e_1dSolution','E','_bPrimary'],
                    'b_1dSecondary' : ['e_1dSolution','E','_bSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        BaseNSEMFields.__init__(self,mesh,survey,**kwargs)

    def _ePrimary(self, eSolution, srcList):
        ePrimary = np.zeros_like(eSolution)
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                ePrimary[:,i] = ep[:,-1]
        return ePrimary

    def _eSecondary(self, eSolution, srcList):
        return eSolution

    def _e(self, eSolution, srcList):
        return self._ePrimary(eSolution,srcList) + self._eSecondary(eSolution,srcList)

    def _eDeriv_u(self, src, v, adjoint = False):
        return v

    def _eDeriv_m(self, src, v, adjoint = False):
        # assuming primary does not depend on the model
        return None

    def _bPrimary(self, eSolution, srcList):
        bPrimary = np.zeros([self.survey.mesh.nE,eSolution.shape[1]], dtype = complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                bPrimary[:,i] += bp[:,-1]
        return bPrimary

    def _bSecondary(self, eSolution, srcList):
        C = self.mesh.nodalGrad
        b = (C * eSolution)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
            # There is no magnetic source in the MT problem
            # S_m, _ = src.eval(self.survey.prob)
            # if S_m is not None:
            #     b[:,i] += 1./(1j*omega(src.freq)) * S_m
        return b

    def _b(self, eSolution, srcList):
        return self._bPrimary(eSolution, srcList) + self._bSecondary(eSolution, srcList)

    def _bSecondaryDeriv_u(self, src, v, adjoint = False):
        C = self.mesh.nodalGrad
        if adjoint:
            return - 1./(1j*omega(src.freq)) * (C.T * v)
        return - 1./(1j*omega(src.freq)) * (C * v)

    def _bSecondaryDeriv_m(self, src, v, adjoint = False):
        # Doesn't depend on m
        # _, S_eDeriv = src.evalDeriv(self.survey.prob, adjoint)
        # S_eDeriv = S_eDeriv(v)
        # if S_eDeriv is not None:
        #     return 1./(1j * omega(src.freq)) * S_eDeriv
        return None

    def _bDeriv_u(self, src, v, adjoint=False):
        # Primary does not depend on u
        return self._bSecondaryDeriv_u(src, v, adjoint)

    def _bDeriv_m(self, src, v, adjoint=False):
        # Assuming the primary does not depend on the model
        return self._bSecondaryDeriv_m(src, v, adjoint)

    def _fDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the fields object wrt u.

        :param NSEMsrc src: NSEM source
        :param numpy.ndarray v: random vector of f_sol.size
        This function stacks the fields derivatives appropriately

        return a vector of size (nreEle+nrbEle)
        """

        de_du = v #Utils.spdiag(np.ones((self.nF,)))
        db_du = self._bDeriv_u(src, v, adjoint)
        # Return the stack
        # This doesn't work...
        return np.vstack((de_du,db_du))

    def _fDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the fields object wrt m.

        This function stacks the fields derivatives appropriately
        """
        return None


###########
# 2D Fields
###########


###########
# 3D Fields
###########
class Fields3D_ePrimSec(BaseNSEMFields):
    """
    Fields storage for the 3D NSEM solution. Labels polarizations by px and py.

        :param SimPEG object mesh: The solution mesh
        :param SimPEG object survey: A survey object
    """
    # Define the known the alias fields
    # Assume that the solution of e on the E.
    ## NOTE: Need to make this more general, to allow for other solutions formats.
    knownFields = {'e_pxSolution':'E','e_pySolution':'E'}
    aliasFields = {
                    'e_px' : ['e_pxSolution','E','_e_px'],
                    'e_pxPrimary' : ['e_pxSolution','E','_e_pxPrimary'],
                    'e_pxSecondary' : ['e_pxSolution','E','_e_pxSecondary'],
                    'e_py' : ['e_pySolution','E','_e_py'],
                    'e_pyPrimary' : ['e_pySolution','E','_e_pyPrimary'],
                    'e_pySecondary' : ['e_pySolution','E','_e_pySecondary'],
                    'b_px' : ['e_pxSolution','F','_b_px'],
                    'b_pxPrimary' : ['e_pxSolution','F','_b_pxPrimary'],
                    'b_pxSecondary' : ['e_pxSolution','F','_b_pxSecondary'],
                    'b_py' : ['e_pySolution','F','_b_py'],
                    'b_pyPrimary' : ['e_pySolution','F','_b_pyPrimary'],
                    'b_pySecondary' : ['e_pySolution','F','_b_pySecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        BaseNSEMFields.__init__(self,mesh,survey,**kwargs)

    def _e_pxPrimary(self, e_pxSolution, srcList):
        e_pxPrimary = np.zeros_like(e_pxSolution)
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                e_pxPrimary[:,i] = ep[:,0]
        return e_pxPrimary

    def _e_pyPrimary(self, e_pySolution, srcList):
        e_pyPrimary = np.zeros_like(e_pySolution)
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                e_pyPrimary[:,i] = ep[:,1]
        return e_pyPrimary

    def _e_pxSecondary(self, e_pxSolution, srcList):
        return e_pxSolution

    def _e_pySecondary(self, e_pySolution, srcList):
        return e_pySolution

    def _e_px(self, e_pxSolution, srcList):
        return self._e_pxPrimary(e_pxSolution,srcList) + self._e_pxSecondary(e_pxSolution,srcList)

    def _e_py(self, e_pySolution, srcList):
        return self._e_pyPrimary(e_pySolution,srcList) + self._e_pySecondary(e_pySolution,srcList)

    #NOTE: For e_p?Deriv_u,
    # v has to be u(2*nE) long for the not adjoint and nE long for adjoint.
    # Returns nE long for not adjoint and 2*nE long for adjoint
    def _e_pxDeriv_u(self, src, v, adjoint = False):
        '''
        Takes the derivative of e_px wrt u
        '''
        if adjoint:
            # adjoint: returns a 2*nE long vector with zero's for py
            return np.vstack((v,np.zeros_like(v)))
        # Not adjoint: return only the px part of the vector
        return v[:len(v)/2]

    def _e_pyDeriv_u(self, src, v, adjoint = False):
        '''
        Takes the derivative of e_py wrt u
        '''
        if adjoint:
            # adjoint: returns a 2*nE long vector with zero's for px
            return np.vstack((np.zeros_like(v),v))
        # Not adjoint: return only the px part of the vector
        return v[len(v)/2::]

    def _e_pxDeriv_m(self, src, v, adjoint = False):
        # assuming primary does not depend on the model
        return None
    def _e_pyDeriv_m(self, src, v, adjoint = False):
        # assuming primary does not depend on the model
        return None

    def _b_pxPrimary(self, e_pxSolution, srcList):
        b_pxPrimary = np.zeros([self.survey.mesh.nF,e_pxSolution.shape[1]], dtype = complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                b_pxPrimary[:,i] += bp[:,0]
        return b_pxPrimary

    def _b_pyPrimary(self, e_pySolution, srcList):
        b_pyPrimary = np.zeros([self.survey.mesh.nF,e_pySolution.shape[1]], dtype = complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                b_pyPrimary[:,i] += bp[:,1]
        return b_pyPrimary

    def _b_pxSecondary(self, e_pxSolution, srcList):
        C = self.mesh.edgeCurl
        b = (C * e_pxSolution)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
            # There is no magnetic source in the NSEM problem
            # S_m, _ = src.eval(self.survey.prob)
            # if S_m is not None:
            #     b[:,i] += 1./(1j*omega(src.freq)) * S_m
        return b

    def _b_pySecondary(self, e_pySolution, srcList):
        C = self.mesh.edgeCurl
        b = (C * e_pySolution)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
            # There is no magnetic source in the NSEM problem
            # S_m, _ = src.eval(self.survey.prob)
            # if S_m is not None:
            #     b[:,i] += 1./(1j*omega(src.freq)) * S_m
        return b

    def _b_px(self, eSolution, srcList):
        return self._b_pxPrimary(eSolution, srcList) + self._b_pxSecondary(eSolution, srcList)

    def _b_py(self, eSolution, srcList):
        return self._b_pyPrimary(eSolution, srcList) + self._b_pySecondary(eSolution, srcList)

    # NOTE: v needs to be length 2*nE to account for both polarizations
    def _b_pxSecondaryDeriv_u(self, src, v, adjoint = False):
        # C = sp.kron(self.mesh.edgeCurl,[[1,0],[0,0]])
        C = sp.hstack((self.mesh.edgeCurl,Utils.spzeros(self.mesh.nF,self.mesh.nE))) # This works for adjoint = None
        if adjoint:
            return - 1./(1j*omega(src.freq)) * (C.T * v)
        return - 1./(1j*omega(src.freq)) * (C * v)

    def _b_pySecondaryDeriv_u(self, src, v, adjoint = False):
        # C = sp.kron(self.mesh.edgeCurl,[[0,0],[0,1]])
        C = sp.hstack((Utils.spzeros(self.mesh.nF,self.mesh.nE),self.mesh.edgeCurl)) # This works for adjoint = None
        if adjoint:
            return - 1./(1j*omega(src.freq)) * (C.T * v)
        return - 1./(1j*omega(src.freq)) * (C * v)

    def _b_pxSecondaryDeriv_m(self, src, v, adjoint = False):
        # Doesn't depend on m
        # _, S_eDeriv = src.evalDeriv(self.survey.prob, adjoint)
        # S_eDeriv = S_eDeriv(v)
        # if S_eDeriv is not None:
        #     return 1./(1j * omega(src.freq)) * S_eDeriv
        return None

    def _b_pySecondaryDeriv_m(self, src, v, adjoint = False):
        # Doesn't depend on m
        # _, S_eDeriv = src.evalDeriv(self.survey.prob, adjoint)
        # S_eDeriv = S_eDeriv(v)
        # if S_eDeriv is not None:
        #     return 1./(1j * omega(src.freq)) * S_eDeriv
        return None

    def _b_pxDeriv_u(self, src, v, adjoint=False):
        # Primary does not depend on u
        return self._b_pxSecondaryDeriv_u(src, v, adjoint)

    def _b_pyDeriv_u(self, src, v, adjoint=False):
        # Primary does not depend on u
        return self._b_pySecondaryDeriv_u(src, v, adjoint)

    def _b_pxDeriv_m(self, src, v, adjoint=False):
        # Assuming the primary does not depend on the model
        return self._b_pxSecondaryDeriv_m(src, v, adjoint)

    def _b_pyDeriv_m(self, src, v, adjoint=False):
        # Assuming the primary does not depend on the model
        return self._b_pySecondaryDeriv_m(src, v, adjoint)

    def _f_pxDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the fields object wrt u.

        :param NSEMsrc src: NSEM source
        :param numpy.ndarray v: random vector of f_sol.size
        This function stacks the fields derivatives appropriately

        return a vector of size (nreEle+nrbEle)
        """

        de_du = v #Utils.spdiag(np.ones((self.nF,)))
        db_du = self._b_pxDeriv_u(src, v, adjoint)
        # Return the stack
        # This doesn't work...
        return np.vstack((de_du,db_du))

    def _f_pyDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the fields object wrt u.

        :param NSEMsrc src: NSEM source
        :param numpy.ndarray v: random vector of f_sol.size
        This function stacks the fields derivatives appropriately

        return a vector of size (nreEle+nrbEle)
        """

        de_du = v #Utils.spdiag(np.ones((self.nF,)))
        db_du = self._b_pyDeriv_u(src, v, adjoint)
        # Return the stack
        # This doesn't work...
        return np.vstack((de_du,db_du))

    def _f_pxDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the fields object wrt m.

        This function stacks the fields derivatives appropriately
        """
        # The fields have no dependance to the model.
        return None

    def _f_pyDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the fields object wrt m.

        This function stacks the fields derivatives appropriately
        """
        # The fields have no dependance to the model.
        return None