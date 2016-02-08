import numpy as np
import scipy.sparse as sp
import SimPEG
from SimPEG import Utils
from SimPEG.EM.Utils import omega
from SimPEG.Utils import Zero, Identity


class Fields(SimPEG.Problem.Fields):
    """
    
    Fancy Field Storage for a FDEM survey. Only one field type is stored for
    each problem, the rest are computed. The fields obejct acts like an array and is indexed by

    .. code-block:: python

        f = problem.fields(m)
        e = f[srcList,'e']
        b = f[srcList,'b']

    If accessing all sources for a given field, use the :code:`:`

    .. code-block:: python

        f = problem.fields(m)
        e = f[:,'e']
        b = f[:,'b']

    The array returned will be size (nE or nF, nSrcs :math:`\\times` nFrequencies)
    """

    knownFields = {}
    dtype = complex

class Fields_e(Fields):
    """
    Fields object for Problem_e. 

    :param Mesh mesh: mesh
    :param Survey survey: survey 
    """

    knownFields = {'eSolution':'E'}
    aliasFields = {
                    'e' : ['eSolution','E','_e'],
                    'ePrimary' : ['eSolution','E','_ePrimary'],
                    'eSecondary' : ['eSolution','E','_eSecondary'],
                    'b' : ['eSolution','F','_b'],
                    'bPrimary' : ['eSolution','F','_bPrimary'],
                    'bSecondary' : ['eSolution','F','_bSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        Fields.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl

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
            ep = src.ePrimary(self.prob)
            ePrimary[:,i] = ePrimary[:,i] + ep
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

    def _e(self, eSolution, srcList):
        """
        Total electric field is sum of primary and secondary 
        
        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total electric field
        """

        return self._ePrimary(eSolution,srcList) + self._eSecondary(eSolution,srcList)

    def _eDeriv_u(self, src, v, adjoint = False):
        """
        Derivative of the total electric field with respect to the thing we 
        solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect to the field we solved for with a vector
        """

        return Identity()*v

    def _eDeriv_m(self, src, v, adjoint = False):
        """
        Derivative of the total electric field with respect to the inversion model. Here, we assume that the primary does not depend on the model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the electric field derivative with respect to the inversion model with a vector
        """

        # assuming primary does not depend on the model
        return Zero()

    def _bPrimary(self, eSolution, srcList):
        """
        Primary magnetic flux density from source

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic flux density as defined by the sources
        """

        bPrimary = np.zeros([self._edgeCurl.shape[0],eSolution.shape[1]],dtype = complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.prob)
            bPrimary[:,i] = bPrimary[:,i] + bp
        return bPrimary

    def _bSecondary(self, eSolution, srcList):
        """
        Secondary magnetic flux density from eSolution

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux density
        """

        C = self._edgeCurl
        b = (C * eSolution)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
            S_m, _ = src.eval(self.prob)
            b[:,i] = b[:,i]+ 1./(1j*omega(src.freq)) * S_m
        return b

    def _bSecondaryDeriv_u(self, src, v, adjoint = False):
        """
        Derivative of the secondary magnetic flux density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary magnetic flux density with respect to the field we solved for with a vector
        """

        C = self._edgeCurl
        if adjoint:
            return - 1./(1j*omega(src.freq)) * (C.T * v)
        return - 1./(1j*omega(src.freq)) * (C * v)

    def _bSecondaryDeriv_m(self, src, v, adjoint = False):
        """
        Derivative of the secondary magnetic flux density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the secondary magnetic flux density derivative with respect to the inversion model with a vector
        """

        S_mDeriv, _ = src.evalDeriv(self.prob, v, adjoint)
        return 1./(1j * omega(src.freq)) * S_mDeriv

    def _b(self, eSolution, srcList):
        """
        Total magnetic flux density is sum of primary and secondary 
        
        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic flux density 
        """

        return self._bPrimary(eSolution, srcList) + self._bSecondary(eSolution, srcList)

    def _bDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic flux density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with respect to the field we solved for with a vector
        """

        # Primary does not depend on u
        return self._bSecondaryDeriv_u(src, v, adjoint)

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic flux density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the magnetic flux density derivative with respect to the inversion model with a vector
        """

        # Assuming the primary does not depend on the model
        return self._bSecondaryDeriv_m(src, v, adjoint)


class Fields_b(Fields):
    """
    Fields object for Problem_b. 

    :param Mesh mesh: mesh
    :param Survey survey: survey 
    """

    knownFields = {'bSolution':'F'}
    aliasFields = {
                    'b' : ['bSolution','F','_b'],
                    'bPrimary' : ['bSolution','F','_bPrimary'],
                    'bSecondary' : ['bSolution','F','_bSecondary'],
                    'e' : ['bSolution','E','_e'],
                    'ePrimary' : ['bSolution','E','_ePrimary'],
                    'eSecondary' : ['bSolution','E','_eSecondary'],
                  }

    def __init__(self,mesh,survey,**kwargs):
        Fields.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MfMui = self.survey.prob.MfMui
        self._MeSigmaIDeriv = self.survey.prob.MeSigmaIDeriv
        self._Me = self.survey.prob.Me

    def _bPrimary(self, bSolution, srcList):
        """
        Primary magnetic flux density from source

        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """

        bPrimary = np.zeros_like(bSolution)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.prob)
            bPrimary[:,i] = bPrimary[:,i] + bp
        return bPrimary

    def _bSecondary(self, bSolution, srcList):
        """
        Secondary magnetic flux density is the thing we solved for

        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux density
        """

        return bSolution

    def _b(self, bSolution, srcList):
        """
        Total magnetic flux density is sum of primary and secondary 
        
        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic flux density 
        """

        return self._bPrimary(bSolution, srcList) + self._bSecondary(bSolution, srcList)

    def _bDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic flux density with respect to the thing we 
        solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with respect to the field we solved for with a vector
        """
        return Identity()*v

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic flux density with respect to the inversion model. Here, we assume that the primary does not depend on the model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the magnetic flux density derivative with respect to the inversion model with a vector
        """

        # assuming primary does not depend on the model
        return Zero()

    def _ePrimary(self, bSolution, srcList):
        """
        Primary electric field from source

        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """

        ePrimary = np.zeros([self._edgeCurl.shape[1],bSolution.shape[1]],dtype = complex)
        for i,src in enumerate(srcList):
            ep = src.ePrimary(self.prob)
            ePrimary[:,i] = ePrimary[:,i] + ep
        return ePrimary

    def _eSecondary(self, bSolution, srcList):
        """
        Secondary electric field from bSolution

        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field
        """

        e = self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * bSolution))
        for i,src in enumerate(srcList):
            _,S_e = src.eval(self.prob)
            e[:,i] = e[:,i]+ -self._MeSigmaI * S_e
        return e

    def _eSecondaryDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the secondary electric field with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary electric field with respect to the field we solved for with a vector
        """

        if not adjoint:
            return self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * v) )
        else:
            return self._MfMui.T * (self._edgeCurl * (self._MeSigmaI.T * v))

    def _eSecondaryDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the secondary electric field with respect to the inversion model 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary electric field with respect to the model with a vector
        """

        bSolution = self[[src],'bSolution']
        _,S_e = src.eval(self.prob)
        Me = self._Me

        if adjoint:
            Me = Me.T

        w = self._edgeCurl.T * (self._MfMui * bSolution)
        w = w - Utils.mkvc(Me * S_e,2)

        if not adjoint:
            de_dm = self._MeSigmaIDeriv(w) * v
        elif adjoint:
            de_dm = self._MeSigmaIDeriv(w).T * v

        _, S_eDeriv = src.evalDeriv(self.prob, v, adjoint)

        de_dm = de_dm - self._MeSigmaI * S_eDeriv

        return de_dm

    def _e(self, bSolution, srcList):
        """
        Total electric field is sum of primary and secondary 
        
        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total electric field
        """

        return self._ePrimary(bSolution, srcList) + self._eSecondary(bSolution, srcList)

    def _eDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the total electric field with respect to the thing we solved for 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect to the field we solved for with a vector
        """

        return self._eSecondaryDeriv_u(src, v, adjoint)

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the total electric field density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the electric field derivative with respect to the inversion model with a vector
        """

        # assuming primary doesn't depend on model
        return self._eSecondaryDeriv_m(src, v, adjoint)


class Fields_j(Fields):
    """
    Fields object for Problem_j. 

    :param Mesh mesh: mesh
    :param Survey survey: survey 
    """

    knownFields = {'jSolution':'F'}
    aliasFields = {
                    'j' : ['jSolution','F','_j'],
                    'jPrimary' : ['jSolution','F','_jPrimary'],
                    'jSecondary' : ['jSolution','F','_jSecondary'],
                    'h' : ['jSolution','E','_h'],
                    'hPrimary' : ['jSolution','E','_hPrimary'],
                    'hSecondary' : ['jSolution','E','_hSecondary'],
                  }

    def __init__(self,mesh,survey,**kwargs):
        Fields.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho
        self._MfRhoDeriv = self.survey.prob.MfRhoDeriv
        self._Me = self.survey.prob.Me

    def _jPrimary(self, jSolution, srcList):
        """
        Primary current density from source

        :param numpy.ndarray jSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary current density as defined by the sources
        """

        jPrimary = np.zeros_like(jSolution,dtype = complex)
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.prob)
            jPrimary[:,i] = jPrimary[:,i] + jp
        return jPrimary

    def _jSecondary(self, jSolution, srcList):
        """
        Secondary current density is the thing we solved for

        :param numpy.ndarray jSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary current density
        """

        return jSolution

    def _j(self, jSolution, srcList):
        """
        Total current density is sum of primary and secondary 
        
        :param numpy.ndarray jSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total current density 
        """

        return self._jPrimary(jSolution, srcList) + self._jSecondary(jSolution, srcList)

    def _jDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the total current density with respect to the thing we 
        solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect to the field we solved for with a vector
        """

        return Identity()*v

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the total current density with respect to the inversion model. Here, we assume that the primary does not depend on the model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the current density derivative with respect to the inversion model with a vector
        """
        # assuming primary does not depend on the model
        return Zero()

    def _hPrimary(self, jSolution, srcList):
        """
        Primary magnetic field from source

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic field as defined by the sources
        """

        hPrimary = np.zeros([self._edgeCurl.shape[1],jSolution.shape[1]],dtype = complex)
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.prob)
            hPrimary[:,i] = hPrimary[:,i] + hp
        return hPrimary

    def _hSecondary(self, jSolution, srcList):
        """
        Secondary magnetic field from bSolution

        :param numpy.ndarray jSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic field
        """

        h =  self._MeMuI * (self._edgeCurl.T * (self._MfRho * jSolution) )
        for i, src in enumerate(srcList):
            h[:,i] *= -1./(1j*omega(src.freq))
            S_m,_ = src.eval(self.prob)
            h[:,i] = h[:,i]+ 1./(1j*omega(src.freq)) * self._MeMuI * (S_m)
        return h

    def _hSecondaryDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the secondary magnetic field with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary magnetic field with respect to the field we solved for with a vector
        """

        if not adjoint:
            return  -1./(1j*omega(src.freq)) * self._MeMuI * (self._edgeCurl.T * (self._MfRho * v) )
        elif adjoint:
            return  -1./(1j*omega(src.freq)) * self._MfRho.T * (self._edgeCurl * ( self._MeMuI.T * v))

    def _hSecondaryDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the secondary magnetic field with respect to the inversion model 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary magnetic field with respect to the model with a vector
        """

        jSolution = self[[src],'jSolution']
        MeMuI = self._MeMuI
        C = self._edgeCurl
        MfRho = self._MfRho
        MfRhoDeriv = self._MfRhoDeriv
        Me = self._Me

        if not adjoint:
            hDeriv_m =  -1./(1j*omega(src.freq)) * MeMuI * (C.T * (MfRhoDeriv(jSolution)*v ) )
        elif adjoint:
            hDeriv_m =  -1./(1j*omega(src.freq)) * MfRhoDeriv(jSolution).T * ( C * (MeMuI.T * v ) )

        S_mDeriv,_ = src.evalDeriv(self.prob, adjoint = adjoint)

        if not adjoint:
            S_mDeriv = S_mDeriv(v)
            hDeriv_m = hDeriv_m + 1./(1j*omega(src.freq)) * MeMuI * (Me * S_mDeriv)
        elif adjoint:
            S_mDeriv = S_mDeriv(Me.T * (MeMuI.T * v))
            hDeriv_m = hDeriv_m + 1./(1j*omega(src.freq)) * S_mDeriv
        return hDeriv_m


    def _h(self, jSolution, srcList):
        """
        Total magnetic field is sum of primary and secondary 
        
        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic field
        """

        return self._hPrimary(jSolution, srcList) + self._hSecondary(jSolution, srcList)

    def _hDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic field with respect to the thing we solved for 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect to the field we solved for with a vector
        """

        return self._hSecondaryDeriv_u(src, v, adjoint)

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic field density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic field derivative with respect to the inversion model with a vector
        """

        # assuming the primary doesn't depend on the model
        return self._hSecondaryDeriv_m(src, v, adjoint)


class Fields_h(Fields):
    """
    Fields object for Problem_h. 

    :param Mesh mesh: mesh
    :param Survey survey: survey 
    """

    knownFields = {'hSolution':'E'}
    aliasFields = {
                    'h' : ['hSolution','E','_h'],
                    'hPrimary' : ['hSolution','E','_hPrimary'],
                    'hSecondary' : ['hSolution','E','_hSecondary'],
                    'j' : ['hSolution','F','_j'],
                    'jPrimary' : ['hSolution','F','_jPrimary'],
                    'jSecondary' : ['hSolution','F','_jSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        Fields.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho

    def _hPrimary(self, hSolution, srcList):
        """
        Primary magnetic field from source

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic field as defined by the sources
        """

        hPrimary = np.zeros_like(hSolution,dtype = complex)
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.prob)
            hPrimary[:,i] = hPrimary[:,i] + hp
        return hPrimary

    def _hSecondary(self, hSolution, srcList):
        """
        Secondary magnetic field is the thing we solved for

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic field
        """

        return hSolution

    def _h(self, hSolution, srcList):
        """
        Total magnetic field is sum of primary and secondary 
        
        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic field
        """

        return self._hPrimary(hSolution, srcList) + self._hSecondary(hSolution, srcList)

    def _hDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic field with respect to the thing we 
        solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect to the field we solved for with a vector
        """

        return Identity()*v

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the total magnetic field with respect to the inversion model. Here, we assume that the primary does not depend on the model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the magnetic field derivative with respect to the inversion model with a vector
        """

        # assuming primary does not depend on the model
        return Zero()

    def _jPrimary(self, hSolution, srcList):
        """
        Primary current density from source

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary current density as defined by the sources
        """

        jPrimary = np.zeros([self._edgeCurl.shape[0], hSolution.shape[1]], dtype = complex)
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.prob)
            jPrimary[:,i] = jPrimary[:,i] + jp
        return jPrimary

    def _jSecondary(self, hSolution, srcList):
        """
        Secondary current density from eSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary current density
        """

        j = self._edgeCurl*hSolution
        for i, src in enumerate(srcList):
            _,S_e = src.eval(self.prob)
            j[:,i] = j[:,i]+ -S_e
        return j

    def _jSecondaryDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the secondary current density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary current density with respect to the field we solved for with a vector
        """

        if not adjoint:
            return self._edgeCurl*v
        elif adjoint:
            return self._edgeCurl.T*v

    def _jSecondaryDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the secondary current density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the secondary current density derivative with respect to the inversion model with a vector
        """

        _,S_eDeriv = src.evalDeriv(self.prob, v, adjoint)
        return -S_eDeriv

    def _j(self, hSolution, srcList):
        """
        Total current density is sum of primary and secondary 
        
        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total current density 
        """

        return self._jPrimary(hSolution, srcList) + self._jSecondary(hSolution, srcList)

    def _jDeriv_u(self, src, v, adjoint=False):
        """
        Derivative of the total current density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect to the field we solved for with a vector
        """
        return self._jSecondaryDeriv_u(src,v,adjoint)

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the total current density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the current density with respect to the inversion model with a vector
        """

        # assuming the primary does not depend on the model
        return self._jSecondaryDeriv_m(src,v,adjoint)
