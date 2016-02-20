import numpy as np
import scipy.sparse as sp
import SimPEG
from SimPEG import Utils
from SimPEG.EM.Utils import omega
from SimPEG.Utils import Zero, Identity, sdiag


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

    def _e(self, solution, srcList):
        """
        Total electric field is sum of primary and secondary 
        
        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total electric field
        """
        if getattr(self, '_ePrimary', None) is None or getattr(self, '_eSecondary', None) is None: 
            raise NotImplementedError ('Getting e from %s is not implemented' %self.knownFields.keys()[0])

        return self._ePrimary(solution,srcList) + self._eSecondary(solution,srcList)

    def _b(self, solution, srcList):
        """
        Total magnetic flux density is sum of primary and secondary 
        
        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic flux density 
        """
        if getattr(self, '_bPrimary', None) is None or getattr(self, '_bSecondary', None) is None: 
            raise NotImplementedError ('Getting b from %s is not implemented' %self.knownFields.keys()[0])

        return self._bPrimary(solution, srcList) + self._bSecondary(solution, srcList)

    def _h(self, solution, srcList):
        """
        Total magnetic field is sum of primary and secondary 
        
        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic field
        """
        if getattr(self, '_hPrimary', None) is None or getattr(self, '_hSecondary', None) is None: 
            raise NotImplementedError ('Getting h from %s is not implemented' %self.knownFields.keys()[0])

        return self._hPrimary(solution, srcList) + self._hSecondary(solution, srcList)

    def _j(self, solution, srcList):
        """
        Total current density is sum of primary and secondary 
        
        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total current density 
        """
        if getattr(self, '_jPrimary', None) is None or getattr(self, '_jSecondary', None) is None: 
            raise NotImplementedError ('Getting j from %s is not implemented' %self.knownFields.keys()[0])

        return self._jPrimary(solution, srcList) + self._jSecondary(solution, srcList)

    def _eDeriv(self, src, du_dm_v, v, adjoint = False):
        """
        Total derivative of e with respect to the inversion model. Returns :math:`d\mathbf{e}/d\mathbf{m}` for forward and (:math:`d\mathbf{e}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for the adjoint

        :param Src src: sorce
        :param numpy.ndarray du_dm_v: derivative of the solution vector with respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if getattr(self, '_eDeriv_u', None) is None or getattr(self, '_eDeriv_m', None) is None: 
            raise NotImplementedError ('Getting eDerivs from %s is not implemented' %self.knownFields.keys()[0])

        if adjoint:
            return self._eDeriv_u(src, v, adjoint), self._eDeriv_m(src, v, adjoint)
        return self._eDeriv_u(src, du_dm_v, adjoint) + self._eDeriv_m(src, v, adjoint)

    def _bDeriv(self, src, du_dm_v, v, adjoint = False):
        """
        Total derivative of b with respect to the inversion model. Returns :math:`d\mathbf{b}/d\mathbf{m}` for forward and (:math:`d\mathbf{b}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for the adjoint

        :param Src src: sorce
        :param numpy.ndarray du_dm_v: derivative of the solution vector with respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if getattr(self, '_bDeriv_u', None) is None or getattr(self, '_bDeriv_m', None) is None: 
            raise NotImplementedError ('Getting bDerivs from %s is not implemented' %self.knownFields.keys()[0])

        if adjoint:
            return self._bDeriv_u(src, v, adjoint), self._bDeriv_m(src, v, adjoint)
        return self._bDeriv_u(src, du_dm_v, adjoint) + self._bDeriv_m(src, v, adjoint)

    def _hDeriv(self, src, du_dm_v, v, adjoint = False):
        """
        Total derivative of h with respect to the inversion model. Returns :math:`d\mathbf{h}/d\mathbf{m}` for forward and (:math:`d\mathbf{h}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for the adjoint

        :param Src src: sorce
        :param numpy.ndarray du_dm_v: derivative of the solution vector with respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if getattr(self, '_hDeriv_u', None) is None or getattr(self, '_hDeriv_m', None) is None: 
            raise NotImplementedError ('Getting hDerivs from %s is not implemented' %self.knownFields.keys()[0])

        if adjoint: 
            return self._hDeriv_u(src, v, adjoint), self._hDeriv_m(src, v, adjoint)
        return self._hDeriv_u(src, du_dm_v, adjoint) + self._hDeriv_m(src, v, adjoint)

    def _jDeriv(self, src, du_dm_v, v, adjoint = False):
        """
        Total derivative of j with respect to the inversion model. Returns :math:`d\mathbf{j}/d\mathbf{m}` for forward and (:math:`d\mathbf{j}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for the adjoint

        :param Src src: sorce
        :param numpy.ndarray du_dm_v: derivative of the solution vector with respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if getattr(self, '_jDeriv_u', None) is None or getattr(self, '_jDeriv_m', None) is None: 
            raise NotImplementedError ('Getting jDerivs from %s is not implemented' %self.knownFields.keys()[0])

        if adjoint:
            return self._jDeriv_u(src, v, adjoint), self._jDeriv_m(src, v, adjoint)
        return self._jDeriv_u(src, du_dm_v, adjoint) + self._jDeriv_m(src, v, adjoint)


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
                    'bSecondary' : ['eSolution','F','_bSecondary'],
                    'j' : ['eSolution','CCV','_j'],
                    'h' : ['eSolution','CCV','_h'],
                  }

    def __init__(self, mesh, survey, **kwargs):
        Fields.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._nC = self.survey.prob.mesh.nC
        self._MeSigma = self.survey.prob.MeSigma
        self._MeSigmaDeriv = self.survey.prob.MeSigmaDeriv

    def _GLoc(self, fieldType):
        if fieldType == 'e':
            return 'E'
        elif fieldType == 'b':
            return 'F'
        elif (fieldType == 'h') or (fieldType == 'j'):
            return 'CCV'
        else:
            raise Exception('Field type must be e, b, h, j')

    def _ePrimary(self, eSolution, srcList):
        """
        Primary electric field from source

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """

        ePrimary = np.zeros([self.prob.mesh.nE,len(srcList)])
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
        ind = self.prob.survey.getSourceIndex(srcList)
        return eSolution[:,ind]

    def _eDeriv_u(self, src, v, adjoint = False):
        """
        Partial derivative of the total electric field with respect to the thing we 
        solved for.
        
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

    def _bSecondaryDeriv_u(self, src, du_dm_v, adjoint = False):
        """
        Derivative of the secondary magnetic flux density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary magnetic flux density with respect to the field we solved for with a vector
        """

        C = self._edgeCurl
        if adjoint:
            return - 1./(1j*omega(src.freq)) * (C.T * du_dm_v)
        return - 1./(1j*omega(src.freq)) * (C * du_dm_v)


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


    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total magnetic flux density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with respect to the field we solved for with a vector
        """

        return self._bSecondaryDeriv_u(src, du_dm_v, adjoint)

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic flux density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the magnetic flux density derivative with respect to the inversion model with a vector
        """

        # Assuming the primary does not depend on the model
        return self._bSecondaryDeriv_m(src, eSolution, v, adjoint)

    def _j(self, eSolution, srcList):
        aveE2CCV = self._aveE2CCV
        n = int(aveE2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        Sigma = self._MeSigma
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))
        return VI * (aveE2CCV * (Sigma *eSolution) )

    def _jDeriv_u(self, src, v, adjoint = False):
        eSolution = self._eSecondary
        aveE2CCV = self._aveE2CCV
        n = int(aveE2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        Sigma = self._MeSigma

        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))

        if not adjoint: 
            return VI * (aveE2CCV * (Sigma * (self._eDeriv_u(src, v, adjoint) ) ) )
        return  self._eDeriv_u(src, Sigma.T * (aveE2CCV.T * (VI.T * v) ), adjoint)

    def _jDeriv_m(self, src, v, adjoint = False):
        eSolution = self._fields['eSolution']

        aveE2CCV = self._aveE2CCV
        Sigma = self._MeSigma
        SigmaDeriv = self._MeSigmaDeriv
        e = self._e(eSolution, [src])

        n = int(aveE2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))

        if not adjoint:
            return VI * (aveE2CCV * ( self._eDeriv_m(src, v, adjoint=adjoint) + SigmaDeriv(e) * v))
        return SigmaDeriv(aveE2CCV.T * (VI.T * e), adjoint=adjoint) * v + self._eDeriv_m(src, aveE2CCV.T * (VI.T * v), adjoint=adjoint)


    def _h(self, eSolution, srcList):
        b = self._b(eSolution, srcList)
        Mui = self.survey.prob.MfMui
        aveF2CCV = self._aveF2CCV
        n = int(aveF2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        # Mui = sdiag(sp.kron(np.ones(n), mui))
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))


        return VI * (aveF2CCV * (Mui * b))



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
                    'j' : ['bSolution','C','_j'],
                    'h' : ['bSolution','C','_h'],
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
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._sigma = self.survey.prob.curModel.sigma
        self._mui = self.survey.prob.curModel.mui
        self._nC = self.survey.prob.mesh.nC



    def _GLoc(self,fieldType):
        if fieldType == 'e':
            return 'E'
        elif fieldType == 'b':
            return 'F'
        elif (fieldType == 'h') or (fieldType == 'j'):
            return'CCV'
        else:
            raise Exception('Field type must be e, b, h, j')

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

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total magnetic flux density with respect to the thing we 
        solved for.
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with respect to the field we solved for with a vector
        """

        return Identity()*du_dm_v

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic flux density with respect to the inversion model. Here, we assume that the primary does not depend on the model. Note that this also includes derivative contributions from the sources. 
        
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

    def _eSecondaryDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the secondary electric field with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary electric field with respect to the field we solved for with a vector
        """

        if not adjoint:
            return self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * du_dm_v) )
        else:
            return self._MfMui.T * (self._edgeCurl * (self._MeSigmaI.T * du_dm_v))


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

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total electric field with respect to the thing we solved for 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect to the field we solved for with a vector
        """
        return self._eSecondaryDeriv_u(src, du_dm_v, adjoint)

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total electric field density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the electric field derivative with respect to the inversion model with a vector
        """
        # assuming primary doesn't depend on model
        return self._eSecondaryDeriv_m(src, bSolution, v, adjoint)

    def _j(self, bSolution, srcList):
        sigma = self._sigma
        aveE2CCV = self._aveE2CCV
        n = int(aveE2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        # Sigma = sdiag(np.kron(np.ones(n), sigma))
        Sigma = self.prob.MeSigma
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))

        e = self._e(bSolution, srcList)

        return VI * (aveE2CCV * (Sigma *e) )

    def _h(self, bSolution, srcList):
        b = self._b(bSolution, srcList)
        Mui = self.survey.prob.MfMui
        aveF2CCV = self._aveF2CCV
        n = int(aveF2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        # Mui = sdiag(sp.kron(np.ones(n), mui))
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))


        return VI * (aveF2CCV * (Mui * b))


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
                    'e' : ['jSolution','C','_e'],
                    'b' : ['jSolution','C','_b'],
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
        self._rho = self.survey.prob.curModel.rho
        self._mu = self.survey.prob.curModel.mui
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._nC = self.survey.prob.mesh.nC

    def _GLoc(self,fieldType):
        if fieldType == 'h':
            return 'E'
        elif fieldType == 'j':
            return 'F'
        elif (fieldType == 'e') or (fieldType == 'b'):
            return 'CCV'
        else:
            raise Exception('Field type must be e, b, h, j')

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


    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the thing we 
        solved for.
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect to the field we solved for with a vector
        """

        return Identity()*du_dm_v


    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the inversion model. Here, we assume that the primary does not depend on the model. Note that this also includes derivative contributions from the sources. 
        
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


    def _hSecondaryDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the secondary magnetic field with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary magnetic field with respect to the field we solved for with a vector
        """

        if not adjoint:
            return  -1./(1j*omega(src.freq)) * self._MeMuI * (self._edgeCurl.T * (self._MfRho * du_dm_v) )
        elif adjoint:
            return  -1./(1j*omega(src.freq)) * self._MfRho.T * (self._edgeCurl * ( self._MeMuI.T * du_dm_v))


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


    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total magnetic field with respect to the thing we solved for 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect to the field we solved for with a vector
        """

        return self._hSecondaryDeriv_u(src, du_dm_v, adjoint)

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic field density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic field derivative with respect to the inversion model with a vector
        """

        # assuming the primary doesn't depend on the model
        return self._hSecondaryDeriv_m(src, u, v, adjoint)

    def _e(self, jSolution, srcList):
        rho = self._rho
        aveF2CCV = self._aveF2CCV
        n = int(aveF2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        
        Rho = self.prob.MfRho
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))


        j = self._j(jSolution, srcList)

        return VI * (aveF2CCV * (Rho *  j)) 

    def _eDeriv_u(self, src, u, v, adjoint=False):
        raise NotImplementedError

    def _eDeriv_m(self, src, u, v, adjoint=False):
        raise NotImplementedError

    def _b(self, jSolution, srcList):
        h = self._h(jSolution, srcList)
        Mu = self.prob.MeMu
        aveE2CCV = self._aveE2CCV
        n = int(aveE2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        # Mu = sdiag(sp.kron(np.ones(n), mu))
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))

        return VI * (aveE2CCV * (Mu * h))


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
                    'jSecondary' : ['hSolution','F','_jSecondary'],
                    'e' : ['hSolution','C','_e'],
                    'b' : ['hSolution','C','_b'],
                  }

    def __init__(self,mesh,survey,**kwargs):
        Fields.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho
        self._rho = self.survey.prob.curModel.rho
        self._mu = self.survey.prob.curModel.mui
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._nC = self.survey.prob.mesh.nC

    def _GLoc(self,fieldType):
        if fieldType == 'h':
            return 'E'
        elif fieldType == 'j':
            return 'F'
        elif (fieldType == 'e') or (fieldType == 'b'):
            return 'CCV'
        else:
            raise Exception('Field type must be e, b, h, j')

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


    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total magnetic field with respect to the thing we 
        solved for.
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect to the field we solved for with a vector
        """

        return Identity()*du_dm_v

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic field with respect to the inversion model. Here, we assume that the primary does not depend on the model. Note that this also includes derivative contributions from the sources. 
        
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

    def _jSecondaryDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the secondary current density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the secondary current density with respect to the field we solved for with a vector
        """

        if not adjoint:
            return self._edgeCurl*du_dm_v
        elif adjoint:
            return self._edgeCurl.T*du_dm_v


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

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the thing we solved for
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect to the field we solved for with a vector
        """

        return self._jSecondaryDeriv_u(src,du_dm_v,adjoint)

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the inversion model. 
        
        :param SimPEG.EM.FDEM.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the current density with respect to the inversion model with a vector
        """

        # assuming the primary does not depend on the model
        return self._jSecondaryDeriv_m(src,v,adjoint)
    
    def _e(self, hSolution, srcList):
        rho = self._rho
        aveF2CCV = self._aveF2CCV
        n = int(aveF2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        
        Rho = self.prob.MfRho
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))


        j = self._j(hSolution, srcList)

        return VI * (aveF2CCV * (Rho *  j)) 

    def _eDeriv_u(self, src, u, v, adjoint=False):
        raise NotImplementedError

    def _eDeriv_m(self, src, u, v, adjoint=False):
        raise NotImplementedError

    def _b(self, hSolution, srcList):
        h = self._h(hSolution, srcList)
        Mu = self.prob.MeMu
        aveE2CCV = self._aveE2CCV
        n = int(aveE2CCV.shape[0] / self._nC) #TODO: This is a bit sloppy
        # Mu = sdiag(sp.kron(np.ones(n), mu))
        VI = sdiag(1./np.kron(np.ones(n), self.prob.mesh.vol))

        return VI * (aveE2CCV * (Mu * h))
