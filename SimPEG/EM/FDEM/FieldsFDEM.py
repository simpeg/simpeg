import numpy as np
import scipy.sparse as sp
import SimPEG
from SimPEG import Utils
from SimPEG.EM.Utils import omega
from SimPEG.Utils import Zero, Identity, sdiag


class FieldsFDEM(SimPEG.Problem.Fields):
    """

    Fancy Field Storage for a FDEM survey. Only one field type is stored for
    each problem, the rest are computed. The fields object acts like an array
    and is indexed by

    .. code-block:: python

        f = problem.fields(m)
        e = f[srcList,'e']
        b = f[srcList,'b']

    If accessing all sources for a given field, use the :code:`:`

    .. code-block:: python

        f = problem.fields(m)
        e = f[:,'e']
        b = f[:,'b']

    The array returned will be size (nE or nF, nSrcs :math:`\\times`
    nFrequencies)
    """

    knownFields = {}
    dtype = complex

    def _GLoc(self, fieldType):
        """Grid location of the fieldType"""
        return self.aliasFields[fieldType][1]

    def _e(self, solution, srcList):
        """
        Total electric field is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total electric field
        """
        if (
            getattr(self, '_ePrimary', None) is None or
            getattr(self, '_eSecondary', None) is None
        ):
            raise NotImplementedError(
                'Getting e from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0]
                )
            )

        return (
            self._ePrimary(solution, srcList) +
            self._eSecondary(solution, srcList)
        )

    def _b(self, solution, srcList):
        """
        Total magnetic flux density is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic flux density
        """
        if (
            getattr(self, '_bPrimary', None) is None or
            getattr(self, '_bSecondary', None) is None
        ):
            raise NotImplementedError(
                'Getting b from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0]
                )
            )

        return (
            self._bPrimary(solution, srcList) +
            self._bSecondary(solution, srcList)
            )

    def _bSecondary(self, solution, srcList):
        """
        Total magnetic flux density is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic flux density
        """
        if getattr(self, '_bSecondary', None) is None:
            raise NotImplementedError(
                'Getting b from {} is not implemented'.format(
                    self.knownFields.keys()[0]
                    )
                )

        return self._bSecondary(solution, srcList)

    def _h(self, solution, srcList):
        """
        Total magnetic field is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total magnetic field
        """
        if (
            getattr(self, '_hPrimary', None) is None or
            getattr(self, '_hSecondary', None) is None
        ):
            raise NotImplementedError(
                'Getting h from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0])
                )

        return (
            self._hPrimary(solution, srcList) +
            self._hSecondary(solution, srcList)
            )

    def _j(self, solution, srcList):
        """
        Total current density is sum of primary and secondary

        :param numpy.ndarray solution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: total current density
        """
        if (
            getattr(self, '_jPrimary', None) is None or
            getattr(self, '_jSecondary', None) is None
        ):
            raise NotImplementedError(
                'Getting j from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0]
                    )
                )

        return (
                self._jPrimary(solution, srcList) +
                self._jSecondary(solution, srcList)
            )

    def _eDeriv(self, src, du_dm_v, v, adjoint=False):
        """
        Total derivative of e with respect to the inversion model. Returns
        :math:`d\mathbf{e}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{e}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`)
        for the adjoint

        :param SimPEG.EM.FDEM.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, '_eDeriv_u', None) is None or
            getattr(self, '_eDeriv_m', None) is None
        ):
            raise NotImplementedError(
                'Getting eDerivs from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (
                self._eDeriv_u(src, v, adjoint),
                self._eDeriv_m(src, v, adjoint)
            )
        return (
            np.array(
                self._eDeriv_u(src, du_dm_v, adjoint) +
                self._eDeriv_m(src, v, adjoint), dtype=complex
            )
        )

    def _bDeriv(self, src, du_dm_v, v, adjoint=False):
        """
        Total derivative of b with respect to the inversion model. Returns
        :math:`d\mathbf{b}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{b}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for
        the adjoint

        :param SimPEG.EM.FDEM.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, '_bDeriv_u', None) is None or
            getattr(self, '_bDeriv_m', None) is None
        ):
            raise NotImplementedError(
                'Getting bDerivs from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (
                self._bDeriv_u(src, v, adjoint),
                self._bDeriv_m(src, v, adjoint)
            )
        return (
            np.array(
                self._bDeriv_u(src, du_dm_v, adjoint) +
                self._bDeriv_m(src, v, adjoint),
                dtype=complex
            )
        )

    def _bSecondaryDeriv(self, src, du_dm_v, v, adjoint=False):
        """
        Total derivative of b with respect to the inversion model. Returns
        :math:`d\mathbf{b}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{b}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for
        the adjoint

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: sorce
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        # TODO: modify when primary field is dependent on m

        return self._bDeriv(src, du_dm_v, v, adjoint=adjoint)

    def _hDeriv(self, src, du_dm_v, v, adjoint=False):
        """
        Total derivative of h with respect to the inversion model. Returns
        :math:`d\mathbf{h}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{h}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`)
        for the adjoint

        :param SimPEG.EM.FDEM.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, '_hDeriv_u', None) is None or
            getattr(self, '_hDeriv_m', None) is None
        ):
            raise NotImplementedError(
                'Getting hDerivs from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (
                self._hDeriv_u(src, v, adjoint),
                self._hDeriv_m(src, v, adjoint)
            )
        return (
            np.array(
                self._hDeriv_u(src, du_dm_v, adjoint) +
                self._hDeriv_m(src, v, adjoint), dtype=complex
            )
        )

    def _jDeriv(self, src, du_dm_v, v, adjoint=False):
        """
        Total derivative of j with respect to the inversion model. Returns
        :math:`d\mathbf{j}/d\mathbf{m}` for forward and
        (:math:`d\mathbf{j}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for
        the adjoint

        :param SimPEG.EM.FDEM.Src.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: derivative of the solution vector with
            respect to the model times a vector (is None for adjoint)
        :param numpy.ndarray v: vector to take sensitivity product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: derivative times a vector (or tuple for adjoint)
        """
        if (
            getattr(self, '_jDeriv_u', None) is None or
            getattr(self, '_jDeriv_m', None) is None
        ):
            raise NotImplementedError(
                'Getting jDerivs from {0!s} is not implemented'.format(
                    self.knownFields.keys()[0]
                )
            )

        if adjoint:
            return (
                self._jDeriv_u(src, v, adjoint),
                self._jDeriv_m(src, v, adjoint)
            )
        return (
            np.array(
                self._jDeriv_u(src, du_dm_v, adjoint) +
                self._jDeriv_m(src, v, adjoint), dtype=complex
            )
        )


class Fields3D_e(FieldsFDEM):
    """
    Fields object for Problem3D_e.

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    :param SimPEG.EM.FDEM.SurveyFDEM.Survey survey: survey
    """

    knownFields = {'eSolution': 'E'}
    aliasFields = {
        'e': ['eSolution', 'E', '_e'],
        'ePrimary': ['eSolution', 'E', '_ePrimary'],
        'eSecondary': ['eSolution', 'E', '_eSecondary'],
        'b': ['eSolution', 'F', '_b'],
        'bPrimary': ['eSolution', 'F', '_bPrimary'],
        'bSecondary': ['eSolution', 'F', '_bSecondary'],
        'j': ['eSolution', 'CCV', '_j'],
        'h': ['eSolution', 'CCV', '_h'],
    }

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._nC = self.survey.prob.mesh.nC
        self._MeSigma = self.survey.prob.MeSigma
        self._MeSigmaDeriv = self.survey.prob.MeSigmaDeriv
        self._MfMui = self.survey.prob.MfMui
        self._MfMuiDeriv = self.survey.prob.MfMuiDeriv

    def _GLoc(self, fieldType):
        if fieldType in ['e', 'eSecondary', 'ePrimary']:
            return 'E'
        elif fieldType in ['b', 'bSecondary', 'bPrimary']:
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

        ePrimary = np.zeros([self.prob.mesh.nE, len(srcList)], dtype=complex)
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.prob)
            ePrimary[:, i] = ePrimary[:, i] + ep
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

    def _eDeriv_u(self, src, v, adjoint=False):
        """
        Partial derivative of the total electric field with respect to the
        thing we solved for.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """

        return Identity()*v

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total electric field with respect to the
        inversion model. Here, we assume that the primary does not depend on
        the model. Note that this also includes derivative contributions from
        the sources.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the electric field derivative with respect to the
            inversion model with a vector
        """

        return src.ePrimaryDeriv(self.prob, v, adjoint)

    def _bPrimary(self, eSolution, srcList):
        """
        Primary magnetic flux density from source

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic flux density as defined by the sources
        """

        bPrimary = np.zeros(
                [self._edgeCurl.shape[0], eSolution.shape[1]], dtype=complex
        )

        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.prob)
            bPrimary[:, i] = bPrimary[:, i] + bp
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
            b[:, i] *= - 1./(1j*omega(src.freq))  # freq depends on the source
            s_m = src.s_m(self.prob)
            b[:, i] = b[:, i] + 1./(1j*omega(src.freq)) * s_m
        return b

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the thing we
        solved for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """

        C = self._edgeCurl
        if adjoint:
            return - 1./(1j*omega(src.freq)) * (C.T * du_dm_v)
        return - 1./(1j*omega(src.freq)) * (C * du_dm_v)

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the inversion
        model.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic flux density derivative with respect
            to the inversion model with a vector
        """

        return self._bDeriv_src(src, v, adjoint=adjoint)

    def _bDeriv_src(self, src, v, adjoint=False):
        s_mDeriv = src.s_mDeriv(self.prob, v, adjoint)
        return (
            1./(1j * omega(src.freq)) * s_mDeriv +
            src.bPrimaryDeriv(self.prob, v, adjoint)
        )

    def _j(self,  eSolution, srcList):
        """
        Current density from eSolution

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: current density
        """
        aveE2CCV = self._aveE2CCV
        # number of components (instead of checking if cyl or not)
        n = int(aveE2CCV.shape[0] / self._nC)
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        return VI * (aveE2CCV * (self._MeSigma * self._e(eSolution, srcList)))

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the current density with respect to the thing we solved
        for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """
        # number of components (instead of checking if cyl or not)
        n = int(self._aveE2CCV.shape[0] / self._nC)
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        if adjoint:
            return (
                self._eDeriv_u(
                    src, self._MeSigma.T * (
                        self._aveE2CCV.T * (VI.T * du_dm_v)
                        ), adjoint=adjoint
                )
            )
        return (
            VI * (
                self._aveE2CCV * (
                    self._MeSigma * (
                        self._eDeriv_u(src, du_dm_v, adjoint=adjoint)
                    )
                )
            )
        )

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the current density with respect to the inversion model.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the current density derivative with respect to the
            inversion model with a vector
        """
        e = self[src, 'e']
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        if adjoint:
            return (
                self._MeSigmaDeriv(e).T * (self._aveE2CCV.T * (VI.T * v)) +
                self._eDeriv_m(
                    src, self._aveE2CCV.T * (VI.T * v), adjoint=adjoint
                )
            ) + src.jPrimaryDeriv(self.prob, v, adjoint)
        return (
            VI * (
                self._aveE2CCV *
                (
                    self._eDeriv_m(src, v, adjoint=adjoint) +
                    self._MeSigmaDeriv(e) * v
                )
            )
        ) + src.jPrimaryDeriv(self.prob, v, adjoint)

    def _h(self, eSolution, srcList):
        """
        Magnetic field from eSolution

        :param numpy.ndarray eSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: magnetic field
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # Number of Components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        return (
            VI * (self._aveF2CCV * (self._MfMui * self._b(eSolution, srcList)))
        )

    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the thing we solved
        for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # Number of Components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        if adjoint:
            v = self._MfMui.T * (self._aveF2CCV.T * (VI.T * du_dm_v))
            return self._bDeriv_u(src, v, adjoint=adjoint)
        return (
            VI * (
                self._aveF2CCV * (
                    self._MfMui *
                    self._bDeriv_u(src, du_dm_v, adjoint=adjoint)
                )
            )
        )

    def _hDeriv_mui(self, src, v, adjoint=False):
        n = int(self._aveF2CCV.shape[0] / self._nC)  # Number of Components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        if adjoint is True:
            return (
                self._MfMuiDeriv(self[src, 'b']).T * (
                    self._aveF2CCV.T * (VI.T * v)
                )
            )

        return (
            VI * (self._aveF2CCV * (self._MfMuiDeriv(self[src, 'b']) * v))
        )

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the inversion model.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic field derivative with respect to the
            inversion model with a vector
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # Number of Components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        if adjoint:
            return (
                self._bDeriv_m(src, self._MfMui.T * (self._aveF2CCV.T * (VI.T * v)), adjoint=adjoint) +
                self._hDeriv_mui(src, v, adjoint=adjoint)
            )
        return (
            VI * (
                self._aveF2CCV * (
                    self._MfMui * self._bDeriv_m(src, v, adjoint=adjoint)
                )
            )
        ) + self._hDeriv_mui(src, v, adjoint=adjoint)


class Fields3D_b(FieldsFDEM):
    """
    Fields object for Problem3D_b.

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    :param SimPEG.EM.FDEM.SurveyFDEM.Survey survey: survey
    """

    knownFields = {'bSolution': 'F'}
    aliasFields = {
        'b': ['bSolution', 'F', '_b'],
        'bPrimary': ['bSolution', 'F', '_bPrimary'],
        'bSecondary': ['bSolution', 'F', '_bSecondary'],
        'e': ['bSolution', 'E', '_e'],
        'ePrimary': ['bSolution', 'E', '_ePrimary'],
        'eSecondary': ['bSolution', 'E', '_eSecondary'],
        'j': ['bSolution', 'CCV', '_j'],
        'h': ['bSolution', 'CCV', '_h'],
    }

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeSigma = self.survey.prob.MeSigma
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MfMui = self.survey.prob.MfMui
        self._MfMuiDeriv = self.survey.prob.MfMuiDeriv
        self._MeSigmaDeriv = self.survey.prob.MeSigmaDeriv
        self._MeSigmaIDeriv = self.survey.prob.MeSigmaIDeriv
        self._Me = self.survey.prob.Me
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._sigma = self.survey.prob.sigma
        self._mui = self.survey.prob.mui
        self._nC = self.survey.prob.mesh.nC

    def _GLoc(self, fieldType):
        if fieldType in ['e', 'eSecondary', 'ePrimary']:
            return 'E'
        elif fieldType in ['b', 'bSecondary', 'bPrimary']:
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

        bPrimary = np.zeros([self.prob.mesh.nF, len(srcList)], dtype=complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.prob)
            bPrimary[:, i] = bPrimary[:, i] + bp
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
        Partial derivative of the total magnetic flux density with respect to
        the thing we solved for.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """

        return Identity()*du_dm_v

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic flux density with respect to
        the inversion model. Here, we assume that the primary does not depend
        on the model. Note that this also includes derivative contributions
        from the sources.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the magnetic flux density derivative with respect
            to the inversion model with a vector
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

        ePrimary = np.zeros(
            [self._edgeCurl.shape[1], bSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.prob)
            ePrimary[:, i] = ePrimary[:, i] + ep
        return ePrimary

    def _eSecondary(self, bSolution, srcList):
        """
        Secondary electric field from bSolution

        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field
        """

        e = (self._edgeCurl.T * (self._MfMui * bSolution))
        for i, src in enumerate(srcList):
            s_e = src.s_e(self.prob)
            e[:, i] = e[:, i] + - s_e

        return self._MeSigmaI * e

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the electric field with respect to the thing we solved
        for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """

        if not adjoint:
            return (
                self._MeSigmaI * (self._edgeCurl.T * (self._MfMui * du_dm_v))
            )
        return self._MfMui.T * (self._edgeCurl * (self._MeSigmaI.T * du_dm_v))

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the electric field with respect to the inversion model

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the model with a vector
        """

        bSolution = Utils.mkvc(self[src, 'bSolution'])
        s_e = src.s_e(self.prob)

        w = -s_e + self._edgeCurl.T * (self._MfMui * bSolution)

        if adjoint:
            s_eDeriv = src.s_eDeriv(self.prob, self._MeSigmaI.T * v, adjoint)
            return (
                self._MeSigmaIDeriv(w).T * v +
                self._MfMuiDeriv(bSolution).T * (
                    self._edgeCurl * (self._MeSigmaI.T * v)
                    ) -
                s_eDeriv +
                src.ePrimaryDeriv(self.prob, v, adjoint)
            )
        s_eDeriv = src.s_eDeriv(self.prob, v, adjoint)
        return (
            self._MeSigmaIDeriv(w) * v +
            self._MeSigmaI * (
                self._edgeCurl.T * (self._MfMuiDeriv(bSolution) * v)
            ) -
            self._MeSigmaI * s_eDeriv +
            src.ePrimaryDeriv(self.prob, v, adjoint)
        )

    def _j(self, bSolution, srcList):
        """
        Secondary current density from bSolution

        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary current density
        """

        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))


        j = (self._edgeCurl.T * (self._MfMui * bSolution))
        for i, src in enumerate(srcList):
            s_e = src.s_e(self.prob)
            j[:, i] = j[:, i] - s_e

        return VI * (self._aveE2CCV * j)

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the current density with respect to the thing we
        solved for.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        if adjoint:
            return (
                self._MfMui.T * (
                    self._edgeCurl * (
                        self._aveE2CCV.T * (VI.T * du_dm_v)
                    )
                )
            )
        return (
            VI * (
                self._aveE2CCV * (
                    self._edgeCurl.T * (
                        self._MfMui * du_dm_v
                    )
                )
            )
        )
        # forgetting the source term here

    def _jDeriv_mui(self, src, v, adjoint=False):
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        MfMuiDeriv = self._MfMuiDeriv(self[src, 'b'])

        if adjoint:
            return (
                MfMuiDeriv.T * (
                    self._edgeCurl * (self._aveE2CCV.T * (VI.T * v))
                )
            )

        return (
            VI * (self._aveE2CCV * (self._edgeCurl.T * (MfMuiDeriv * v)))
        )

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the current density with respect to the inversion model

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the model with a vector
        """

        return (
            self._jDeriv_mui(src, v, adjoint)
        )

    def _h(self, bSolution, srcList):
        """
        Magnetic field from bSolution

        :param numpy.ndarray bSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: magnetic field
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        return (
            VI * (self._aveF2CCV * (self._MfMui * self._b(bSolution, srcList)))
        )

    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the magnetic field with respect to the thing we
        solved for.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        if adjoint:
            return self._MfMui.T * (self._aveF2CCV.T * (VI.T * du_dm_v))
        return VI * (self._aveF2CCV * (self._MfMui * du_dm_v))

    def _hDeriv_mui(self, src, v, adjoint=False):
        b = self[src, 'b']
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        if adjoint:
            return (
                self._MfMuiDeriv(b).T * (self._aveF2CCV.T * (VI * v))
            )
        return VI * (self._aveF2CCV * (self._MfMuiDeriv(b) * v))

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the inversion model

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the model with a vector
        """
        return (
            src.hPrimaryDeriv(self.prob, v, adjoint) +
            self._hDeriv_mui(src, v, adjoint)
        )


class Fields3D_j(FieldsFDEM):
    """
    Fields object for Problem3D_j.

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    :param SimPEG.EM.FDEM.SurveyFDEM.Survey survey: survey
    """

    knownFields = {'jSolution': 'F'}
    aliasFields = {
        'j': ['jSolution', 'F', '_j'],
        'jPrimary': ['jSolution', 'F', '_jPrimary'],
        'jSecondary': ['jSolution', 'F', '_jSecondary'],
        'h': ['jSolution', 'E', '_h'],
        'hPrimary': ['jSolution', 'E', '_hPrimary'],
        'hSecondary': ['jSolution', 'E', '_hSecondary'],
        'e': ['jSolution', 'CCV', '_e'],
        'b': ['jSolution', 'CCV', '_b'],
      }

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMu = self.survey.prob.MeMu
        self._MeMuI = self.survey.prob.MeMuI
        self._MeMuIDeriv = self.survey.prob.MeMuIDeriv
        self._MfRho = self.survey.prob.MfRho
        self._MfRhoDeriv = self.survey.prob.MfRhoDeriv
        self._rho = self.survey.prob.rho
        self._mu = self.survey.prob.mui
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._nC = self.survey.prob.mesh.nC

    def _GLoc(self, fieldType):
        if fieldType in ['h', 'hSecondary', 'hPrimary']:
            return 'E'
        elif fieldType in ['j', 'jSecondary', 'jPrimary']:
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

        jPrimary = np.zeros_like(jSolution, dtype=complex)
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.prob)
            jPrimary[:, i] = jPrimary[:, i] + jp
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

        return (
            self._jPrimary(jSolution, srcList) +
            self._jSecondary(jSolution, srcList)
        )

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the
        thing we solved for.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """

        return Identity()*du_dm_v

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total current density with respect to the
        inversion model. Here, we assume that the primary does not depend on
        the model. Note that this also includes derivative contributions from
        the sources.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the current density derivative with respect to the
            inversion model with a vector
        """
        # assuming primary does not depend on the model
        return src.jPrimaryDeriv(self.prob, v, adjoint)

    def _hPrimary(self, jSolution, srcList):
        """
        Primary magnetic field from source

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic field as defined by the sources
        """

        hPrimary = np.zeros(
            [self._edgeCurl.shape[1], jSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.prob)
            hPrimary[:, i] = hPrimary[:, i] + hp
        return hPrimary

    def _hSecondary(self, jSolution, srcList):
        """
        Secondary magnetic field from bSolution

        :param numpy.ndarray jSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic field
        """

        h = (self._edgeCurl.T * (self._MfRho * jSolution))
        for i, src in enumerate(srcList):
            h[:, i] *= -1./(1j*omega(src.freq))
            s_m = src.s_m(self.prob)
            h[:, i] = h[:, i] + 1./(1j*omega(src.freq)) * (s_m)
        return self._MeMuI * h

    def _hDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the thing we solved
        for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """

        if adjoint:
            return (
                -1./(1j*omega(src.freq)) * self._MfRho.T *
                (self._edgeCurl * (self._MeMuI.T * du_dm_v))
            )
        return (
            -1./(1j*omega(src.freq)) * self._MeMuI *
            (self._edgeCurl.T * (self._MfRho * du_dm_v))
        )

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic field with respect to the inversion model

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the model with a vector
        """

        jSolution = Utils.mkvc(self[[src], 'jSolution'])
        MeMuI = self._MeMuI
        MeMuIDeriv = self._MeMuIDeriv
        C = self._edgeCurl
        MfRho = self._MfRho
        MfRhoDeriv = self._MfRhoDeriv

        s_m = src.s_m(self.prob)

        def s_mDeriv(v):
            return src.s_mDeriv(self.prob, v, adjoint=adjoint)

        if not adjoint:
            hDeriv_m = 1./(1j*omega(src.freq)) * (
                -1. *  (
                    MeMuI * (C.T * (MfRhoDeriv(jSolution)*v)) +
                    MeMuIDeriv(C.T * (MfRho * jSolution)) *  v
                ) +
                MeMuI * s_mDeriv(v) + MeMuIDeriv(s_m) * v
            )

        elif adjoint:
            hDeriv_m = 1./(1j*omega(src.freq)) * (
                (
                    -1. * (
                        MfRhoDeriv(jSolution).T * (C * (MeMuI.T * v)) +
                        MeMuIDeriv(C.T * (MfRho * jSolution)).T * v
                    )
                ) + s_mDeriv(MeMuI.T*v) + MeMuIDeriv(s_m).T * v
            )

        return hDeriv_m + src.hPrimaryDeriv(self.prob, v, adjoint)

    def _e(self, jSolution, srcList):
        """
        Electric field from jSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: electric field
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        return (
            VI * (self._aveF2CCV * (self._MfRho * self._j(jSolution, srcList)))
        )

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the electric field with respect to the thing we solved
        for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        if adjoint:
            return self._MfRho.T * (self._aveF2CCV.T * (VI.T * du_dm_v))
        return VI * (self._aveF2CCV * (self._MfRho * du_dm_v))

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the electric field with respect to the inversion model

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the model with a vector
        """
        jSolution = Utils.mkvc(self[src, 'jSolution'])
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        if adjoint:
            return (
                self._MfRhoDeriv(jSolution).T * (self._aveF2CCV.T * (VI.T*v)) +
                src.ePrimaryDeriv(self.prob, v, adjoint)
            )
        return (
            VI * (self._aveF2CCV * (self._MfRhoDeriv(jSolution) * v)) +
            src.ePrimaryDeriv(self.prob, v, adjoint)
        )

    def _b(self, jSolution, srcList):
        """
        Secondary magnetic flux density from jSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux density
        """
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        return (
            VI * (self._aveE2CCV * (self._MeMu * self._h(jSolution, srcList)))
        )

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the thing we
        solved for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """
        # if self.prob.mesh._meshType == 'CYL':
        #     self.
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        if adjoint:
            return (
                -1./(1j*omega(src.freq)) *
                self._MfRho.T * (
                    self._edgeCurl * (
                        self._aveE2CCV.T * (VI.T * du_dm_v)
                    )
                )
            )

        return (
            -1./(1j*omega(src.freq)) * VI *
            (self._aveE2CCV * (self._edgeCurl.T * (self._MfRho * du_dm_v)))
        )

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the inversion
        model

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the model with a vector
        """
        jSolution = self[src, 'jSolution']
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        def s_mDeriv(v):
            return src.s_mDeriv(self.prob, v, adjoint=adjoint)

        if adjoint:
            v = self._aveE2CCV.T * (VI.T * v)
            return (
                1./(1j * omega(src.freq)) *
                (
                    s_mDeriv(v) - self._MfRhoDeriv(jSolution).T *
                    (self._edgeCurl * v)
                ) +
                src.bPrimaryDeriv(self.prob, v, adjoint)
            )
        return (
            1./(1j * omega(src.freq)) *
            VI * (
                self._aveE2CCV * (
                    s_mDeriv(v) - self._edgeCurl.T *
                    (self._MfRhoDeriv(jSolution) * v))
            ) +
            src.bPrimaryDeriv(self.prob, v, adjoint)
        )


class Fields3D_h(FieldsFDEM):
    """
    Fields object for Problem3D_h.

    :param discretize.BaseMesh.BaseMesh mesh: mesh
    :param SimPEG.EM.FDEM.SurveyFDEM.Survey survey: survey
    """

    knownFields = {'hSolution': 'E'}
    aliasFields = {
        'h': ['hSolution', 'E', '_h'],
        'hPrimary': ['hSolution', 'E', '_hPrimary'],
        'hSecondary': ['hSolution', 'E', '_hSecondary'],
        'j': ['hSolution', 'F', '_j'],
        'jPrimary': ['hSolution', 'F', '_jPrimary'],
        'jSecondary': ['hSolution', 'F', '_jSecondary'],
        'e': ['hSolution', 'CCV', '_e'],
        'b': ['hSolution', 'CCV', '_b'],
      }

    def startup(self):
        self.prob = self.survey.prob
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMu = self.survey.prob.MeMu
        self._MeMuDeriv = self.survey.prob.MeMuDeriv
        # self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho
        self._MfRhoDeriv = self.survey.prob.MfRhoDeriv
        self._rho = self.survey.prob.rho
        self._mu = self.survey.prob.mui
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._aveE2CCV = self.survey.prob.mesh.aveE2CCV
        self._nC = self.survey.prob.mesh.nC

    def _GLoc(self, fieldType):
        if fieldType in ['h', 'hSecondary', 'hPrimary']:
            return 'E'
        elif fieldType in ['j', 'jSecondary', 'jPrimary']:
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

        hPrimary = np.zeros_like(hSolution, dtype=complex)
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.prob)
            hPrimary[:, i] = hPrimary[:, i] + hp
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
        Partial derivative of the total magnetic field with respect to the
        thing we solved for.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic field with respect
            to the field we solved for with a vector
        """

        return Identity()*du_dm_v

    def _hDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total magnetic field with respect to the
            inversion model. Here, we assume that the primary does not depend
            on the model. Note that this also includes derivative contributions
            from the sources.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.Utils.Zero
        :return: product of the magnetic field derivative with respect to the
            inversion model with a vector
        """

        return src.hPrimaryDeriv(self.prob, v, adjoint)

    def _jPrimary(self, hSolution, srcList):
        """
        Primary current density from source

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: primary current density as defined by the sources
        """

        jPrimary = np.zeros(
            [self._edgeCurl.shape[0], hSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.prob)
            jPrimary[:, i] = jPrimary[:, i] + jp
        return jPrimary

    def _jSecondary(self, hSolution, srcList):
        """
        Secondary current density from hSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: secondary current density
        """

        j = self._edgeCurl*hSolution
        for i, src in enumerate(srcList):
            s_e = src.s_e(self.prob)
            j[:, i] = j[:, i] + -s_e
        return j

    def _jDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the current density with respect to the thing we solved
        for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the current density with respect
            to the field we solved for with a vector
        """

        if not adjoint:
            return self._edgeCurl*du_dm_v
        elif adjoint:
            return self._edgeCurl.T*du_dm_v

    def _jDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the current density with respect to the inversion model.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the current density derivative with respect to the
            inversion model with a vector
        """

        return (
            -src.s_eDeriv(self.prob, v, adjoint) +
            src.jPrimaryDeriv(self.prob, v, adjoint)
        )

    def _e(self, hSolution, srcList):
        """
        Electric field from hSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: electric field
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        return (
            VI *
            (self._aveF2CCV * (self._MfRho * self._j(hSolution, srcList)))
        )

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the electric field with respect to the thing we solved
        for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the electric field with respect
            to the field we solved for with a vector
        """
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        if adjoint:
            return (
                self._edgeCurl.T *
                (
                    self._MfRho.T * (self._aveF2CCV.T * (VI.T * du_dm_v))
                )
            )
        return (
            VI * (self._aveF2CCV * (self._MfRho * self._edgeCurl * du_dm_v))
        )

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the electric field with respect to the inversion model.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the electric field derivative with respect to the
            inversion model with a vector
        """
        hSolution = Utils.mkvc(self[src, 'hSolution'])
        n = int(self._aveF2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        s_e = src.s_e(self.prob)

        if adjoint:
            w = self._aveF2CCV.T * (VI.T * v)
            return (
                self._MfRhoDeriv(self._edgeCurl * hSolution).T * w -
                self._MfRhoDeriv(s_e).T * w +
                src.ePrimaryDeriv(self.prob, v, adjoint)
            )
        return (
            VI *
            (
                self._aveF2CCV *
                (
                    self._MfRhoDeriv(self._edgeCurl * hSolution) * v -
                    self._MfRhoDeriv(s_e) * v

                )
            ) +
            src.ePrimaryDeriv(self.prob, v, adjoint)
        )

    def _b(self, hSolution, srcList):
        """
        Magnetic flux density from hSolution

        :param numpy.ndarray hSolution: field we solved for
        :param list srcList: list of sources
        :rtype: numpy.ndarray
        :return: magnetic flux density
        """
        h = self._h(hSolution, srcList)
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))

        return VI * (self._aveE2CCV * (self._MeMu * h))

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the thing we
        solved for

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with
            respect to the field we solved for with a vector
        """
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        if adjoint:
            return self._MeMu.T * (self._aveE2CCV.T * (VI.T * du_dm_v))
        return VI * (self._aveE2CCV * (self._MeMu * du_dm_v))

    def _bDeriv_mu(self, src, v, adjoint=False):
        h = self[src, 'h']
        n = int(self._aveE2CCV.shape[0] / self._nC)  # number of components
        VI = sdiag(np.kron(np.ones(n), 1./self.prob.mesh.vol))
        MeMuDeriv = self._MeMuDeriv(h)

        if adjoint:
            return MeMuDeriv.T * (self._aveE2CCV.T * (VI.T * v))
        return VI * (self._aveE2CCV * (MeMuDeriv * v))
        # return VI * (self._aveE2CCV * (self._MeMu * h))

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the inversion
        model.

        :param SimPEG.EM.FDEM.SrcFDEM.BaseFDEMSrc src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic flux density derivative with respect
            to the inversion model with a vector
        """
        return (
            src.bPrimaryDeriv(self.prob, v, adjoint) +
            self._bDeriv_mu(src, v, adjoint)
        )


