import numpy as np
import scipy.sparse as sp
from ...utils.code_utils import deprecate_class

from ...fields import Fields
from ..frequency_domain.fields import FieldsFDEM
from ...utils import spzeros, Identity, Zero
from ..utils import omega


# ##############
# #   Fields   #
# ##############
# class BaseNSEMFields(Fields):
#     """Field Storage for a NSEM method."""
#     knownFields = {}
#     dtype = complex


###########
# 1D Fields
###########
class Fields1DPrimarySecondary(FieldsFDEM):
    """
    Fields storage for the 1D NSEM solution.

    Solving for e fields, using primary/secondary formulation
    """

    knownFields = {"e_1dSolution": "F"}
    aliasFields = {
        "e_1d": ["e_1dSolution", "F", "_e"],
        "e_1dPrimary": ["e_1dSolution", "F", "_ePrimary"],
        "e_1dSecondary": ["e_1dSolution", "F", "_eSecondary"],
        "b_1d": ["e_1dSolution", "E", "_b"],
        "b_1dPrimary": ["e_1dSolution", "E", "_bPrimary"],
        "b_1dSecondary": ["e_1dSolution", "E", "_bSecondary"],
    }

    # def __init__(self, mesh, survey, **kwargs):
    #     super(Fields1DPrimarySecondary, self).__init__(mesh, survey, **kwargs)

    def _ePrimary(self, eSolution, source_list):
        """
        Primary electric field from source

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """
        ePrimary = np.zeros_like(eSolution)
        for i, src in enumerate(source_list):
            ep = src.ePrimary(self.simulation)
            ePrimary[:, i] = ePrimary[:, i] + ep[:, -1]
        return ePrimary

    def _eSecondary(self, eSolution, source_list):
        """
        Secondary electric field is the thing we solved for

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field
        """
        return eSolution

    # Overwriting a base FDEM method, could use it.
    # def _e(self, eSolution, source_list):
    #     """
    #     Total electric field is sum of primary and secondary

    #     :param numpy.ndarray solution: field we solved for
    #     :param list source_list: list of sources
    #     :rtype: numpy.ndarray
    #     :return: total electric field
    #     """
    #     return self._ePrimary(eSolution, source_list) + self._eSecondary(eSolution, source_list)

    # def _eDeriv(self, src, du_dm_v, v, adjoint=False):
    #     """
    #     Total derivative of e with respect to the inversion model. Returns :math:`d\mathbf{e}/d\mathbf{m}` for forward and (:math:`d\mathbf{e}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for the adjoint

    #     :param Src src: source
    #     :param numpy.ndarray du_dm_v: derivative of the solution vector with respect to the model times a vector (is None for adjoint)
    #     :param numpy.ndarray v: vector to take sensitivity product with
    #     :param bool adjoint: adjoint?
    #     :rtype: numpy.ndarray
    #     :return: derivative times a vector (or tuple for adjoint)
    #     """
    #     if getattr(self, '_eDeriv_u', None) is None or getattr(self, '_eDeriv_m', None) is None:
    #         raise NotImplementedError ('Getting eDerivs from %s is not implemented' %self.knownFields.keys()[0])

    #     # if adjoint:
    #     #     return self._eDeriv_u(src, v, adjoint), self._eDeriv_m(src, v, adjoint)
    #     return np.array(self._eDeriv_u(src, du_dm_v, adjoint) + self._eDeriv_m(src, v, adjoint), dtype = complex)

    def _eDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Partial derivative of the total electric field with respect to the solution.

        :param SimPEG.EM.NSEM.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
            Size (nE,) when adjoint=True, (nU,) when adjoint=False
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: The calculated derivative, size (nU,) when adjoint=True
            (nE,) when adjoint=False"""

        return Identity() * du_dm_v

    def _eDeriv_m(self, src, v, adjoint=False):
        """
        Partial derivative of the total electric field with respect to the inversion model. Here, we assume that the primary does not depend on the model. Note that this also includes derivative contributions from the sources.

        :param SimPEG.electromagnetics.frequency_domain.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: SimPEG.utils.Zero
        :return: product of the electric field derivative with respect to the inversion model with a vector
        """

        # assuming primary does not depend on the model
        return Zero()

    def _bPrimary(self, eSolution, source_list):
        bPrimary = np.zeros(
            [self.simulation.mesh.nE, eSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(source_list):
            bp = src.bPrimary(self.simulation)
            bPrimary[:, i] = bPrimary[:, i] + bp[:, -1]
        return bPrimary

    def _bSecondary(self, eSolution, source_list):
        """
        Primary magnetic flux density from source

        :param numpy.ndarray eSolution: field we solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic flux density as defined by the sources
        """
        C = self.mesh.nodalGrad
        b = C * eSolution
        for i, src in enumerate(source_list):
            b[:, i] *= -1.0 / (1j * omega(src.freq))
        return b

    # def _b(self, eSolution, source_list):
    #     """
    #     Total magnetic field is sum of primary and secondary

    #     :param numpy.ndarray solution: field we solved for
    #     :param list source_list: list of sources
    #     :rtype: numpy.ndarray
    #     :return: total magnetic field
    #     """
    #     return self._bPrimary(eSolution, source_list) + self._bSecondary(eSolution, source_list)

    # def _bDeriv(self, src, du_dm_v, v, adjoint=False):
    #     """
    #     Total derivative of b with respect to the inversion model. Returns :math:`d\mathbf{b}/d\mathbf{m}` for forward and (:math:`d\mathbf{b}/d\mathbf{u}`, :math:`d\mathb{u}/d\mathbf{m}`) for the adjoint

    #     :param Src src: source
    #     :param numpy.ndarray du_dm_v: derivative of the solution vector with respect to the model times a vector (is None for adjoint)
    #     :param numpy.ndarray v: vector to take sensitivity product with
    #     :param bool adjoint: adjoint?
    #     :rtype: numpy.ndarray
    #     :return: derivative times a vector (or tuple for adjoint)
    #     """
    #     if getattr(self, '_bDeriv_u', None) is None or getattr(self, '_bDeriv_m', None) is None:
    #         raise NotImplementedError ('Getting bDerivs from %s is not implemented' % self.knownFields.keys()[0])

    #     # if adjoint:
    #     #     return self._bDeriv_u(src, v, adjoint), self._bDeriv_m(src, v, adjoint)
    #     return np.array(self._bDeriv_u(src, du_dm_v, adjoint) + self._bDeriv_m(src, v, adjoint), dtype=complex)

    def _bDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the solution

        :param SimPEG.electromagnetics.frequency_domain.Src src: source
        :param numpy.ndarray du_dm_v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the derivative of the magnetic flux density with respect to the field we solved for with a vector
        """
        # bPrimary: no model depenency
        C = self.mesh.nodalGrad
        if adjoint:
            bSecondaryDeriv_u = -1.0 / (1j * omega(src.freq)) * (C.T * du_dm_v)
        else:
            bSecondaryDeriv_u = -1.0 / (1j * omega(src.freq)) * (C * du_dm_v)
        return bSecondaryDeriv_u

    def _bDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of the magnetic flux density with respect to the inversion model.

        :param SimPEG.electromagnetics.frequency_domain.Src src: source
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of the magnetic flux density derivative with respect to the inversion model with a vector
        """
        # Neither bPrimary nor bSeconary have model dependency => return Zero
        return Zero()


###########
# 2D Fields
###########


###########
# 3D Fields
###########
class Fields3DPrimarySecondary(Fields):
    """
    Fields storage for the 3D NSEM solution. Labels polarizations by px and py.

        :param SimPEG object mesh: The solution mesh
        :param SimPEG object survey: A survey object
    """

    # Define the known the alias fields
    # Assume that the solution of e on the E.
    # NOTE: Need to make this more general, to allow for other formats.
    dtype = complex
    knownFields = {"e_pxSolution": "E", "e_pySolution": "E"}
    aliasFields = {
        "e_px": ["e_pxSolution", "E", "_e_px"],
        "e_pxPrimary": ["e_pxSolution", "E", "_e_pxPrimary"],
        "e_pxSecondary": ["e_pxSolution", "E", "_e_pxSecondary"],
        "e_py": ["e_pySolution", "E", "_e_py"],
        "e_pyPrimary": ["e_pySolution", "E", "_e_pyPrimary"],
        "e_pySecondary": ["e_pySolution", "E", "_e_pySecondary"],
        "b_px": ["e_pxSolution", "F", "_b_px"],
        "b_pxPrimary": ["e_pxSolution", "F", "_b_pxPrimary"],
        "b_pxSecondary": ["e_pxSolution", "F", "_b_pxSecondary"],
        "b_py": ["e_pySolution", "F", "_b_py"],
        "b_pyPrimary": ["e_pySolution", "F", "_b_pyPrimary"],
        "b_pySecondary": ["e_pySolution", "F", "_b_pySecondary"],
    }

    # def __init__(self, mesh, survey, **kwargs):
    #     BaseNSEMFields.__init__(self, mesh, survey, **kwargs)

    def _e_pxPrimary(self, e_pxSolution, source_list):
        """
        px polarization of primary electric field from source

        :param numpy.ndarray e_pxSolution: px polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """
        e_pxPrimary = np.zeros_like(e_pxSolution)
        for i, src in enumerate(source_list):
            ep = src.ePrimary(self.simulation)
            if ep is not None:
                e_pxPrimary[:, i] = ep[:, 0]
        return e_pxPrimary

    def _e_pyPrimary(self, e_pySolution, source_list):
        """
        py polarization of primary electric field from source

        :param numpy.ndarray e_pySolution: py polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary electric field as defined by the sources
        """

        e_pyPrimary = np.zeros_like(e_pySolution)
        for i, src in enumerate(source_list):
            ep = src.ePrimary(self.simulation)
            if ep is not None:
                e_pyPrimary[:, i] = ep[:, 1]
        return e_pyPrimary

    def _e_pxSecondary(self, e_pxSolution, source_list):
        """
        px polarization of secondary electric field from source

        :param numpy.ndarray e_pxSolution: px polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field as defined by the sources
        """

        return e_pxSolution

    def _e_pySecondary(self, e_pySolution, source_list):
        """
        py polarization of secondary electric field from source

        :param numpy.ndarray e_pySolution: py polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary electric field as defined by the sources
        """
        return e_pySolution

    def _e_px(self, e_pxSolution, source_list):
        """
        px polarization of electric field from source

        :param numpy.ndarray e_pxSolution: px polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: electric field as defined by the sources
        """
        return self._e_pxPrimary(e_pxSolution, source_list) + self._e_pxSecondary(
            e_pxSolution, source_list
        )

    def _e_py(self, e_pySolution, source_list):
        """
        py polarization of electric field from source

        :param numpy.ndarray e_pySolution: py polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: electric field as defined by the sources
        """
        return self._e_pyPrimary(e_pySolution, source_list) + self._e_pySecondary(
            e_pySolution, source_list
        )

    def _b_pxPrimary(self, e_pxSolution, source_list):
        """
        px polarization of primary magnetic flux from source

        :param numpy.ndarray e_pxSolution: px polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic flux as defined by the sources
        """
        b_pxPrimary = np.zeros(
            [self.simulation.mesh.nF, e_pxSolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(source_list):
            bp = src.bPrimary(self.simulation)
            if bp is not None:
                b_pxPrimary[:, i] += bp[:, 0]
        return b_pxPrimary

    def _b_pyPrimary(self, e_pySolution, source_list):
        """
        py polarization of primary magnetic flux from source

        :param numpy.ndarray e_pySolution: py polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: primary magnetic flux as defined by the sources
        """
        b_pyPrimary = np.zeros(
            [self.simulation.mesh.nF, e_pySolution.shape[1]], dtype=complex
        )
        for i, src in enumerate(source_list):
            bp = src.bPrimary(self.simulation)
            if bp is not None:
                b_pyPrimary[:, i] += bp[:, 1]
        return b_pyPrimary

    def _b_pxSecondary(self, e_pxSolution, source_list):
        """
        px polarization of secondary magnetic flux from source

        :param numpy.ndarray e_pxSolution: px polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux as defined by the sources
        """
        C = self.mesh.edgeCurl
        b = C * e_pxSolution
        for i, src in enumerate(source_list):
            b[:, i] *= -1.0 / (1j * omega(src.freq))
        return b

    def _b_pySecondary(self, e_pySolution, source_list):
        """
        py polarization of secondary magnetic flux from source

        :param numpy.ndarray e_pySolution: py polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: secondary magnetic flux as defined by the sources
        """
        C = self.mesh.edgeCurl
        b = C * e_pySolution
        for i, src in enumerate(source_list):
            b[:, i] *= -1.0 / (1j * omega(src.freq))
        return b

    def _b_px(self, eSolution, source_list):
        """
        py polarization of magnetic flux from source

        :param numpy.ndarray e_pySolution: py polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: magnetic flux as defined by the sources
        """
        return self._b_pxPrimary(eSolution, source_list) + self._b_pxSecondary(
            eSolution, source_list
        )

    def _b_py(self, eSolution, source_list):
        """
        py polarization of magnetic flux from source

        :param numpy.ndarray e_pySolution: py polarization that was solved for
        :param list source_list: list of sources
        :rtype: numpy.ndarray
        :return: magnetic flux as defined by the sources
        """
        return self._b_pyPrimary(eSolution, source_list) + self._b_pySecondary(
            eSolution, source_list
        )

    # Derivatives
    # NOTE: For e_p?Deriv_u,
    # v has to be u(2*nE) long for the not adjoint and nE long for adjoint.
    # Returns nE long for not adjoint and 2*nE long for adjoint
    def _e_pxDeriv(self, src, du_dm_v, v, adjoint=False):
        """ Derivative of e_px with respect to the solution (u) and model (m) """
        # e_px does not depend on the model
        return np.array(
            self._e_pxDeriv_u(src, du_dm_v, adjoint)
            + self._e_pxDeriv_m(src, v, adjoint),
            complex,
        )

    def _e_pyDeriv(self, src, du_dm_v, v, adjoint=False):
        """ Derivative of e_py with respect to the solution (u) and model (m) """
        return np.array(
            self._e_pyDeriv_u(src, du_dm_v, adjoint)
            + self._e_pyDeriv_m(src, v, adjoint),
            complex,
        )

    def _e_pxDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of e_px wrt u

        :param SimPEG.NSEM.src src: The source of the problem
        :param numpy.ndarray du_dm_v: vector to take product with Size (nE,) when adjoint=True, (nU,) when adjoint=False
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: The calculated derivative, size (nU,) when adjoint=True (nE,) when adjoint=False
        """
        # e_pxPrimary doesn't depend on u, only e_pxSecondary
        if adjoint:
            # adjoint: returns a 2*nE long vector with zero's for py
            return np.concatenate((du_dm_v, np.zeros_like(du_dm_v)))
        # Not adjoint: return only the px part of the vector
        return du_dm_v[: int(len(du_dm_v) / 2)]

    def _e_pyDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of e_py wrt u

        :param SimPEG.NSEM.src src: The source of the problem
        :param numpy.ndarray du_dm_v: vector to take product with Size (nE,) when adjoint=True, (nU,) when adjoint=False
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: The calculated derivative, size (nU,) when adjoint=True (nE,) when adjoint=False

        """

        if adjoint:
            # adjoint: returns a 2*nE long vector with zero's for px
            return np.concatenate((np.zeros_like(du_dm_v), du_dm_v))
        # Not adjoint: return only the px part of the vector
        return du_dm_v[int(len(du_dm_v) / 2) : :]

    def _e_pxDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of e_px wrt m

        :param SimPEG.NSEM.src src: The source of the problem
        :param numpy.ndarray v: vector to take product with Size (nE,) when adjoint=True, (nU,) when adjoint=False
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: The calculated derivative, size (nU,) when adjoint=True (nE,) when adjoint=False


        """
        # e_px does not depend on the model
        return Zero()

    def _e_pyDeriv_m(self, src, v, adjoint=False):
        """
        Derivative of e_py wrt m

        :param SimPEG.NSEM.src src: The source of the problem
        :param numpy.ndarray v: vector to take product with Size (nE,) when adjoint=True, (nU,) when adjoint=False
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: The calculated derivative, size (nU,) when adjoint=True (nE,) when adjoint=False


        """
        # e_py does not depend on the model
        return Zero()

    # Magnetic flux
    def _b_pxDeriv(self, src, du_dm_v, v, adjoint=False):
        """ Derivative of b_px with respect to the solution (u) and model (m) """
        # b_px does not depend on the model

        return np.array(
            self._b_pxDeriv_u(src, du_dm_v, adjoint)
            + self._b_pxDeriv_m(src, v, adjoint),
            complex,
        )

    def _b_pyDeriv(self, src, du_dm_v, adjoint=False):
        """ Derivative of b_px with respect to the solution (u) and model (m) """
        # Primary does not depend on u
        return np.array(
            self._b_pyDeriv_u(src, du_dm_v, adjoint)
            + self._b_pyDeriv_m(src, v, adjoint),
            complex,
        )

    def _b_pxDeriv_u(self, src, du_dm_v, adjoint=False):
        """
        Derivative of b_px with wrt u

        :param SimPEG.NSEM.src src: The source of the problem
        :param numpy.ndarray du_dm_v: vector to take product with. Size (nF,) when adjoint=True, (nU,) when adjoint=False
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: The calculated derivative, size (nU,) when adjoint=True. (nF,) when adjoint=False
        """
        # Primary does not depend on u
        C = sp.hstack(
            (self.mesh.edgeCurl, spzeros(self.mesh.nF, self.mesh.nE))
        )  # This works for adjoint = None
        if adjoint:
            return -1.0 / (1j * omega(src.freq)) * (C.T * du_dm_v)
        return -1.0 / (1j * omega(src.freq)) * (C * du_dm_v)

    def _b_pyDeriv_u(self, src, du_dm_v, adjoint=False):
        """ Derivative of b_py with wrt u

        :param SimPEG.NSEM.src src: The source of the problem
        :param numpy.ndarray du_dm_v: vector to take product with. Size (nF,) when adjoint=True, (nU,) when adjoint=False
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: The calculated derivative, size (nU,) when adjoint=True. (nF,) when adjoint=False
        """
        # Primary does not depend on u
        C = sp.hstack(
            (spzeros(self.mesh.nF, self.mesh.nE), self.mesh.edgeCurl)
        )  # This works for adjoint = None
        if adjoint:
            return -1.0 / (1j * omega(src.freq)) * (C.T * du_dm_v)
        return -1.0 / (1j * omega(src.freq)) * (C * du_dm_v)

    def _b_pxDeriv_m(self, src, v, adjoint=False):
        """ Derivative of b_px wrt m """
        # b_px does not depend on the model
        return Zero()

    def _b_pyDeriv_m(self, src, v, adjoint=False):
        """ Derivative of b_py wrt m """
        # b_py does not depend on the model
        return Zero()


############
# Deprecated
############
@deprecate_class(removal_version="0.15.0")
class Fields1D_ePrimSec(Fields1DPrimarySecondary):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields3D_ePrimSec(Fields3DPrimarySecondary):
    pass
