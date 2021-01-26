import numpy as np
import scipy as sp
import properties
from scipy.constants import mu_0

from ...utils import mkvc, sdiag, Zero
from ..base import BaseEMSimulation
from ...data import Data
from ... import props

from .survey import Survey1D

from discretize import TensorMesh


class Simulation1DRecursive(BaseEMSimulation):
    """
    Simulation class for the 1D MT problem using recursive solution.

    This solution is defined with z +ve upward and a :math:`+i\\omega t`
    Fourier convention. First, let:

    .. math::
        \\alpha_i^2 = i\\omega\\mu_i\\sigma_i

    The complex impedance in layer :math:`i` is given by:

    .. math::
        Z_i = \\dfrac{\\alpha_i}{\\sigma_i} \\Bigg [
        \\dfrac{\\sigma_i Z_{i+1} - \\alpha_i tanh(\\alpha_i h_i)}
        {\\alpha_i - \\sigma_i Z_{i+1}tanh(\\alpha_i h_i)} \\Bigg ]

    where the complex impedance in the bottom half-space is given by:

    .. math::
        Z_N = - \\frac{\\alpha_N}{\\sigma_N}


    """

    # Must be 1D survey object
    survey = properties.Instance("a Survey1D survey object", Survey1D, required=True)

    # Finite difference or analytic sensitivities can be computed
    sensitivity_method = properties.StringChoice(
        "Choose 1st or 2nd order computations with sensitivity matrix ('1st order', '2nd order')",
        {
            "1st order": ["1st_order", "1st-order", "1st", "1", "first"],
            "2nd order": ["2nd_order", "2nd-order", "2nd", "2", "second"],
        },
    )

    # Add layer thickness as invertible property
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers starting from the positive end of the mesh"
    )

    # Storing sensitivity
    _Jmatrix = None
    fix_Jmatrix = False
    storeJ = properties.Bool("store the sensitivity", default=False)

    # Frequency for each datum
    _frequency_vector = None

    @property
    def frequency_vector(self):
        """
        A vector containing the corresponding frequency for each datum.
        """

        if getattr(self, "_frequency_vector", None) is None:
            if self._frequency_vector is None:
                fvec = []
                for src in self.survey.source_list:
                    fvec.append(src.frequency)
                self._frequency_vector = np.hstack(fvec)

        return self._frequency_vector

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super(Simulation1DRecursive, self).deleteTheseOnModelUpdate
        if self.fix_Jmatrix:
            return toDelete

        if self._Jmatrix is not None:
            toDelete += ["_Jmatrix"]
        return toDelete

    # Instantiate
    def __init__(self, sensitivity_method="2nd order", **kwargs):

        self.sensitivity_method = sensitivity_method

        BaseEMSimulation.__init__(self, **kwargs)

    def _get_recursive_impedances_1d(self, fvec, thicknesses, sigma_1d):
        """
        For a given layered Earth model, this returns the complex impedances
        at the surface for all frequencies.

        :param numpy.ndarray fvec: vector with frequencies in Hz (nFreq,)
        :param numpy.ndarray thicknesses: layer thicknesses (nLayers-1,)
        :param numpy.ndarray sigma_1d: layer conductivities (nLayers,)
        :return numpy.ndarray Z: complex impedances at surface for all frequencies (nFreq,)
        """

        omega = 2 * np.pi * fvec
        n_layer = len(sigma_1d)

        # Bottom layer quantities
        alpha = np.sqrt(1j * omega * mu_0 * sigma_1d[-1])
        ratio = alpha / sigma_1d[-1]
        Z = -ratio

        # Work from lowest layer to top layer
        for ii in range(n_layer - 2, -1, -1):
            alpha = np.sqrt(1j * omega * mu_0 * sigma_1d[ii])
            ratio = alpha / sigma_1d[ii]
            tanh = np.tanh(alpha * thicknesses[ii])

            top = Z / ratio - tanh
            bot = 1 - Z / ratio * tanh
            Z = ratio * (top / bot)

        return Z

    def _get_sigma_sensitivities(self, fvec, thicknesses, sigma_1d):
        """
        For a given layered Earth model, this returns the Jacobian for the
        complex impedances at the surface with respect to the layer conductivities.

        :param numpy.ndarray fvec: vector with frequencies for each datum in Hz (nD,)
        :param numpy.ndarray thicknesses: layer thicknesses (nLayers-1,)
        :param numpy.ndarray sigma_1d: layer conductivities (nLayers,)
        :return numpy.ndarray J: Jacobian (nD, nLayers)
        """

        omega = 2 * np.pi * fvec
        n_layer = len(sigma_1d)
        J = np.empty((len(fvec), n_layer), dtype=np.complex128)

        # Bottom layer quantities
        alpha = np.sqrt(1j * omega * mu_0 * sigma_1d[-1])
        alpha_ds = 1j * omega * mu_0 / (2 * alpha)

        ratio = alpha / sigma_1d[-1]
        ratio_ds = alpha_ds / sigma_1d[-1] - ratio / sigma_1d[-1]

        Z = -ratio
        dZ_dsigma = -ratio_ds
        J[:, -1] = dZ_dsigma

        # Work from lowest layer to top layer
        for ii in range(n_layer - 2, -1, -1):
            alpha = np.sqrt(1j * omega * mu_0 * sigma_1d[ii])
            alpha_ds = 1j * omega * mu_0 / (2 * alpha)

            ratio = alpha / sigma_1d[ii]
            ratio_ds = alpha_ds / sigma_1d[ii] - ratio / sigma_1d[ii]

            tanh = np.tanh(alpha * thicknesses[ii])
            tanh_ds = thicknesses[ii] * (1 - tanh * tanh) * alpha_ds

            top = Z / ratio - tanh
            top_ds = -tanh_ds - Z / (ratio * ratio) * ratio_ds
            top_dZ = 1 / ratio

            bot = 1 - Z / ratio * tanh
            bot_ds = (-Z / ratio) * tanh_ds + Z * tanh / (ratio ** 2) * ratio_ds
            bot_dZ = -(tanh / ratio)

            Z = ratio * (top / bot)
            Z_dratio = top / bot
            Z_dtop = ratio / bot
            Z_dbot = -ratio * top / (bot * bot)

            dZ_ds = Z_dtop * top_ds + Z_dbot * bot_ds + Z_dratio * ratio_ds
            dZ_dZp1 = Z_dtop * top_dZ + Z_dbot * bot_dZ

            J[:, ii] = dZ_ds
            J[:, ii + 1 :] *= dZ_dZp1[:, None]

        return J

    def _get_thickness_sensitivities(self, fvec, thicknesses, sigma_1d):
        """
        For a given layered Earth model, this returns the Jacobian for the
        complex impedances at the surface with respect to the layer thicknesses.

        :param numpy.ndarray fvec: vector with frequencies for each datum in Hz (nD,)
        :param numpy.ndarray thicknesses: layer thicknesses (nLayers-1,)
        :param numpy.ndarray sigma_1d: layer conductivities (nLayers,)
        :return numpy.ndarray J: Jacobian (nD, nLayers-1)
        """

        # Bottom layer quantities
        omega = 2 * np.pi * fvec
        n_layer = len(sigma_1d)
        J = np.empty((len(fvec), n_layer - 1), dtype=np.complex128)

        alpha = np.sqrt(1j * omega * mu_0 * sigma_1d[-1])
        ratio = alpha / sigma_1d[-1]

        Z = -ratio

        # Work from lowest layer to top layer
        for ii in range(n_layer - 2, -1, -1):
            alpha = np.sqrt(1j * omega * mu_0 * sigma_1d[ii])
            ratio = alpha / sigma_1d[ii]

            tanh = np.tanh(alpha * thicknesses[ii])
            tanh_dh = alpha * (1 - tanh * tanh)

            top = Z / ratio - tanh
            top_dh = -tanh_dh
            top_dZ = 1 / ratio

            bot = 1 - Z / ratio * tanh
            bot_dh = (-Z / ratio) * tanh_dh
            bot_dZ = -(tanh / ratio)

            Z = ratio * (top / bot)
            Z_dtop = ratio / bot
            Z_dbot = -ratio * top / (bot * bot)

            dZ_dh = Z_dtop * top_dh + Z_dbot * bot_dh
            dZ_dZp1 = Z_dtop * top_dZ + Z_dbot * bot_dZ

            J[:, ii] = dZ_dh
            J[:, ii + 1 :] *= dZ_dZp1[:, None]

        return J

    def fields(self, m):
        """
        Computes the data for a given 1D model.

        :param np.array m: inversion model (nP,)
        :return np.array f: data (nD,)
        """

        if m is not None:
            self.model = m

        # Compute complex impedances for each datum
        complex_impedance = self._get_recursive_impedances_1d(
            self.frequency_vector, self.thicknesses, self.sigma
        )

        # For each complex impedance, extract compute datum
        f = []
        for ii, src in enumerate(self.survey.source_list):
            for rx in src.receiver_list:
                if rx.component == "real":
                    f.append(np.real(complex_impedance[ii]))
                elif rx.component == "imag":
                    f.append(np.imag(complex_impedance[ii]))
                elif rx.component == "apparent resistivity":
                    f.append(
                        np.abs(complex_impedance[ii]) ** 2
                        / (2 * np.pi * src.frequency * mu_0)
                    )
                elif rx.component == "phase":
                    f.append(
                        (180.0 / np.pi)
                        * np.arctan(
                            np.imag(complex_impedance[ii])
                            / np.real(complex_impedance[ii])
                        )
                    )

        return np.array(f)

    def dpred(self, m=None, f=None):
        """
        Predict data vector for a given model.

        :param numpy.ndarray m: inversion model (nP,)
        :return numpy.ndarray d: data (nD,)
        """

        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        return f

    def getJ(self, m, f=None, sensitivity_method=None):

        """
        Compute and store the sensitivity matrix.

        :param numpy.ndarray m: inversion model (nP,)
        :param String method: Choose from '1st_order' or '2nd_order'
        :return numpy.ndarray J: Sensitivity matrix (nD, nP)
        """

        if sensitivity_method is None:
            sensitivity_method = self.sensitivity_method

        if self._Jmatrix is not None:
            pass

        # Finite difference computation
        elif sensitivity_method == "1st order":

            # 1st order computation
            self.model = m

            N = self.survey.nD
            M = self.model.size
            Jmatrix = np.zeros((N, M), dtype=float, order="F")

            factor = 0.01
            for ii in range(0, len(m)):
                m1 = m.copy()
                m2 = m.copy()
                dm = np.max([factor * np.abs(m[ii]), 1e-3])
                m1[ii] = m[ii] - 0.5 * dm
                m2[ii] = m[ii] + 0.5 * dm
                d1 = self.dpred(m1)
                d2 = self.dpred(m2)
                Jmatrix[:, ii] = (d2 - d1) / dm

            self._Jmatrix = Jmatrix

        # Analytic computation
        elif sensitivity_method == "2nd order":

            self.model = m

            dMdm = []  # Derivative of properties with respect to model
            Jmatrix = []  # Jacobian

            # Derivatives for conductivity
            if self.sigmaMap != None:
                dMdm.append(self.sigmaDeriv)
                Jmatrix.append(
                    self._get_sigma_sensitivities(
                        self.frequency_vector, self.thicknesses, self.sigma
                    )
                )

            # Derivatives for thicknesses
            if self.thicknessesMap != None:
                dMdm.append(self.thicknessesDeriv)
                Jmatrix.append(
                    self._get_thickness_sensitivities(
                        self.frequency_vector, self.thicknesses, self.sigma
                    )
                )

            # Combine
            if len(dMdm) == 1:
                dMdm = dMdm[0]
                Jmatrix = Jmatrix[0]
            else:
                dMdm = sp.sparse.vstack(dMdm)
                Jmatrix = np.hstack(Jmatrix)
            J = np.empty((self.survey.nD, Jmatrix.shape[1]))

            start = 0
            for ii, source_ii in enumerate(self.survey.source_list):
                for rx in source_ii.receiver_list:
                    if rx.component == "real":
                        Jrows = np.real(Jmatrix[ii, :])
                    elif rx.component == "imag":
                        Jrows = np.imag(Jmatrix[ii, :])
                    else:
                        Z = self._get_recursive_impedances_1d(
                            source_ii.frequency, self.thicknesses, self.sigma
                        )
                        if rx.component == "apparent resistivity":
                            Jrows = (np.pi * source_ii.frequency * mu_0) ** -1 * (
                                np.real(Z) * np.real(Jmatrix[ii, :])
                                + np.imag(Z) * np.imag(Jmatrix[ii, :])
                            )
                        elif rx.component == "phase":
                            C = 180 / np.pi
                            real = np.real(Z)
                            imag = np.imag(Z)
                            bot = np.abs(Z) ** 2
                            d_real_dm = np.real(Jmatrix[ii, :])
                            d_imag_dm = np.imag(Jmatrix[ii, :])
                            Jrows = C * (
                                -imag / bot * d_real_dm + real / bot * d_imag_dm
                            )
                    end = start + rx.nD
                    J[start:end] = Jrows
                    start = end

            self._Jmatrix = J * dMdm

        return self._Jmatrix

    def Jvec(self, m, v, f=None, sensitivity_method=None):
        """
        Sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product
            witH (nP,)
        :param String method: Choose from '1st_order' or '2nd_order'
        :return numpy.ndarray Jv: Jv (nD,)
        """

        if sensitivity_method is None:
            sensitivity_method = self.sensitivity_method

        J = self.getJ(m, f=None, sensitivity_method=sensitivity_method)

        return mkvc(np.dot(J, v))

    def Jtvec(self, m, v, f=None, sensitivity_method=None):
        """
        Transpose of sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product
            with (nD,)
        :param String method: Choose from '1st_order' or '2nd_order'
        :return numpy.ndarray Jtv: Jtv (nP,)
        """

        if sensitivity_method is None:
            sensitivity_method = self.sensitivity_method

        J = self.getJ(m, f=None, sensitivity_method=sensitivity_method)

        return mkvc(np.dot(v, J))
