import numpy as np
import properties
from scipy.constants import mu_0

from ...simulation import BaseSimulation
from ... import props
from ..frequency_domain.survey import Survey


class Simulation1DRecursive(BaseSimulation):
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

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    # Add layer thickness as invertible property
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers starting from the bottom of the mesh"
    )

    # Must be 1D survey object
    survey = properties.Instance("a frequency_domain survey", Survey, required=True)
    fix_Jmatrix = False

    # TODO: These should be moved to geoana
    def _get_recursive_impedances(self, frequencies, thicknesses, sigmas):
        """
        For a given layered Earth model, this returns the complex impedances
        at the surface for all frequencies.

        Parameters
        ----------
        frequencies : (n_freq, ) np.ndarray
            Frequencies in Hz
        thicknesses : (n_layer-1, ) np.ndarray
            Layer thicknesses in meters, starting from the bottom
        sigmas : (n_layer, ) np.ndarray
            Layer conductivities in S/m, starting from the bottom

        Returns
        -------
        Z : (n_freq, ) np.ndarray
            complex impedances at surface
        """
        frequencies = np.asarray(frequencies)
        thicknesses = np.asarray(thicknesses)[::-1]
        sigmas = np.asarray(sigmas)[::-1]
        omega = 2 * np.pi * frequencies
        n_layer = len(sigmas)

        # layer quantities
        alphas = np.sqrt(1j * omega * mu_0 * sigmas[:, None])
        ratios = alphas / sigmas[:, None]
        tanhs = np.tanh(alphas[:-1] * thicknesses[:, None])

        Z = -ratios[-1]
        # Work from lowest layer to top layer
        for ii in range(n_layer - 2, -1, -1):
            top = Z / ratios[ii] - tanhs[ii]
            bot = 1 - Z / ratios[ii] * tanhs[ii]
            Z = ratios[ii] * top / bot
        return Z

    def _get_recursive_impedances_deriv(self, frequencies, thicknesses, sigmas):
        """
        For a given layered Earth model, this returns the complex impedances
        at the surface for all frequencies.

        Parameters
        ----------
        frequencies : (n_freq, ) np.ndarray
            Frequencies in Hz
        thicknesses : (n_layer-1, ) np.ndarray
            Layer thicknesses in meters, starting from the bottom
        sigmas : (n_layer, ) np.ndarray
            Layer conductivities in S/m, starting from the bottom

        Returns
        -------
        Z : (n_freq, ) np.ndarray
            Complex impedance at surface
        Z_dsigma : (n_freq, n_layer) np.ndarray
            Derivative of complex impedances at surface with respect to sigma
        Z_dsigma : (n_freq, n_layer-1) np.ndarray
            Derivative of complex impedances at surface with respect to thicknesses
        """
        frequencies = np.asarray(frequencies)
        thicknesses = np.asarray(thicknesses)[::-1]
        sigmas = np.asarray(sigmas)[::-1]
        omega = 2 * np.pi * frequencies
        n_layer = len(sigmas)

        # Bottom layer quantities
        alphas = np.sqrt(1j * omega * mu_0 * sigmas[:, None])
        ratios = alphas / sigmas[:, None]
        tanhs = np.tanh(alphas[:-1] * thicknesses[:, None])

        tops = np.empty_like(tanhs)
        bots = np.empty_like(tanhs)
        Zs = np.empty_like(ratios)
        Zs[-1] = -ratios[-1]
        # Work from lowest layer to top layer
        for ii in range(n_layer - 2, -1, -1):
            tops[ii] = Zs[ii + 1] / ratios[ii] - tanhs[ii]
            bots[ii] = 1 - Zs[ii + 1] / ratios[ii] * tanhs[ii]
            Zs[ii] = ratios[ii] * (tops[ii] / bots[ii])

        gZ = 1.0
        gratios = np.empty_like(ratios)
        gtanhs = np.empty_like(tanhs)
        for ii in range(n_layer - 1):
            gratios[ii] = (tops[ii] / bots[ii]) * gZ
            gtop = ratios[ii] / bots[ii] * gZ
            gbot = -Zs[ii] / bots[ii] * gZ

            gZ = -tanhs[ii] / ratios[ii] * gbot
            gratios[ii] += Zs[ii + 1] * tanhs[ii] / ratios[ii] ** 2 * gbot
            gtanhs[ii] = -Zs[ii + 1] / ratios[ii] * gbot

            gZ += gtop / ratios[ii]
            gratios[ii] -= Zs[ii + 1] / ratios[ii] ** 2 * gtop
            gtanhs[ii] -= gtop
        gratios[-1] = -gZ
        d_thick = (1 - tanhs ** 2) * alphas[:-1] * gtanhs

        galphas = gratios / sigmas[:, None]
        galphas[:-1] += (1 - tanhs ** 2) * thicknesses[:, None] * gtanhs

        d_sigma = -ratios / sigmas[:, None] * gratios
        d_sigma += (0.5j * omega * mu_0) / alphas * galphas

        # d_mu would be this below when it gets activated:
        # d_mu = (0.5j * omega * sigmas[:, None]) / alphas * galphas
        return Zs[0], d_sigma[::-1].T, d_thick[::-1].T

    def fields(self, m):
        # The layered simulation does not have fields.
        return None

    def dpred(self, m, f=None):
        """
        Computes the data for a given 1D model.

        :param np.array m: inversion model (nP,)
        :return np.array f: data (nD,)
        """
        self.model = m

        # Compute complex impedances for each frequency=
        Z = self._get_recursive_impedances(
            self.survey.frequencies, self.thicknesses, self.sigma
        )

        # For each complex impedance, extract compute datum
        d = []
        for src in self.survey.source_list:
            i_freq = np.searchsorted(self.survey.frequencies, src.frequency)
            for rx in src.receiver_list:
                if rx.component == "real":
                    d.append(np.real(Z[i_freq]))
                elif rx.component == "imag":
                    d.append(np.imag(Z[i_freq]))
                elif rx.component == "apparent_resistivity":
                    d.append(
                        np.abs(Z[i_freq]) ** 2 / (2 * np.pi * src.frequency * mu_0)
                    )
                elif rx.component == "phase":
                    d.append(
                        (180.0 / np.pi)
                        * np.arctan(np.imag(Z[i_freq]) / np.real(Z[i_freq]))
                    )

        return np.array(d)

    def getJ(self, m, f=None):
        """
        Compute and store the sensitivity matrix.

        :param numpy.ndarray m: inversion model (nP,)
        :return numpy.ndarray J: Sensitivity matrix (nD, nP)
        """
        # Analytic computation
        self.model = m
        if getattr(self, "_Jmatrix", None) is not None:
            return self._Jmatrix

        # Derivatives for conductivity
        Z, Z_dsigma, Z_dthick = self._get_recursive_impedances_deriv(
            self.survey.frequencies, self.thicknesses, self.sigma
        )
        Js = []
        if self.sigmaMap is not None:
            Js.append(Z_dsigma)
        if self.thicknessesMap is not None:
            Js.append(Z_dthick)
        Js = np.hstack(Js)

        J = np.empty((self.survey.nD, Js.shape[1]))

        start = 0
        for src in self.survey.source_list:
            i_freq = np.searchsorted(self.survey.frequencies, src.frequency)
            Js_row = Js[i_freq]
            for rx in src.receiver_list:
                if rx.component == "real":
                    Jrows = np.real(Js_row)
                elif rx.component == "imag":
                    Jrows = np.imag(Js_row)
                elif rx.component == "apparent_resistivity":
                    Jrows = (np.pi * src.frequency * mu_0) ** -1 * (
                        np.real(Z[i_freq]) * np.real(Js_row)
                        + np.imag(Z[i_freq]) * np.imag(Js_row)
                    )
                elif rx.component == "phase":
                    C = 180 / np.pi
                    real = np.real(Z[i_freq])
                    imag = np.imag(Z[i_freq])
                    bot = real ** 2 + imag ** 2
                    d_real_dm = np.real(Js_row)
                    d_imag_dm = np.imag(Js_row)
                    Jrows = C * (-imag / bot * d_real_dm + real / bot * d_imag_dm)
                end = start + rx.nD
                J[start:end] = Jrows
                start = end
        self._Jmatrix = {}
        start = 0
        if self.sigmaMap is not None:
            end = start + Z_dsigma.shape[1]
            self._Jmatrix["sigma"] = J[:, start:end]
            start = end
        if self.thicknessesMap is not None:
            end = start + Z_dthick.shape[1]
            self._Jmatrix["thick"] = J[:, start:end]
        return self._Jmatrix

    def getJtJdiag(self, m, W=None):
        if getattr(self, "_gtgdiag", None) is None:
            Js = self.getJ(m)
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal() ** 2

            gtgdiag = 0
            if self.sigmaMap is not None:
                J = Js["sigma"] @ self.sigmaDeriv
                gtgdiag += np.einsum("i,ij,ij->j", W, J, J)
            if self.thicknessesMap is not None:
                J = Js["thick"] @ self.thicknessesDeriv
                gtgdiag += np.einsum("i,ij,ij->j", W, J, J)
            self._gtgdiag = gtgdiag
        return self._gtgdiag

    def Jvec(self, m, v, f=None):
        J = self.getJ(m, f=None)
        Jvec = 0
        if self.sigmaMap is not None:
            Jvec += J["sigma"] @ (self.sigmaDeriv * v)
        if self.thicknessesMap is not None:
            Jvec += J["thick"] @ (self.thicknessesDeriv * v)
        return Jvec

    def Jtvec(self, m, v, f=None):
        J = self.getJ(m, f=None)
        JTvec = 0
        if self.sigmaMap is not None:
            JTvec += self.sigmaDeriv.T @ (J["sigma"].T @ v)
        if self.thicknessesMap is not None:
            JTvec += self.thicknessesDeriv.T @ (J["thick"].T @ v)
        return JTvec

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self.fix_Jmatrix:
            return toDelete
        else:
            toDelete = toDelete + ["_Jmatrix", "_gtgdiag"]
        return toDelete
