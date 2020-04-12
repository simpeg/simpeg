import numpy as np
import scipy as sp
import properties
from scipy.constants import mu_0

from ...utils import mkvc, sdiag, Zero
from ..base import BaseEMSimulation
from ...data import Data
from ... import props

from .survey import Survey1D


class BaseSimulation1D(BaseEMSimulation):

    # Must be 1D survey object
    survey = properties.Instance(
        "a Survey1D survey object", Survey1D, required=True
    )

    sensitivity_method = properties.StringChoice(
        "Choose 1st or 2nd order computations with sensitivity matrix ('approximate', 'analytic')", {
            "approximate": [],
            "analytic": []
        }
    )

    # Instantiate
    def __init__(self, sensitivity_method='approximate', **kwargs):
        
        self.sensitivity_method = sensitivity_method

        BaseEMSimulation.__init__(self, **kwargs)


    # Compute layer admittances for a 1d model
    def _get_admittance(self, f, sigma_1d):

        """
        Layer admittances

        :param np.float f: frequency in Hz
        :pamam np.array sig: layer conductivities in S/m (nLayers,)
        :return a: layer admittances (nLayers,)
        """

        return (1 - 1j)*np.sqrt(sigma_1d/(4*np.pi*f*mu_0))


    def _get_propagator_matricies_1d(self, src, thicknesses, sigma_1d):
        """
        For a given source and layered Earth model, this returns the list of
        propagator matricies.

        :param SimPEG.electromagnetics.sources.AnalyticPlanewave1D src: Analytic 1D planewave source
        :param np.array thicknesses: layer thicknesses (nLayers-1,)
        :param np.array sigma_1d: layer conductivities (nLayers,)
        :return list Q: list containing matrix for each layer [nLayers,]
        """
        
        # Get frequency for planewave source
        f = src.frequency

        # Admittance for all layers
        a = self._get_admittance(f, sigma_1d)

        # List of empty places for each 2x2 array Qj
        Q = len(thicknesses)*[None]

        # Compute exponent terms
        e_neg = np.exp(-(1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*thicknesses)
        e_pos = np.exp( (1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*thicknesses)

        # Create Q matrix for each layer
        for jj in range(0, len(thicknesses)):

            Q[jj] = (0.5/a[jj])*np.array([
                [a[jj]*(e_neg[jj] + e_pos[jj]), (e_neg[jj] - e_pos[jj])],
                [a[jj]**2*(e_neg[jj] - e_pos[jj]), a[jj]*(e_neg[jj] + e_pos[jj])]
                ], dtype=complex)

        # Compute 2x2 matrix for bottom layer and append
        Q.append(np.array([[1, 1], [a[-1], -a[-1]]], dtype=complex))

        return Q


    def _get_sigma_derivative_matricies_1d(self, src, thicknesses, sigma_1d):
        """
        For a given source (frequency) and layered Earth, return the list containing
        the derivative of each layer's propagator matrix with respect to conductivity.
        
        :param SimPEG.electromagnetics.sources.AnalyticPlanewave1D src: Analytic 1D planewave source
        :param np.array thicknesses: layer thicknesses (nLayers-1,)
        :param np.array sigma_1d: layer conductivities (nLayers,)
        :return list dQdig: list containing matrix for each layer [nLayers,]
        """
        
        # Get frequency for planewave source
        f = src.frequency

        # Admittance for all layers
        a = self._get_admittance(f, sigma_1d)

        # List of empty places for each 2x2 array dQdsig_j
        dQdsig = len(thicknesses)*[None]

        # Compute exponent terms
        e_neg = np.exp(-(1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*thicknesses)
        e_pos = np.exp( (1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*thicknesses)

        # product of i, wavenumber and layer thicknesses
        ikh = (1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*thicknesses  

        # Create dQdm matrix for each layer
        for jj in range(0, len(thicknesses)):

            dQdsig[jj] = np.array([
                [
                ikh[jj]*(-e_neg[jj] + e_pos[jj])/(4*sigma_1d[jj]),
                (-1/(4*sigma_1d[jj]*a[jj]))*(e_neg[jj] - e_pos[jj]) - (thicknesses[jj]/(4*a[jj]**2))*(e_neg[jj] + e_pos[jj])
                ],[
                (a[jj]/(4*sigma_1d[jj]))*(e_neg[jj] - e_pos[jj]) - (thicknesses[jj]/4)*(e_neg[jj] + e_pos[jj]),
                ikh[jj]*(-e_neg[jj] + e_pos[jj])/(4*sigma_1d[jj])]
                ], dtype=complex)

        # Compute 2x2 matrix for bottom layer
        dQdsig.append(np.array([[0, 0], [0.5*a[-1]/sigma_1d[-1], -0.5*a[-1]/sigma_1d[-1]]], dtype=complex))

        return dQdsig

    def _get_thicknesses_derivative_matricies_1d(self, src, thicknesses, sigma_1d):
        """
        For a given source (frequency) and layered Earth, return the list containing
        the derivative of each layer's propagator matrix with respect to thickness.
        
        :param SimPEG.electromagnetics.sources.AnalyticPlanewave1D src: Analytic 1D planewave source
        :param np.array thicknesses: layer thicknesses (nLayers-1,)
        :param np.array sigma_1d: layer conductivities (nLayers,)
        :return list dQdig: list containing matrix for each layer [nLayers-1,]
        """
        
        # Get frequency for planewave source
        f = src.frequency

        # Admittance for all layers
        a = self._get_admittance(f, sigma_1d)

        # List of empty places for each 2x2 array dQdh_j
        dQdh = len(thicknesses)*[None]

        # Compute exponent terms
        e_neg = np.exp(-(1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*thicknesses)
        e_pos = np.exp( (1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*thicknesses)

        C = 1j*np.pi*f*mu_0  # Constant

        # Create dQdh matrix for each layer
        for jj in range(0, len(thicknesses)):

            dQdh[jj] = C*np.array([
                [a[jj]*(-e_neg[jj] + e_pos[jj]), (-e_neg[jj] - e_pos[jj])],
                [a[jj]**2*(-e_neg[jj] - e_pos[jj]), a[jj]*(-e_neg[jj] + e_pos[jj])]
                ], dtype=complex)

        return dQdh


    def _compute_dMdsig_jj(self, Q, dQdsig, jj):
        """
        Combine propagator matricies
        """

        if len(Q) > 1:
            return np.linalg.multi_dot(Q[0:jj] + [dQdsig[jj]] + Q[jj+1:])
        else:
            return dQdsig[0]



class Simulation1DLayers(BaseSimulation1D):

    """
    Simulation class for the 1D MT problem using propagator matrix solution.
    """

    # Add layer thickness as invertible property
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers"
    )

    # Instantiate
    def __init__(self, **kwargs):
        BaseSimulation1D.__init__(self, **kwargs)


    def fields(self, m):
        """
        Compute the complex impedance for a given model.

        :param np.array m: inversion model (nP,)
        :return f: complex impedances
        """

        if m is not None:
            self.model = m

        f = []

        # For each source
        for source_ii in self.survey.source_list:

            # We can parallelize this
            Q = self._get_propagator_matricies_1d(
                    source_ii, self.thicknesses, self.sigma
                    )

            # Create final matix
            if len(Q) > 1:
                M = np.linalg.multi_dot(Q)
            else:
                M = Q[0]

            # Add to fields
            f.append(M[0, 1]/M[1, 1])

        return f


    def dpred(self, m=None, f=None):
        """
        Predict data vector for a given model.

        :param np.array m: inversion model (nP,)
        :return d: data vector
        """

        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        d = []

        # For each source and receiver, compute the datum.
        for ii in range(0, len(self.survey.source_list)):
            src = self.survey.source_list[ii]
            for rx in src.receiver_list:
                if rx.component is 'real':
                    d.append(f[ii].real)
                elif rx.component is 'imag':
                    d.append(f[ii].imag)
                elif rx.component is 'app_res':
                    d.append(np.abs(f[ii])**2/(2*np.pi*src.frequency*mu_0))
       
        return mkvc(np.hstack(d))

    def Jvec(self, m, v, f=None, method=None):
        """
        Sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with
            (nP,)
        :param String method: Choose from 'approximate' or 'analytic'
        :return: Jv (ndata,)
        """

        if method == None:
            method = self.sensitivity_method

        # 1st order computation
        if method == 'approximate':
            factor = 0.001
            Jv = np.zeros((self.survey.nD), dtype=float)
            for ii in range(0, len(m)):
                m1 = m.copy()
                m2 = m.copy()
                dm = np.max([factor*np.abs(m[ii]), 1e-6])
                m1[ii] = m[ii] - 0.5*dm
                m2[ii] = m[ii] + 0.5*dm
                d1 = self.dpred(m1)
                d2 = self.dpred(m2)
                Jv = Jv + v[ii]*(d2 - d1)/dm  # =+ doesn't keep floating point accuracy

            return Jv

        # 2nd order computation
        elif method == 'analytic':

            self.model = m

            if self.thicknessesMap == None:
                v = self.sigmaDeriv*v
            else:
                v = sp.sparse.vstack([self.sigmaDeriv, self.thicknessesDeriv])*v

            Jv = []

            for source_ii in self.survey.source_list:

                # Get Propagator matricies
                Q = self._get_propagator_matricies_1d(
                        source_ii, self.thicknesses, self.sigma
                        )

                # Create product of propagator matricies
                if len(Q) > 1:
                    M = np.linalg.multi_dot(Q)
                else:
                    M = Q[0]

                # Get sigma derivative matricies
                dQdsig = self._get_sigma_derivative_matricies_1d(
                        source_ii, self.thicknesses, self.sigma
                        )

                dMdsig = np.zeros((4, len(self.sigma)), dtype=float)

                for jj in range(0, len(self.sigma)):
                    if len(Q) > 1:
                        dMdsig_jj =  np.linalg.multi_dot(Q[0:jj] + [dQdsig[jj]] + Q[jj+1:])
                    else:
                        dMdsig_jj =  dQdsig[0]

                    dMdsig[:, jj] = np.r_[
                        dMdsig_jj[0, 1].real,
                        dMdsig_jj[0, 1].imag,
                        dMdsig_jj[1, 1].real,
                        dMdsig_jj[1, 1].imag
                        ]

                # Get h derivative matricies
                if self.thicknessesMap != None:

                    dQdh = self._get_thicknesses_derivative_matricies_1d(
                            source_ii, self.thicknesses, self.sigma
                            )

                    dMdh = np.zeros((4, len(self.thicknesses)), dtype=float)
                    for jj in range(0, len(self.thicknesses)):
                        dMdh_jj = np.linalg.multi_dot(Q[0:jj] + [dQdh[jj]] + Q[jj+1:])
                        dMdh[:, jj] = np.r_[
                            dMdh_jj[0, 1].real,
                            dMdh_jj[0, 1].imag,
                            dMdh_jj[1, 1].real,
                            dMdh_jj[1, 1].imag
                            ]

                # Compute for each receiver
                for rx in source_ii.receiver_list:
                    if rx.component is 'real':
                        C = 2*(M[0, 1].real*M[1, 1].real + M[0, 1].imag*M[1, 1].imag)/np.abs(M[1, 1])**2
                        A = (
                            np.abs(M[1, 1])**-2*np.c_[
                                M[1, 1].real,
                                M[1, 1].imag,
                                M[0, 1].real-C*M[1, 1].real,
                                M[0, 1].imag-C*M[1, 1].imag
                                ]
                            )
                    elif rx.component is 'imag':
                        C = 2*(-M[0, 1].real*M[1, 1].imag + M[0, 1].imag*M[1, 1].real)/np.abs(M[1, 1])**2
                        A = (
                            np.abs(M[1, 1])**-2*np.c_[
                                -M[1, 1].imag,
                                M[1, 1].real,
                                M[0, 1].imag-C*M[1, 1].real,
                                -M[0, 1].real-C*M[1, 1].imag
                                ]
                            )
                    elif rx.component is 'app_res':
                        rho_a = np.abs(M[0, 1]/M[1, 1])**2/(2*np.pi*source_ii.frequency*mu_0)
                        A = (
                            2*np.abs(M[1, 1])**-2*np.c_[
                                M[0, 1].real/(2*np.pi*source_ii.frequency*mu_0),
                                M[0, 1].imag/(2*np.pi*source_ii.frequency*mu_0),
                                -rho_a*M[1, 1].real,
                                -rho_a*M[1, 1].imag
                                ]
                            )

                    # Compute row of sensitivity
                    if self.thicknessesMap == None:
                        Jrow = np.dot(A, dMdsig)
                    else:
                        Jrow = np.dot(A, np.hstack([dMdsig, dMdh]))

                    Jv.append(np.dot(Jrow, v))

            return mkvc(np.vstack(Jv))


    def Jtvec(self, m, v, f=None, method=None):
        """
        Transpose of sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with
            (nD,)
        :param String method: Choose from 'approximate' or 'analytic'
        :return: Jtv (nP,)
        """

        if method == None:
            method = self.sensitivity_method

        # 1st order method
        if method == 'approximate':
            factor = 0.001
            Jtv = np.zeros(len(m), dtype=float)
            for ii in range(0, len(m)):
                m1 = m.copy()
                m2 = m.copy()
                dm = np.max([factor*np.abs(m[ii]), 1e-6])
                m1[ii] = m[ii] - 0.5*dm
                m2[ii] = m[ii] + 0.5*dm
                d1 = self.dpred(m1)
                d2 = self.dpred(m2)
                Jtv[ii] = np.dot((d2 - d1)/dm, v)

            return Jtv

        # 2nd order computation
        elif method == 'analytic':

            self.model = m
            Jtv = np.zeros(len(m), dtype=float)

            COUNT = 0
            for source_ii in self.survey.source_list:

                # Get Propagator matricies
                Q = self._get_propagator_matricies_1d(
                        source_ii, self.thicknesses, self.sigma
                        )

                # Create product of propagator matricies
                if len(Q) > 1:
                    M = np.linalg.multi_dot(Q)
                else:
                    M = Q[0]

                # Get sigma derivative matricies
                dQdsig = self._get_sigma_derivative_matricies_1d(
                        source_ii, self.thicknesses, self.sigma
                        )

                dMdsig = np.zeros((4, len(self.sigma)), dtype=float)

                for jj in range(0, len(self.sigma)):
                    if len(Q) > 1:
                        dMdsig_jj =  np.linalg.multi_dot(Q[0:jj] + [dQdsig[jj]] + Q[jj+1:])
                    else:
                        dMdsig_jj =  dQdsig[0]

                    dMdsig[:, jj] = np.r_[
                        dMdsig_jj[0, 1].real,
                        dMdsig_jj[0, 1].imag,
                        dMdsig_jj[1, 1].real,
                        dMdsig_jj[1, 1].imag
                        ]

                # Get h derivative matricies
                if self.thicknessesMap != None:

                    dQdh = self._get_thicknesses_derivative_matricies_1d(
                            source_ii, self.thicknesses, self.sigma
                            )

                    dMdh = np.zeros((4, len(self.thicknesses)), dtype=float)

                    for jj in range(0, len(self.thicknesses)):
                        dMdh_jj = np.linalg.multi_dot(Q[0:jj] + [dQdh[jj]] + Q[jj+1:])
                        dMdh[:, jj] = np.r_[
                            dMdh_jj[0, 1].real,
                            dMdh_jj[0, 1].imag,
                            dMdh_jj[1, 1].real,
                            dMdh_jj[1, 1].imag
                            ]

                # Compute for each receiver
                for rx in source_ii.receiver_list:
                    if rx.component is 'real':
                        C = 2*(M[0, 1].real*M[1, 1].real + M[0, 1].imag*M[1, 1].imag)/np.abs(M[1, 1])**2
                        A = (
                            np.abs(M[1, 1])**-2*np.c_[
                                M[1, 1].real,
                                M[1, 1].imag,
                                M[0, 1].real-C*M[1, 1].real,
                                M[0, 1].imag-C*M[1, 1].imag
                                ]
                            )
                    elif rx.component is 'imag':
                        C = 2*(-M[0, 1].real*M[1, 1].imag + M[0, 1].imag*M[1, 1].real)/np.abs(M[1, 1])**2
                        A = (
                            np.abs(M[1, 1])**-2*np.c_[
                                -M[1, 1].imag,
                                M[1, 1].real,
                                M[0, 1].imag-C*M[1, 1].real,
                                -M[0, 1].real-C*M[1, 1].imag
                                ]
                            )
                    elif rx.component is 'app_res':
                        rho_a = np.abs(M[0, 1]/M[1, 1])**2/(2*np.pi*source_ii.frequency*mu_0)
                        A = (
                            2*np.abs(M[1, 1])**-2*np.c_[
                                M[0, 1].real/(2*np.pi*source_ii.frequency*mu_0),
                                M[0, 1].imag/(2*np.pi*source_ii.frequency*mu_0),
                                -rho_a*M[1, 1].real,
                                -rho_a*M[1, 1].imag
                                ]
                            )

                    # Compute column of sensitivity transpose
                    if self.thicknessesMap == None:
                        Jtcol = np.dot(A, dMdsig)
                    else:
                        Jtcol = np.dot(A, np.hstack([dMdsig, dMdh]))
                    
                    Jtv = Jtv + v[COUNT]*Jtcol
                    COUNT = COUNT + 1

            if self.thicknessesMap == None:
                return mkvc(Jtv*self.sigmaDeriv.T)
            else:
                return mkvc(Jtv*sp.sparse.vstack([self.sigmaDeriv, self.thicknessesDeriv]).T)




        
