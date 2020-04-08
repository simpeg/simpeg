import numpy as np
import properties
from scipy.constants import mu_0

from ...utils import mkvc, sdiag, Zero
from ..base import BaseEMSimulation
from ...data import Data
from ... import props

from .survey import Survey1D


class Simulation1DLayers(BaseEMSimulation):
    """
    
    """

    _Q = None
    _dQdsig = None

    forward_only = properties.Bool("store propagator matricies", default=True)

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers"
    )

    survey = properties.Instance(
        "a Survey1D survey object", Survey1D, required=True
    )

    storeJ = properties.Bool(
        "store the sensitivity", default=False
    )

    def __init__(self, **kwargs):
        BaseEMSimulation.__init__(self, **kwargs)


    def _get_wavenumber(self, f, sig):

        """
        Wavenumbers for all frequencies
        """

        return (1 - 1j)*np.sqrt(np.pi*mu_0*f*sig)


    def _get_admittance(self, f, sig):

        """
        Admittances for all layers
        """

        return (1 - 1j)*np.sqrt(sig/(4*np.pi*f*mu_0))


    def _get_propagator_matricies_for_source(self, src, layers, sigma_1d):

        
        # Get frequency for planewave source
        f = src.frequency

        # Admittance for all layers
        a = self._get_admittance(f, sigma_1d)

        # List of empty places for each 2x2 array Qj
        Q = len(layers)*[None]

        # Compute exponent terms
        e_neg = np.exp(-(1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*layers)
        e_pos = np.exp( (1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*layers)

        # Create Q matrix for each layer
        for jj in range(0, len(layers)):

            Q[jj] = (2/a[jj])*np.array([
                [a[jj]*(e_neg[jj] + e_pos[jj]), (e_neg[jj] - e_pos[jj])],
                [a[jj]**2*(e_neg[jj] - e_pos[jj]), a[jj]*(e_neg[jj] + e_pos[jj])]
                ], dtype=complex)

        # Compute 2x2 matrix for bottom layer
        Q.append(np.array([[1, 1], [a[-1], -a[-1]]], dtype=complex))

        return Q

    def _get_sigma_derivative_matricies_for_source(self, src, layers, sigma_1d):

        
        # Get frequency for planewave source
        f = src.frequency

        # Admittance for all layers
        a = self._get_admittance(f, sigma_1d)

        # List of empty places for each 2x2 array Qj
        dQdsig = len(layers)*[None]

        # Compute exponent terms
        e_neg = np.exp(-(1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*layers)
        e_pos = np.exp( (1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*layers)

        ikh = (1 + 1j)*np.sqrt(np.pi*mu_0*f*sigma_1d[0:-1])*layers  # product of i, wavenumber and layer thickness


        # Create Q matrix for each layer
        for jj in range(0, len(layers)):

            dQdsig[jj] = np.array([
                [
                ikh[jj]*(-e_neg[jj] + e_pos[jj])/sigma_1d[jj],
                (-1/(sigma_1d[jj]*a[jj]))*(e_neg[jj] - e_pos[jj]) - (layers[jj]/a[jj]**2)*(e_neg[jj] + e_pos[jj])
                ],[
                (a[jj]/sigma_1d[jj])*(e_neg[jj] - e_pos[jj]) - layers[jj]*(e_neg[jj] + e_pos[jj]),
                ikh[jj]*(-e_neg[jj] + e_pos[jj])/sigma_1d[jj]]
                ], dtype=complex)

        # Compute 2x2 matrix for bottom layer
        dQdsig.append(np.array([[0, 0], [0.5*a[-1]/sigma_1d[-1], -0.5*a[-1]/sigma_1d[-1]]], dtype=complex))

        return dQdsig

    def _compute_dMdsig_jj(self, Q, dQdsig, jj):

        return np.linalg.multi_dot(Q[0:jj] + [dQdsig[jj]] + Q[jj+1:])


    def fields(self, m):

        replace_Q = False

        if m is not None:
            self.model = m

        f = []

        for source_ii in self.survey.source_list:

            # We can parallelize this
            Q = self._get_propagator_matricies_for_source(
                    source_ii, self.thicknesses, self.model
                    )

            # Create final matix
            M = np.linalg.multi_dot(Q)

            # Add to fields
            f.append(M[0, 1]/M[1, 1])


        return f


    def dpred(self, m=None, f=None):

        if m is not None:
            self.model = m

        if f is None:
            f = self.fields(m)

        d = []

        for ii in range(0, len(self.survey.source_list)):
            src = self.survey.source_list[ii]
            for rx in src.receiver_list:

                if rx.component is 'real':
                    d.append(f[ii].real)
                elif rx.component is 'imag':
                    d.append(f[ii].imag)
                elif rx.component is 'app_res':
                    d.append(np.abs(f[ii])**2/(2*np.pi*src.frequency*mu_0))
       
        return np.hstack(d)



    def Jvec(self, m, v):

        self.model = m

        Jv = []

        for source_ii in self.survey.source_list:

            # Get Propagator matricies
            Q = self._get_propagator_matricies_for_source(
                    source_ii, self.thicknesses, self.model
                    )

            # Create final matix
            M = np.linalg.multi_dot(Q)

            # Get derivative matricies
            dQdsig = self._get_sigma_derivative_matricies_for_source(
                    source_ii, self.thicknesses, self.model
                    )

            dMdsig = np.empty([4, len(self.model)])

            for jj in range(0, len(self.model)):
                
                dMdsig_jj = self._compute_dMdsig_jj(Q, dQdsig, jj)
                dMdsig[:, jj] = np.r_[
                    dMdsig_jj[0, 1].real,
                    dMdsig_jj[0, 1].imag,
                    dMdsig_jj[1, 1].real,
                    dMdsig_jj[1, 1].imag
                    ]

            for rx in source_ii.receiver_list:

                if rx.component is 'real':

                    C = 2*(M[0, 1].real*M[1, 1].real + M[0, 1].imag*M[1, 1].imag)/np.abs(M[1, 1])**2

                    A = (
                        np.abs(M[1, 1])**-2*
                        np.c_[M[1, 1].real, M[1, 1].imag, M[0, 1].real-C*M[1, 1].real, M[0, 1].imag-C*M[1, 1].imag]
                        )

                elif rx.component is 'imag':

                    C = 2*(-M[0, 1].real*M[1, 1].imag + M[0, 1].imag*M[1, 1].real)/np.abs(M[1, 1])**2

                    A = (
                        np.abs(M[1, 1])**-2*
                        np.c_[-M[1, 1].imag, M[1, 1].real, M[0, 1].imag-C*M[1, 1].real, -M[0, 1].real-C*M[1, 1].imag]
                        )

                elif rx.component is 'app_res':

                    rho_a = np.abs(M[0, 1]/M[1, 1])**2/(2*np.pi*source_ii.frequency*mu_0)
                    A = (
                        (2/np.abs(M[1, 1])**2)*
                        np.c_[M[0, 1].real/(source_ii.frequency*mu_0), M[0, 1].imag/(source_ii.frequency*mu_0), -rho_a*M[1, 1].real, -rho_a*M[1, 1].imag]
                        )

                Jrow = np.dot(A, dMdsig)

                Jv.append(np.dot(Jrow, v))

        return mkvc(np.vstack(Jv))



    def Jtvec(self, m, v):

        self.model = m

        COUNT = 0

        Jtv = np.zeros(len(v))

        for source_ii in self.survey.source_list:

            # Get Propagator matricies
            Q = self._get_propagator_matricies_for_source(
                    source_ii, self.thicknesses, self.model
                    )

            # Create final matix
            M = np.linalg.multi_dot(Q)

            # Get derivative matricies
            dQdsig = self._get_sigma_derivative_matricies_for_source(
                    source_ii, self.thicknesses, self.model
                    )

            dMdsig = np.empty([4, len(self.model)])

            for jj in range(0, len(self.model)):
                
                dMdsig_jj = self._compute_dMdsig_jj(Q, dQdsig, jj)
                dMdsig[:, jj] = np.r_[
                    dMdsig_jj[0, 1].real,
                    dMdsig_jj[0, 1].imag,
                    dMdsig_jj[1, 1].real,
                    dMdsig_jj[1, 1].imag
                    ]

            for rx in source_ii.receiver_list:

                if rx.component is 'real':

                    C = 2*(M[0, 1].real*M[1, 1].real + M[0, 1].imag*M[1, 1].imag)/np.abs(M[1, 1])**2

                    A = (
                        np.abs(M[1, 1])**-2*
                        np.c_[M[1, 1].real, M[1, 1].imag, M[0, 1].real-C*M[1, 1].real, M[0, 1].imag-C*M[1, 1].imag]
                        )

                elif rx.component is 'imag':

                    C = 2*(-M[0, 1].real*M[1, 1].imag + M[0, 1].imag*M[1, 1].real)/np.abs(M[1, 1])**2

                    A = (
                        np.abs(M[1, 1])**-2*
                        np.c_[-M[1, 1].imag, M[1, 1].real, M[0, 1].imag-C*M[1, 1].real, -M[0, 1].real-C*M[1, 1].imag]
                        )

                elif rx.component is 'app_res':

                    rho_a = np.abs(M[0, 1]/M[1, 1])**2/(2*np.pi*source_ii.frequency*mu_0)
                    A = (
                        (2/np.abs(M[1, 1])**2)*
                        np.c_[M[0, 1].real/(source_ii.frequency*mu_0), M[0, 1].imag/(source_ii.frequency*mu_0), -rho_a*M[1, 1].real, -rho_a*M[1, 1].imag]
                        )

                Jcol = np.dot(A, dMdsig)

                Jtv =+ v[COUNT]*Jcol
                COUNT =+ 1

        return mkvc(Jtv)







        
