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

        return (1 - 1j)*np.sqrt(sig/(4*np.pi*mu_0*f))


    def _get_M(self, fi, layers, sigma_1d):

        a = self._get_admittance(fi, sigma_1d)

        # List of empty places for each 2x2 array
        M = len(layers)*[None]

        # Compute Pn+1
        Pend = np.array([[1, 1], [a[-1], -a[-1]]], dtype=complex)

        # Compute exponents
        e_neg = np.exp(-(1 + 1j)*np.sqrt(np.pi*mu_0*fi*sigma_1d[0:-1])*layers)
        e_pos = np.exp( (1 + 1j)*np.sqrt(np.pi*mu_0*fi*sigma_1d[0:-1])*layers)

        # parallelize this later
        for jj in range(0, len(layers)):

            M[jj] = (2/a[jj])*np.array([
                [a[jj]*(e_neg[jj] + e_pos[jj]), (e_neg[jj] - e_pos[jj])],
                [a[jj]**2*(e_neg[jj] - e_pos[jj]), a[jj]*(e_neg[jj] + e_pos[jj])]
                ], dtype=complex)

        M.append(Pend)

        return np.linalg.multi_dot(M)



    def fields(self, m):

        if m is not None:
            self.model = m

        f = [] 

        for source_ii in self.survey.source_list:

            M = self._get_M(
                    source_ii.frequency, self.thicknesses, m
                    )

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
                    d.append(f[ii].real())
                elif rx.component is 'imag':
                    d.append(f[ii].imag())
                elif rx.component is 'app_res':
                    d.append(np.abs(f[ii])**2/(2*np.pi*src.frequency*4*np.pi*1e-7))

        
        return d


        
