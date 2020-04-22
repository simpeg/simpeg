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
    Simulation class for the 1D MT problem using propagator matrix solution.
    """

    # Must be 1D survey object
    survey = properties.Instance(
        "a Survey1D survey object", Survey1D, required=True
    )

    sensitivity_method = properties.StringChoice(
        "Choose 1st or 2nd order computations with sensitivity matrix ('1st_order', '2nd_order')", {
            "1st_order": [],
            "2nd_order": []
        }
    )

    # Add layer thickness as invertible property
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers"
    )



    # Store sensitivity
    _Jmatrix = None
    fix_Jmatrix = False
    storeJ = properties.Bool("store the sensitivity", default=False)

    _frequency_vector = None
    

    @ property
    def frequency_vector(self):
        
        if getattr(self, '_frequency_vector', None) is None:
            if self._frequency_vector == None:
                fvec = []
                for src in self.survey.source_list:
                    fvec.append(src.frequency * np.ones(len(src.receiver_list)))
                self._frequency_vector = np.hstack(fvec)

        return self._frequency_vector


    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super(Simulation1DRecursive, self).deleteTheseOnModelUpdate
        if self.fix_Jmatrix:
            return toDelete

        if self._Jmatrix is not None:
            toDelete += ['_Jmatrix']
        return toDelete



    # Instantiate
    def __init__(self, sensitivity_method="1st_order", **kwargs):
        
        self.sensitivity_method = sensitivity_method

        BaseEMSimulation.__init__(self, **kwargs)





    def _get_recursive_impedances_1d(self, fvec, thicknesses, sigma_1d):
        """
        For a given source and layered Earth model, this returns the list of
        propagator matricies.

        :param float: frequency in Hz
        :param np.array thicknesses: layer thicknesses (nLayers-1,)
        :param np.array sigma_1d: layer conductivities (nLayers,)
        :return list Q: list containing matrix for each layer [nLayers,]
        """
        
        omega = 2*np.pi*fvec
        n_layer = len(sigma_1d)

        # Bottom layer
        alpha = np.sqrt(1j*omega*mu_0*sigma_1d[-1])
        ratio = alpha/sigma_1d[-1]
        Z = -ratio

        # Work from lowest layer to top layer
        for ii in range(n_layer-2, -1, -1):
            alpha = np.sqrt(1j*omega*mu_0*sigma_1d[ii])
            ratio = alpha/sigma_1d[ii]
            tanh = np.tanh(alpha*thicknesses[ii])

            top = Z/ratio - tanh
            bot = 1 - Z/ratio*tanh
            Z = ratio*(top/bot)

        return Z


    def _get_sigma_sensitivities(self, fvec, thicknesses, sigma_1d):
        omega = 2*np.pi*fvec
        n_layer = len(sigma_1d)
        J = np.empty((len(fvec), n_layer), dtype=np.complex128)

        alpha = np.sqrt(1j*omega*mu_0*sigma_1d[-1])
        alpha_ds = 1j*omega*mu_0/(2*alpha)

        ratio = alpha/sigma_1d[-1]
        ratio_ds = alpha_ds/sigma_1d[-1] - ratio/sigma_1d[-1]

        Z = -ratio
        dZ_dsigma = -ratio_ds
        J[:, -1] = dZ_dsigma
        for ii in range(n_layer-2, -1, -1):
            alpha = np.sqrt(1j*omega*mu_0*sigma_1d[ii])
            alpha_ds = 1j*omega*mu_0/(2*alpha)

            ratio = alpha/sigma_1d[ii]
            ratio_ds = alpha_ds/sigma_1d[ii] - ratio/sigma_1d[ii]

            tanh = np.tanh(alpha*thicknesses[ii])
            tanh_ds = thicknesses[ii]*(1-tanh*tanh)*alpha_ds

            top = Z/ratio - tanh
            top_ds = -tanh_ds - Z/(ratio*ratio)*ratio_ds
            top_dZ = 1/ratio

            bot = 1 - Z/ratio*tanh
            bot_ds = (-Z/ratio) * tanh_ds + Z*tanh/(ratio**2)*ratio_ds
            bot_dZ = -(tanh/ratio)

            Z = ratio*(top/bot)
            Z_dratio = (top/bot)
            Z_dtop = ratio/bot
            Z_dbot = -ratio*top/(bot*bot)

            dZ_ds = Z_dtop*top_ds + Z_dbot*bot_ds + Z_dratio*ratio_ds
            dZ_dZp1 = Z_dtop*top_dZ + Z_dbot*bot_dZ

            J[:, ii] = dZ_ds
            J[:, ii+1:] *= dZ_dZp1[:, None]
        return J


    def _get_thickness_sensitivities(self, fvec, thicknesses, sigma_1d):
    
        omega = 2*np.pi*fvec
        n_layer = len(sigma_1d)
        J = np.empty((len(fvec), n_layer-1), dtype=np.complex128)

        alpha = np.sqrt(1j*omega*mu_0*sigma_1d[-1])
        ratio = alpha/sigma_1d[-1]

        Z = -ratio
        for ii in range(n_layer-2, -1, -1):
            alpha = np.sqrt(1j*omega*mu_0*sigma_1d[ii])
            ratio = alpha/sigma_1d[ii]

            tanh = np.tanh(alpha*thicknesses[ii])
            tanh_dh = alpha*(1-tanh*tanh)

            top = Z/ratio - tanh
            top_dh = -tanh_dh
            top_dZ = 1/ratio

            bot = 1 - Z/ratio*tanh
            bot_dh = (-Z/ratio) * tanh_dh
            bot_dZ = -(tanh/ratio)

            Z = ratio*(top/bot)
            Z_dtop = ratio/bot
            Z_dbot = -ratio*top/(bot*bot)

            dZ_dh = Z_dtop*top_dh + Z_dbot*bot_dh
            dZ_dZp1 = Z_dtop*top_dZ + Z_dbot*bot_dZ

            J[:, ii] = dZ_dh
            J[:, ii+1:] *= dZ_dZp1[:, None]
        return J


    




    def fields(self, m):
        """
        Compute the complex impedance for a given model.

        :param np.array m: inversion model (nP,)
        :return f: complex impedances
        """

        if m is not None:
            self.model = m

        # Compute complex impedances for each datum
        complex_impedance = self._get_recursive_impedances_1d(
                self.frequency_vector, self.thicknesses, self.sigma
                )

        # For each source-receiver pair, compute datum
        f = []
        COUNT = 0
        for source_ii in self.survey.source_list:
            for rx in source_ii.receiver_list:
                if rx.component is 'real':
                    f.append(np.real(complex_impedance[COUNT]))
                elif rx.component is 'imag':
                    f.append(np.real(complex_impedance[COUNT]))
                elif rx.component is 'app_res':
                    f.append(np.abs(complex_impedance[COUNT])**2/(2*np.pi*source_ii.frequency*mu_0))
                COUNT = COUNT + 1

        return np.array(f)


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

        return f


    def getJ(self, m, f=None, sensitivity_method=None):

        """
        Compute and store the sensitivity matrix.

        :param numpy.ndarray m: inversion model (nP,)
        :param String method: Choose from '1st_order' or '2nd_order'
        :return: J (ndata, nP)
        """
        
        if sensitivity_method == None:
            sensitivity_method = self.sensitivity_method

        if self._Jmatrix is not None:

            pass

        elif sensitivity_method == '1st_order':

            # 1st order computation
            self.model = m

            N = self.survey.nD
            M = self.model.size
            Jmatrix = np.zeros((N, M), dtype=float, order='F')

            factor = 0.01
            for ii in range(0, len(m)):
                m1 = m.copy()
                m2 = m.copy()
                dm = np.max([factor*np.abs(m[ii]), 1e-3])
                m1[ii] = m[ii] - 0.5*dm
                m2[ii] = m[ii] + 0.5*dm
                d1 = self.dpred(m1)
                d2 = self.dpred(m2)
                Jmatrix[:, ii] = (d2 - d1)/dm 

            self._Jmatrix = Jmatrix

        elif sensitivity_method == '2nd_order':

            self.model = m

            # Sensitivity of parameters with model
            dMdm = []
            Jmatrix = []
            
            if self.sigmaMap != None:
                dMdm.append(self.sigmaDeriv)
                Jmatrix.append(
                    self._get_sigma_sensitivities(
                        self.frequency_vector, self.thicknesses, self.sigma
                        )
                    )

            if self.thicknessesMap != None:
                dMdm.append(self.thicknessesDeriv)
                Jmatrix.append(
                    self._get_thickness_sensitivities(
                        self.frequency_vector, self.thicknesses, self.sigma
                        )
                    )
            
            if len(dMdm) == 1:
                dMdm = dMdm[0]
                Jmatrix = Jmatrix[0]
            else:
                dMdm = sp.sparse.vstack(dMdm[:])
                Jmatrix = np.hstack(Jmatrix[:])

            COUNT = 0
            for source_ii in self.survey.source_list:
                for rx in source_ii.receiver_list:
                    if rx.component is 'real':
                        Jmatrix[COUNT, :] = np.real(Jmatrix[COUNT, :])
                    elif rx.component is 'imag':
                        Jmatrix[COUNT, :] = np.imag(Jmatrix[COUNT, :])
                    elif rx.component is 'app_res':
                        Z = self._get_recursive_impedances_1d(
                            source_ii.frequency, self.thicknesses, self.sigma
                            )
                        Jmatrix[COUNT, :] = (
                            (np.pi*source_ii.frequency*mu_0)**-1 *
                            (np.real(Z)*np.real(Jmatrix[COUNT, :]) + np.imag(Z)*np.imag(Jmatrix[COUNT, :]))
                            )


                    COUNT = COUNT + 1

            self._Jmatrix = np.real(Jmatrix)*dMdm

        return self._Jmatrix


    def Jvec(self, m, v, f=None, sensitivity_method=None):
        """
        Sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with
            (nP,)
        :param String method: Choose from 'approximate' or 'analytic'
        :return: Jv (ndata,)
        """

        if sensitivity_method == None:
            sensitivity_method = self.sensitivity_method

        J = self.getJ(m, f=None, sensitivity_method=sensitivity_method)

        return mkvc(np.dot(J, v))


    def Jtvec(self, m, v, f=None, sensitivity_method=None):
        """
        Transpose of sensitivity times a vector.

        :param numpy.ndarray m: inversion model (nP,)
        :param numpy.ndarray v: vector which we take sensitivity product with
            (nD,)
        :param String method: Choose from 'approximate' or 'analytic'
        :return: Jtv (nP,)
        """

        if sensitivity_method == None:
            sensitivity_method = self.sensitivity_method

        J = self.getJ(m, f=None, sensitivity_method=sensitivity_method)

        return mkvc(np.dot(v, J))
        
