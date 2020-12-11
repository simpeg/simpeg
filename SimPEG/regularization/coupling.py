import numpy as np
import scipy.sparse as sp

from .. import utils
from .base import BaseRegularization

class BaseCoupling(BaseRegularization):

    '''
    Base class for the coupling term in joint inversions. Inherit this for building
    your own coupling term.  The BaseCoupling assumes couple two different
    geophysical models through one coupling term. However, if you wish
    to combine more than two models, e.g., 3 models,
    you may want to add a total of three coupling terms:

    e.g., lambda1*(m1, m2) + lambda2*(m1, m3) + lambda3*(m2, m3)

    where, lambdas are weights for coupling terms. m1, m2 and m3 indicate
    three different models.

    :param discretize.base.BaseMesh mesh: SimPEG mesh

    Contributors:
    Jae Deok Kim, Xiaolong Wei, Jiajia Sun



    '''
    def __init__(self, mesh, indActive, mapping, **kwargs):

        self.as_super.__init__(mesh, indActive=indActive, mapping=mapping)

    def deriv(self):
        '''
        First derivative of the coupling term with respect to individual models.
        Returns an array of dimensions [k*M,1],
        k: number of models we are inverting for.
        M: number of cells in each model.

        '''
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def deriv2(self):
        '''
        Second derivative of the coupling term with respect to individual models.
        Returns either an array of dimensions [k*M,1] (v is not None), or
        sparse matrix of dimensions [k*M, k*M] (v is None).
        k: number of models we are inverting for.
        M: number of cells in each model.

        '''
        raise NotImplementedError(
            "The method _deriv2 has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def __call__(self):
        ''' Returns the computed value of the coupling term. '''
        raise NotImplementedError(
            "The method __call__ has not been implemented for {}".format(
                self.__class__.__name__
            )
        )


###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################


class CrossGradient(BaseCoupling):
    '''
    The cross-gradient constraint for joint inversions.

    ..math::
        \phi_c(\mathbf{m_1},\mathbf{m_2}) = \lambda \sum_{i=1}^{M} \|
        \nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2

    All methods assume that we are working with two models only.

    '''
    def __init__(self, mesh, indActive, mapping, **kwargs):

        self.as_super.__init__(mesh, indActive, mapping, **kwargs)
        self.map1, self.map2 = mapping.maps # Assume a map has been passed for each model.

        assert mesh.dim in (2,3), 'Cross-Gradient is only defined for 2D or 3D'

    def models(self, ind_models):
        '''
        Method to pass models to CrossGradient object and ensures models are
        compatible for further use. Checks that models are of same size.

        :param container of numpy.ndarray ind_models: [model1, model2,...]

        rtype: list of numpy.ndarray models: [model1, model2,...]
        return: models

        '''
        models = []
        n = len(ind_models)
        for i in range(n):
            # check that the models are either a list, tuple, or np.ndarray
            assert isinstance(ind_models[i], (list,tuple,np.ndarray))
            if isinstance(ind_models[i], (list,tuple)):
                ind_models[i] = np.array(ind_models[i])

        # check if models are of same size
        it = iter(ind_models)
        length = len(next(it))
        if not all(len(l)==length for l in it):
            raise ValueError('not all models are of the same size!')

        for i in range(n):
            models.append(ind_models[i])

        return models


    def calculate_gradient(self, model, normalize=False):
        '''
        Calculate the spatial gradients of the model using central difference.

        Concatenates gradient components into a single array.
        [[x_grad1, y_grad1, z_grad1],
         [x_grad2, y_grad2, z_grad2],
         [x_grad3, y_grad3, z_grad3],...]

        :param numpy.ndarray model: model

        :rtype: numpy.ndarray
        :return: gradient_vector: array where each row represents a model cell,
                 and each column represents a component of the gradient.

        '''

        if self.regmesh.mesh.dim == 2:
            x_grad = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx.dot(model))
            y_grad = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy.dot(model))

            return np.column_stack((x_grad, y_grad))

        elif self.regmesh.mesh.dim == 3:
            x_grad = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx.dot(model))
            y_grad = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy.dot(model))
            z_grad = self.regmesh.aveFz2CC.dot(self.regmesh.cellDiffz.dot(model))

            return np.column_stack((x_grad, y_grad, z_grad))




    # def gradient_list(self, x_grad, y_grad, z_grad=np.array([])):
        # '''
        # :param numpy.ndarray x_grad: x-gradients
        # :param numpy.ndarray y_grad: y-gradients
        # :param numpy.ndarray z_grad: z-gradients
        #
        # :rtype: numpy.ndarray
        # :return: gradient_vector: array where each row represents a model cell,
        #          and each column represents a component of the gradient.
        # '''
        # if z_grad.size == 0:
        #     gradient_vector = np.c_[x_grad, y_grad]
        # else:
        #     gradient_vector = np.c_[x_grad, y_grad, z_grad]

        # return gradient_vector






    def gradient_applitude_inv(self, m1, m2, fltr=True, fltr_per=0.05):
        '''
        Computes the norms of the gradients for two models.

        :param numpy.ndarray m1: model1
        :param numpy.ndarray m2: model2
        :param bool reduced_space: if True, return only norms of active cells.
        :param bool fltr: if True, filter out predefined percentage of lowest norms.
        :param float fltr_per: Percentage of lowest norms to be filtered out.
                               Default is 0.05

        :rtype tuple of numpy.ndarray
        :return (norms_1, norms_2, tot_norms)
                norms_1: np.array containing reciprocals of the gradient norms for model1
                norms_2: np.array containing reciprocals of the gradient norms for model2
                tot_norms: np.array containing element-wise product of norms_1 and norms_2.
        '''
        per = int(fltr_per*self.regmesh.nC)

        if self.regmesh.mesh.dim == 2:
            Dx_m1, Dy_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2 = self.calculate_gradient(m2)

            norms_1 = np.sqrt(Dx_m1**2 + Dy_m1**2)
            # compute (1 / applitude of gradients) and background when norms < 1e-10
            norms_1 = np.divide(1, norms_1, out=np.zeros_like(norms_1), where=norms_1>1e-10)
            norms_2 = np.sqrt(Dx_m2**2 + Dy_m2**2)
            norms_2 = np.divide(1, norms_2, out=np.zeros_like(norms_2), where=norms_2>1e-10)

        elif self.regmesh.mesh.dim == 3:
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(m2)

            norms_1 = np.sqrt(Dx_m1**2 + Dy_m1**2 + Dz_m1**2)
            norms_1 = np.divide(1, norms_1, out=np.zeros_like(norms_1), where=norms_1>1e-10)
            norms_2 = np.sqrt(Dx_m2**2 + Dy_m2**2 + Dz_m2**2)
            norms_2 = np.divide(1, norms_2, out=np.zeros_like(norms_2), where=norms_2>1e-10)

        # set lowest 5% of norms (largest 5% of 1/norms) to 0.0
        if fltr:
            inds1 = norms_1.argsort()[-per:]
            norms_1[inds1] = 0.0
            inds2 = norms_2.argsort()[-per:]
            norms_2[inds2] = 0.0

        tot_norms = norms_1*norms_2

        return (norms_1, norms_2, tot_norms)

    def normalized_gradients(self,grad_list):
        '''
        Normalizes the spatial gradients of a model.

        :param numpy.ndarray grad_list: array where each row represents a model cell,
                                        and each column represents a component of the gradient.

        :rtype: numpy.ndarray
        :return: norm_gradient: array where the gradients have been normalized by their norms.
                 Each row represents a model cell, and each column represents the normalized
                 component of the gradient.

        '''
        elems = grad_list.shape[0]
        norm_gradients = np.zeros_like(grad_list)
        for i in range(elems):
            gradient = grad_list[i]
            norm = np.linalg.norm(gradient)
            if norm<1e-10:
                continue
            else:
                norm_gradients[i] = gradient / norm
        return norm_gradients

    def calculate_cross_gradient(self, m1, m2, normalized=False):
        '''
        Calculates the cross-gradients of two models at each cell center.

        :param numpy.ndarray m1: model1
        :param numpy.ndarray m2: model2
        :param bool normalized: normalizes gradients if True

        :rtype: numpy.ndarray
        :returns: array where at each location, we've computed the cross-product
                  of the gradients of two models.

        '''
        m1, m2 = self.models([m1,m2])

        # Compute the gradients and concatenate components.
        if self.regmesh.mesh.dim == 2:
            Dx_m1, Dy_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2 = self.calculate_gradient(m2)

            grad_list_1 = self.gradient_list(Dx_m1, Dy_m1)
            grad_list_2 = self.gradient_list(Dx_m2, Dy_m2)

            if normalized:
                grad_list_1 = self.normalized_gradients(grad_list_1)
                grad_list_2 = self.normalized_gradients(grad_list_2)

        elif self.regmesh.mesh.dim == 3:
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(m2)

            grad_list_1 = self.gradient_list(Dx_m1, Dy_m1, Dz_m1)
            grad_list_2 = self.gradient_list(Dx_m2, Dy_m2, Dz_m2)

            if normalized:
                grad_list_1 = self.normalized_gradients(grad_list_1)
                grad_list_2 = self.normalized_gradients(grad_list_2)

        cross_prod_list = []
        num_cells = len(m1)
        # for each model cell, compute the cross product of the gradient vectors.
        for x in range(num_cells):
            if self.regmesh.mesh.dim == 3:
                cross_prod_vector = np.cross(grad_list_1[x],grad_list_2[x])
                cross_prod = np.linalg.norm(cross_prod_vector)
            else:
                cross_prod = np.cross(grad_list_1[x],grad_list_2[x])
            cross_prod_list.append(cross_prod)
        cross_prod = np.array(cross_prod_list)

        return cross_prod

    def __call__(self, model, normalized=False):
        '''
        Computes the sum of all cross-gradient values at all cell centers.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        :param bool normalized: returns value of normalized cross-gradient if True

        :rtype: float
        :returns: the computed value of the cross-gradient term.


        ..math::

            \phi_c(\mathbf{m_1},\mathbf{m_2})

            = \lambda \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2

            = \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i\|^2 \ast \|\nabla \mathbf{m_2}_i\|^2
                - (\nabla \mathbf{m_1}_i \cdot \nabla \mathbf{m_2}_i )^2

            = \|\phi_{cx}\|^2 + \|\phi_{cy}\|^2 + \|\phi_{cz}\|^2 (optional strategy, not used in this script)


        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1,m2])

        if self.regmesh.mesh.dim == 2:
            Dx_m1, Dy_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2 = self.calculate_gradient(m2)

            temp1 = Dx_m1**2 + Dy_m1**2
            temp2 = Dx_m2**2 + Dy_m2**2
            term1 = temp1.dot(temp2)

            temp1 = Dx_m1*Dx_m2 + Dy_m1*Dy_m2
            term2 = (np.linalg.norm(temp1))**2
            result = term1 - term2

            if normalized:
                norms1, norms2, norms = self.gradient_applitude_inv(m1, m2, fltr=False)
                temp1 = (Dx_m1*norms1)**2 + (Dy_m1*norms1)**2
                temp2 = (Dx_m2*norms2)**2 + (Dy_m2*norms2)**2
                term1 = temp1.dot(temp2)

                temp1 = Dx_m1*Dx_m2*norms + Dy_m1*Dy_m2*norms
                term2 = (np.linalg.norm(temp1))**2
                result = term1 - term2

        elif self.regmesh.mesh.dim == 3:
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(m2)

            temp1 = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            temp2 = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            term1 = temp1.dot(temp2)

            temp1 = Dx_m1*Dx_m2 + Dy_m1*Dy_m2 + Dz_m1*Dz_m2
            term2 = (np.linalg.norm(temp1))**2
            result = term1 - term2

            if normalized:
                norms1, norms2, norms = self.gradient_applitude_inv(m1, m2, fltr=False)
                temp1 = (Dx_m1*norms1)**2 + (Dy_m1*norms1)**2 + (Dz_m1*norms1)**2
                temp2 = (Dx_m2*norms2)**2 + (Dy_m2*norms2)**2 + (Dz_m2*norms2)**2
                term1 = temp1.dot(temp2)

                temp1 = Dx_m1*Dx_m2*norms + Dy_m1*Dy_m2*norms + Dz_m1*Dz_m2*norms
                term2 = (np.linalg.norm(temp1))**2
                result = term1 - term2

        return 0.5*result

    def deriv(self, model):
        '''
        Computes the Jacobian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2,...]

        :rtype: numpy.ndarray
        :return: result: gradient of the cross-gradient with respect to model1, model2

        .. math::

            See tutorial or documentation.

        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1,m2])

        if self.regmesh.mesh.dim == 2:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)

            # common terms for dc_dm1
            Dx_m1, Dy_m1 = Dx.dot(m1), Dy.dot(m1)
            Dx_m2, Dy_m2 = Dx.dot(m2), Dy.dot(m2)
            v1 = Dx_m2**2 + Dy_m2**2
            v2 = Dx_m1**2 + Dy_m1**2
            w = Dx_m1*Dx_m2 + Dy_m1*Dy_m2

            dc_dm1 = (Dx.T.dot(Dx_m1*v1) + Dy.T.dot(Dy_m1*v1) -
                      Dx.T.dot(Dx_m2*w) - Dy.T.dot(Dy_m2*w))
            dc_dm2 = (Dx.T.dot(Dx_m2*v2) + Dy.T.dot(Dy_m2*v2) -
                     Dx.T.dot(Dx_m1*w) - Dy.T.dot(Dy_m1*w))
            result = np.concatenate((dc_dm1,dc_dm2))
        elif self.regmesh.mesh.dim == 3:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            Dz = self.regmesh.aveFz2CC.dot(self.regmesh.cellDiffz)

            # common terms for dc_dm1
            Dx_m1, Dy_m1, Dz_m1 = Dx.dot(m1), Dy.dot(m1), Dz.dot(m1)
            Dx_m2, Dy_m2, Dz_m2 = Dx.dot(m2), Dy.dot(m2), Dz.dot(m2)
            v1 = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            v2 = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            w = Dx_m1*Dx_m2 + Dy_m1*Dy_m2 + Dz_m1*Dz_m2

            dc_dm1 = (Dx.T.dot(Dx_m1*v1) + Dy.T.dot(Dy_m1*v1) + Dz.T.dot(Dz_m1*v1) -
                      Dx.T.dot(Dx_m2*w) - Dy.T.dot(Dy_m2*w) - Dz.T.dot(Dz_m2*w))
            dc_dm2 = (Dx.T.dot(Dx_m2*v2) + Dy.T.dot(Dy_m2*v2) + Dz.T.dot(Dz_m2*v2) -
                      Dx.T.dot(Dx_m1*w) - Dy.T.dot(Dy_m1*w) - Dz.T.dot(Dz_m1*w))
            result = np.concatenate((dc_dm1,dc_dm2))

        return result

    def hessian_offdiag(self, D, grad1, grad2):
        '''
        Computes the off-diagonals blocks of the Hessian of corss-gradient.

        :param tuple of scipy.sparse.csr_matrix D: (Dx, Dy, Dz) in 3D
                                                   (Dx, Dy) in 2D
        :param tuple of numpy.ndarray grad1: gradients for model 1
                                             (x_grad1, y_grad1, z_grad1)
        :param tuple of numpy.ndarray grad2: gradients for model2
                                             (x_grad2, y_grad2, z_grad2)

        :rtype: scipy.sparse.csr_matrix
        :return: off-diagonal term of Hessian matrix

        '''
        n = len(D)
        D_result = np.zeros_like(D[0])
        for i in range(n):
            for j in range(n):
                if j==i:
                    continue
                else:
                    D_result += 2*(D[i].T.dot(utils.sdiag(grad1[j]*grad2[i])).dot(D[j]))
                    D_result -= D[i].T.dot(utils.sdiag(grad1[j]*grad2[j])).dot(D[i])
                    D_result -= D[j].T.dot(utils.sdiag(grad1[j]*grad2[i])).dot(D[i])

        D_result = sp.csr_matrix(D_result)
        return D_result

    def deriv2(self, model, v=None):
        '''
        Computes the Hessian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2, ...]
        :param numpy.ndarray v: vector to be multiplied by Hessian

        :rtype: scipy.sparse.csr_matrix if v is None
                numpy.ndarray if v is not None
        :return Hessian matrix if v is None
                Hessian multiplied by vector if v is not None


        .. math::

            See tutorial or documentation.

        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1, m2])

        if v is not None:
            assert v.size == 2*m1.size, 'vector v must be of size 2*M'
            v1 = self.map1*v
            v2 = self.map2*v

        func1 = self.hessian_offdiag

        if self.regmesh.mesh.dim == 2:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            # define common terms
            Dx_m1, Dy_m1 = Dx.dot(m1), Dy.dot(m1)
            Dx_m2, Dy_m2 = Dx.dot(m2), Dy.dot(m2)
            a = Dx_m2**2 + Dy_m2**2
            b = Dx_m1**2 + Dy_m1**2
            A = Dx.T.dot(utils.sdiag(Dx_m2)) + Dy.T.dot(utils.sdiag(Dy_m2))
            B = Dx.T.dot(utils.sdiag(Dx_m1)) + Dy.T.dot(utils.sdiag(Dy_m1))

            d2c_dm1 = (Dx.T.dot(utils.sdiag(a)).dot(Dx) +
                       Dy.T.dot(utils.sdiag(a)).dot(Dy) - A.dot(A.T))
            d2c_dm2 = (Dx.T.dot(utils.sdiag(b)).dot(Dx) +
                       Dy.T.dot(utils.sdiag(b)).dot(Dy) - B.dot(B.T))

            d_dm2_dc_dm1 = func1((Dx, Dy), (Dx_m1, Dy_m1), (Dx_m2, Dy_m2))
            d_dm1_dc_dm2 = d_dm2_dc_dm1.T

        elif self.regmesh.mesh.dim == 3:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            Dz = self.regmesh.aveFz2CC.dot(self.regmesh.cellDiffz)
            # define common terms
            Dx_m1, Dy_m1, Dz_m1 = Dx.dot(m1), Dy.dot(m1), Dz.dot(m1)
            Dx_m2, Dy_m2, Dz_m2 = Dx.dot(m2), Dy.dot(m2), Dz.dot(m2)
            a = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            b = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            A = (Dx.T.dot(utils.sdiag(Dx_m2)) + Dy.T.dot(utils.sdiag(Dy_m2)) +
                 Dz.T.dot(utils.sdiag(Dz_m2)))
            B = (Dx.T.dot(utils.sdiag(Dx_m1)) + Dy.T.dot(utils.sdiag(Dy_m1)) +
                 Dz.T.dot(utils.sdiag(Dz_m1)))

            d2c_dm1 = (Dx.T.dot(utils.sdiag(a)).dot(Dx) +
                       Dy.T.dot(utils.sdiag(a)).dot(Dy) +
                       Dz.T.dot(utils.sdiag(a)).dot(Dz) - A.dot(A.T))
            d2c_dm2 = (Dx.T.dot(utils.sdiag(b)).dot(Dx) +
                       Dy.T.dot(utils.sdiag(b)).dot(Dy) +
                       Dz.T.dot(utils.sdiag(b)).dot(Dz) - B.dot(B.T))

            d_dm2_dc_dm1 = func1((Dx, Dy, Dz), (Dx_m1, Dy_m1, Dz_m1), (Dx_m2, Dy_m2, Dz_m2))
            d_dm1_dc_dm2 = d_dm2_dc_dm1.T

        if v is not None:
            d2c_dm1 = d2c_dm1.dot(v1)
            d2c_dm2 = d2c_dm2.dot(v2)
            d_dm2_dc_dm1 = d_dm2_dc_dm1.dot(v1)
            d_dm1_dc_dm2 = d_dm1_dc_dm2.dot(v2)
            result = np.concatenate((d2c_dm1 + d_dm1_dc_dm2, d_dm2_dc_dm1 + d2c_dm2))
        else:
            temp1 = sp.vstack((d2c_dm1,d_dm2_dc_dm1))
            temp2 = sp.vstack((d_dm1_dc_dm2, d2c_dm2))
            result = sp.hstack((temp1,temp2))
            result = sp.csr_matrix(result)

        return result


###############################################################################
#                                                                             #
#               Linear petrophysical relationship constraint                  #
#                                                                             #
###############################################################################

class Linear(BaseCoupling):
    '''
    The petrophysical linear constraint for joint inversions.

    ..math::
        \phi_c({\mathbf m}_{\mathbf1},{\mathbf m}_{\mathbf2})=\lambda\sum_{i=1}^M
        (k_1*m_1 + k_2*m_2 + k_3)

    Assuming that we are working with two models only.

    '''
    def __init__(self, mesh, indActive, mapping, **kwargs):
        self.as_super = super(Linear, self)
        self.as_super.__init__(mesh, indActive, mapping, **kwargs)
        self.map1, self.map2 = mapping.maps ### Assume a map has been passed for each model.


    def models(self, ind_models):
        '''
        Method to pass models to Joint Total Variation object and ensures models are compatible
        for further use. Checks that models are of same size.

        :param container of numpy.ndarray ind_models: [model1, model2,...]

        rtype: list of numpy.ndarray models: [model1, model2,...]
        return: models
        '''
        models = []
        n = len(ind_models) # number of individual models
        for i in range(n):
            ### check that the models are either a list, tuple, or np.ndarray
            assert isinstance(ind_models[i], (list,tuple,np.ndarray))
            if isinstance(ind_models[i], (list,tuple)):
                ind_models[i] = np.array(ind_models[i]) ### convert to np.ndarray

        ### check if models are of same size
        it = iter(ind_models)
        length = len(next(it))
        if not all(len(l)==length for l in it):
            raise ValueError('not all models are of the same size!')

        for i in range(n):
            models.append(ind_models[i])

        return models

    @property
    def coefficients(self):
        """
        :param list val: [k1, k2, k3]

        f(m1, m2)  = k1*m1 + k2*m2 + k3

        """
        if getattr(self, '_coefficients', None) is None:
            self._coefficients = [1, -1, 0]

        return self._coefficients

    @coefficients.setter
    def coefficients(self, val):
        """
        :param list val: [k1, k2, k3]

        f(m1, m2)  = k1*m1 + k2*m2 + k3

        """
        assert isinstance(val, (list))
        assert len(val)==3, 'length of coefficients must be 3!'

        self._coefficients = val


    def relation(self, model):
        """
        Computes the values of petrophysical linear relationship between two different
        geophysical models.

        The linear relationship is defined as:

        f(m1, m2)  = k1*m1 + k2*m2 + k3

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]

        :rtype: float
        :return: linearly related petrophysical values of two different models,
                  dimension: M by 1, :M number of model parameters.

        """

        m1 = self.map1*model
        m2 = self.map2*model
        k1 = self.coefficients[0]
        k2 = self.coefficients[1]
        k3 = self.coefficients[2]

        return k1*m1 + k2*m2 + k3


    def __call__(self, model):
        '''
        Computes the sum of values of petrophysical linear relationship
        between two different geophysical models.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]

        :rtype: float
        :return: a scalar value.
        '''

        result = np.linalg.norm(
            self.relation(model)
            )
        return result


    def deriv(self, model):
        '''
        Computes the Jacobian of the coupling term.

        :param list of numpy.ndarray ind_models: [model1, model2,...]

        :rtype: numpy.ndarray
        :return: result: gradient of the coupling term with respect to model1, model2,
                 :dimension 2M by 1, :M number of model parameters.
        '''

        k1 = self.coefficients[0]
        k2 = self.coefficients[1]

        dc_dm1 = 2 * k1 * self.relation(model)
        dc_dm2 = 2 * k2 * self.relation(model)

        result = np.concatenate((dc_dm1,dc_dm2))

        return result


    def deriv2(self, model, v=None):
        '''
        Computes the Hessian of the linear coupling term.

        :param list of numpy.ndarray ind_models: [model1, model2, ...]
        :param numpy.ndarray v: vector to be multiplied by Hessian
        :rtype: scipy.sparse.csr_matrix if v is None
                numpy.ndarray if v is not None
        :return Hessian matrix: | h11, h21 | :dimension 2M*2M.
                                |          |
                                | h12, h22 |

        '''

        k1 = self.coefficients[0]
        k2 = self.coefficients[1]

        m1 = self.map1*model
        m2 = self.map2*model
        n = m1.shape[0]

        if v is not None:
            assert v.size == 2*m1.size, 'vector v must be of size 2*M'
            v1 = self.map1*v
            v2 = self.map2*v

        d2c_dm1 = utils.sdiag(
            np.ones(n) * (2*k1**2)
            )

        d2c_dm2 = utils.sdiag(
            np.ones(n) * (2*k2**2)
            )

        d_dm1_dc_dm2 = utils.sdiag(
            np.ones(n) * (2*k2*k1)
            )

        d_dm2_dc_dm1 = d_dm1_dc_dm2


        if v is not None:
            d2c_dm1 = d2c_dm1.dot(v1)
            d2c_dm2 = d2c_dm2.dot(v2)
            d_dm2_dc_dm1 = d_dm2_dc_dm1.dot(v1)
            d_dm1_dc_dm2 = d_dm1_dc_dm2.dot(v2)
            result = np.concatenate((d2c_dm1 + d_dm1_dc_dm2, d_dm2_dc_dm1 + d2c_dm2))
        else:
            temp1 = sp.vstack((d2c_dm1,d_dm2_dc_dm1))
            temp2 = sp.vstack((d_dm1_dc_dm2, d2c_dm2))
            result = sp.hstack((temp1,temp2))
            result = sp.csr_matrix(result)


        return result
