import numpy as np
from scipy.sparse import csr_matrix

from .. import Utils
from .base import BaseRegularization


class BaseCoupling(BaseRegularization):

    '''
    BaseCoupling
    
        ..note::
            
            You should inherit from this class to create your own
            coupling term.
    '''
    def __init__(self, mesh, actvMap, mapping, **kwargs):
        
        self.as_super = super(BaseCoupling, self)
        self.as_super.__init__()
        self.mesh = mesh
        self.actvMap = actvMap # Map active cells.
        self.mapping = mapping # We expect Wire Maps
        
    def deriv(self):
        ''' 
        First derivative of the coupling term with respect to individual models.
        Returns an array of dimensions [k*M,1],
        k: number of models we are inverting for.
        M: number of model cells. 
        '''
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def deriv2(self):
        ''' 
        Second derivative of the coupling term with respect to individual models.
        Returns either an array of dimensions [k*M,1], or
        sparse matrix of dimensions [k*M, k*M]
        k: number of models we are inverting for.
        M: number of model cells.
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
#                               GradientCoupling                              #
#                                                                             #
###############################################################################

class GradientCoupling(BaseCoupling):
    '''
    ***GradientCoupling***
    
    This class contains methods needed for gradient-based structural couplings.
    
    '''
    def __init__(self, mesh, actvMap, mapping, **kwargs):
        
        self.as_super = super(GradientCoupling, self)
        self.as_super.__init__(mesh, actvMap, mapping, **kwargs)
        
        
    def models(self, ind_models):
        '''
        Method to pass models to CrossGradient object and makes sure models are compatible
        for further use. Checks that models are of same size and maps from active-space
        to full-space.
        
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

        ### check if models have already been mapped to full-space. It's easier
        ### to work in full-space when working with spatial gradients.
        if self.actvMap.nC == ind_models[0].size:
            for i in range(n):
                models.append(ind_models[i])
        else:
            for i in range(n):
                if ind_models[i].ndim > 1:
                    ind_models[i] = Utils.mkvc(ind_models[i]) # flatten models
                    models.append(self.actvMap._transform(ind_models[i]))
                else:
                    models.append(self.actvMap._transform(ind_models[i]))
        return models

    def models_as_arrays(self, ind_models):
        '''
        Returns input models reshaped into 2D or 3D.
        Working in 2D or 3D makes it easier to compute the spatial gradients.
        Assume m1 and m2 are numpy arrays.
        
        :param container of numpy.ndarray ind_models: [model1, model2,...]
        
        :rtype: list of of numpy.ndarray
        :return: models: [model1, model2, ...] 
                 where model1, model2, ... are 2D or 3D
        '''
        models = []
        n = len(ind_models) # number of individual models
        ### check if models have been mapped to full-space.
        if self.actvMap.nC != ind_models[0].size:
            print('models have not been mapped to full-space')
            print('mapping models to full-space')
            ind_models = self.models(ind_models)
        
        if self.mesh.dim == 2:
            x_dim = self.mesh.nCx
            y_dim = self.mesh.nCy
            for i in range(n):
                models.append(ind_models[i].reshape([x_dim, y_dim], order='F'))
        elif self.mesh.dim == 3:
            x_dim = self.mesh.nCx
            y_dim = self.mesh.nCy
            z_dim = self.mesh.nCz
            for i in range(n):
                models.append(ind_models[i].reshape([x_dim, y_dim, z_dim], order='F'))
        return models
    
    def _remove_nan(self, model, grad, dir_vec, dim_ind, indactive, d):
        '''
        Method used to remove unwanted np.nan in gradient.
        
        :param numpy.ndarray model: model
        :param numpy.ndarray grad: gradient array
        :param numpy.ndarray dir_vec: standard basis vector in x,y, or z direction
        :param numpy.ndarray dim_ind: index of dimension (0:x, 1:y, 2:z)
        :param bool numpy.ndarray indactive: array of active indices
        :param numpy.ndarray d: cell centers
        
        :rtype: numpy.ndarray
        :return: grad: gradient where unwanted np.nan have been removed
                       by replacing with forward/backward difference
        '''
        nan_ind = list(map(tuple, np.argwhere(np.isnan(grad))))
        for ind in nan_ind:
            ### We assume that unwanted np.nan occur adjacent to the topo.
            ### Replace with forward/backward difference.
            if indactive[ind]==True:
                d_ind = ind[dim_ind]
                if indactive[tuple(np.array(ind) + dir_vec)]==True:
                    grad[ind] = ((model[tuple(np.array(ind) + dir_vec)] - model[ind])
                                /(d[d_ind+1] - d[d_ind]))
                elif indactive[tuple(np.array(ind) - dir_vec)]==True:
                    grad[ind] = ((model[ind] - model[tuple(np.array(ind) - dir_vec)])
                                /(d[d_ind] - d[d_ind-1]))
                else:
                    continue
            else:
                continue
        return grad
                
    def calculate_gradient(self, model, reduced_space=False):
        '''
        Calculate the spatial gradients of the model using central difference
        and forward/backward difference on edges.
        Assume that input model is 2D or 3D numpy array.
        
        :param numpy.ndarray model: model
        :param bool vectorize: return gradients in 1D if True
        :param bool noNaN: return gradients without any numpy.nan caused by actvMap if True.
        
        :rtype: tuple of numpy.ndarray
        :return: (x-gradient, y-gradient) if 2D
                 (x-gradient, y-gradient, z-gradient) if 3D
        '''        
        indactive = self.actvMap.indActive
        if self.mesh.dim == 2:
            dx = self.mesh.vectorCCx
            dy = self.mesh.vectorCCy
            indactive = indactive.reshape((self.mesh.nCx, self.mesh.nCy), order='F')
            x_grad, y_grad = np.gradient(model, dx, dy)

            ### Check for np.nan's in gradients
            x_dir = np.array([1,0])
            y_dir = np.array([0,1])
            x_grad = self._remove_nan(model, x_grad, x_dir, 0, indactive, dx)
            y_grad = self._remove_nan(model, y_grad, y_dir, 1, indactive, dy)
            x_grad = Utils.mkvc(x_grad)
            y_grad = Utils.mkvc(y_grad)
        
            if reduced_space:
                indactive = self.actvMap.indActive
                x_grad = x_grad[indactive]
                y_grad = y_grad[indactive]
                
            return x_grad, y_grad
        
        elif self.mesh.dim == 3:
            dx = self.mesh.vectorCCx
            dy = self.mesh.vectorCCy
            dz = self.mesh.vectorCCz
            indactive = indactive.reshape([self.mesh.nCx, self.mesh.nCy, self.mesh.nCz], order='F')
            x_grad, y_grad, z_grad = np.gradient(model, dx, dy, dz)

            ### Check for np.nan's in gradients
            x_dir = np.array([1,0,0])
            y_dir = np.array([0,1,0])
            z_dir = np.array([0,0,1])
            x_grad = self._remove_nan(model, x_grad, x_dir, 0, indactive, dx)
            y_grad = self._remove_nan(model, y_grad, y_dir, 1, indactive, dy)
            z_grad = self._remove_nan(model, z_grad, z_dir, 2, indactive, dz)
            x_grad = Utils.mkvc(x_grad)
            y_grad = Utils.mkvc(y_grad)
            z_grad = Utils.mkvc(z_grad)
                
            if reduced_space:
                indactive = self.actvMap.indActive
                x_grad = x_grad[indactive]
                y_grad = y_grad[indactive]
                z_grad = z_grad[indactive]
                
            return x_grad, y_grad, z_grad

    def _build_D(self, Dx, i, r, width, dim, num_dims, indactive, j, k=None):
        '''
        Method for internal use only.
        Returns a difference matrix for central-difference method.
        
        :param numpy.ndarray Dx: difference matrix in x-direction from previous iteration.
        :param int i: row index
        :param int r: current index of difference matrix
        :param numpy.ndarray width: widths of model cells in x-direction.
        :param int dim: size of dimension in x-direction.
        :param int num_dims: number of dimensions in mesh
        :param int j: column index
        
        :rtype: numpy.ndarray
        :returns: Difference matrix in x-direction
        '''
        if num_dims == 2:
            if i==0:
                d = width[0]
                Dx[r,r] = -1/d
                Dx[r,r+1] = 1/d
            elif i==dim-1:
                d = width[-1]
                Dx[r,r] = 1/d
                Dx[r,r-1] = -1/d
            else:
                ### check if adjacent cells are inactive. 
                ### If so, use forward/backward difference.
                if indactive[i+1,j]==False:
                    d = width[i-1]
                    Dx[r,r] = 1/d
                    Dx[r,r-1] = -1/d
                elif indactive[i-1,j]==False:
                    d = width[i]
                    Dx[r,r] = -1/d
                    Dx[r,r+1] = 1/d
                else:
                    d = width[i-1]*width[i]*(width[i]+width[i-1])
                    Dx[r,r-1] = -width[i]**2/d
                    Dx[r,r] = (width[i]**2-width[i-1]**2)/d
                    Dx[r,r+1] = width[i-1]**2/d
        else:
            indactive = indactive.reshape([self.mesh.nCx, self.mesh.nCy, self.mesh.nCz], order='F')
            if i==0:
                d = width[0]
                Dx[r,r] = -1/d
                Dx[r,r+1] = 1/d
            elif i==dim-1:
                d = width[-1]
                Dx[r,r] = 1/d
                Dx[r,r-1] = -1/d
            else:
                ### check if adjacent cells are inactive. 
                ### If so, use forward/backward difference.
                if indactive[i+1,j,k]==False:
                    d = width[i-1]
                    Dx[r,r] = 1/d
                    Dx[r,r-1] = -1/d
                elif indactive[i-1,j,k]==False:
                    d = width[i]
                    Dx[r,r] = -1/d
                    Dx[r,r+1] = 1/d
                else:
                    d = width[i-1]*width[i]*(width[i]+width[i-1])
                    Dx[r,r-1] = -width[i]**2/d
                    Dx[r,r] = (width[i]**2-width[i-1]**2)/d
                    Dx[r,r+1] = width[i-1]**2/d
        return Dx

    def _build_Dy(self,Dy, j, r, width, dim_1, dim_2, indactive, i, k=None):
        '''
        Method for internal use only.
        Returns a difference matrix for central-difference method used in 3D.
        
        :param numpy.ndarray Dy: difference matrix in y-direction from previous iteration.
        :param int j: column index
        :param int r: current index of difference matrix
        :param numpy.ndarray width: widths of model cells in y-direction
        :param int dim_1: size of dimension in x-direction.
        :param int dim_2: size of dimension in y-direction.
        :param int num_dims: number of dimensions in mesh
        :param int i: row index
        
        :rtype: numpy.ndarray
        :returns: Difference matrix in y-direction
        '''
        if j==0:
            d = width[0]
            Dy[r,r] = -1/d
            Dy[r,r+dim_1] = 1/d
        elif j==dim_2-1:
            d = width[-1]
            Dy[r,r] = 1/d
            Dy[r,r-dim_1] = -1/d
        else:
            ### check if adjacent cells are inactive. 
            ### If so, use forward/backward difference.
            if indactive[i,j+1,k]==False:
                d = width[j-1]
                Dy[r,r] = 1/d
                Dy[r,r-dim_1] = -1/d
            elif indactive[i,j-1,k]==False:
                d = width[j]
                Dy[r,r] = -1/d
                Dy[r,r+dim_1] = 1/d
            else:
                d = width[j-1]*width[j]*(width[j]+width[j-1])
                Dy[r,r-dim_1] = -width[j]**2/d
                Dy[r,r] = (width[j]**2 - width[j-1]**2)/d
                Dy[r,r+dim_1] = width[j-1]**2/d
        return Dy

    def build_difference_matrix(self, reduced_space=False):
        '''
        Builds difference matrices for central-difference method used in calculating
        the gradients of an array.
        
        :rtype: tuple of scipy.sparse.csr_matrix
        :return: (Dx,Dy): difference matrices in x- and y-directions if 2D
                 (Dx, Dy, Dz): difference matrices in x-, y-, z-directions if 3D
        '''
        indactive = self.actvMap.indActive
        dims = self.mesh.dim
        if dims == 2:
            width_x = np.diff(self.mesh.vectorCCx)
            width_y = np.diff(self.mesh.vectorCCy)
            indactive = indactive.reshape([self.mesh.nCx, self.mesh.nCy], order='F')
            dim_1, dim_2 = self.mesh.nCx, self.mesh.nCy
            n = self.mesh.nC
            Dx = np.zeros([n,n])
            Dy = np.zeros([n,n])
            r = 0
            for j in range(dim_2):
                for i in range(dim_1):
                    if j==0:
                        Dx = self._build_D(Dx, i, r, width_x, dim_1, dims, indactive, j)
                        d = width_y[0]
                        Dy[r,r] = -1/d
                        Dy[r,r+dim_1] = 1/d
                        r += 1
                    elif j==dim_2-1:
                        Dx = self._build_D(Dx, i, r, width_x, dim_1, dims, indactive, j)
                        d = width_y[-1]
                        Dy[r,r-dim_1] = -1/d
                        Dy[r,r] = 1/d
                        r += 1
                    else:
                        Dx = self._build_D(Dx, i, r, width_x, dim_1, dims, indactive, j)
                        ### check if adjacent cells are inactive. 
                        ### If so, use forward/backward difference.
                        if indactive[i,j+1]==False:
                            d = width_y[j-1]
                            Dy[r,r-dim_1] = -1/d
                            Dy[r,r] = 1/d
                        elif indactive[i,j-1]==False:
                            d = width_y[j]
                            Dy[r,r] = -1/d
                            Dy[r,r+dim_1] = 1/d
                        else:
                            d = width_y[j-1]*width_y[j]*(width_y[j]+width_y[j-1])
                            Dy[r,r-dim_1] = -width_y[j]**2/d
                            Dy[r,r] = (width_y[j]**2 - width_y[j-1]**2)/d
                            Dy[r,r+dim_1] = width_y[j-1]**2/d
                        r += 1
            Dx = csr_matrix(Dx)
            Dy = csr_matrix(Dy)
            
            if reduced_space:
                indactive = self.actvMap.indActive
                Dx = Dx[indactive]
                Dx = Dx[:,indactive]
                Dy = Dy[indactive]
                Dy = Dy[:,indactive]
                
            return Dx, Dy
        
        elif dims == 3:
            width_x = np.diff(self.mesh.vectorCCx)
            width_y = np.diff(self.mesh.vectorCCy)
            width_z = np.diff(self.mesh.vectorCCz)
            indactive = indactive.reshape([self.mesh.nCx, self.mesh.nCy, self.mesh.nCz], order='F')
            dim_1, dim_2, dim_3 = self.mesh.nCx, self.mesh.nCy, self.mesh.nCz
            n = self.mesh.nC
            Dx = np.zeros([n,n])
            Dy = np.zeros([n,n])
            Dz = np.zeros([n,n])
            r = 0
            for k in range(dim_3):
                for j in range(dim_2):
                    for i in range(dim_1):
                        if k==0:
                            Dx = self._build_D(Dx, i, r, width_x, dim_1, dims, indactive, j, k)
                            Dy = self._build_Dy(Dy, j, r, width_y, dim_1, dim_2, indactive, i, k)
                            d = width_z[0]
                            Dz[r,r] = -1/d
                            Dz[r,r+dim_1*dim_2] = 1/d
                            r += 1
                        elif k==dim_3-1:
                            Dx = self._build_D(Dx, i, r, width_x, dim_1, dims, indactive, j, k)
                            Dy = self._build_Dy(Dy, j, r, width_y, dim_1, dim_2, indactive, i, k)
                            d = width_z[-1]
                            Dz[r,r] = 1/d
                            Dz[r,r-dim_1*dim_2] = -1/d
                            r += 1
                        else:
                            Dx = self._build_D(Dx, i, r, width_x, dim_1, dims, indactive, j, k)
                            Dy = self._build_Dy(Dy, j, r, width_y, dim_1, dim_2, indactive, i, k)
                            ### check if adjacent cells are inactive. 
                            ### If so, use forward/backward difference.
                            if indactive[i,j,k+1]==False:
                                d = width_z[k-1]
                                Dz[r,r] = 1/d
                                Dz[r,r-dim_1*dim_2] = -1/d
                            elif indactive[i,j,k-1]==False:
                                d = width_z[k]
                                Dz[r,r] = 1/d
                                Dz[r,r+dim_1*dim_2] = -1/d
                            else:
                                d = width_z[k-1]*width_z[k]*(width_z[k]+width_z[k-1])
                                Dz[r,r-dim_1*dim_2] = -width_z[k]**2/d
                                Dz[r,r] = (width_z[k]**2 - width_z[k-1]**2)/d
                                Dz[r,r+dim_1*dim_2] = width_z[k-1]**2/d
                            r += 1
            Dx = csr_matrix(Dx)
            Dy = csr_matrix(Dy)
            Dz = csr_matrix(Dz)
            
            if reduced_space:
                indactive = self.actvMap.indActive
                Dx = Dx[indactive]
                Dx = Dx[:,indactive]
                Dy = Dy[indactive]
                Dy = Dy[:,indactive]
                Dz = Dz[indactive]
                Dz = Dz[:,indactive]
                
            return Dx, Dy, Dz

    def gradient_list(self, x_grad, y_grad, z_grad=np.array([])):
        '''
        Concatenates gradient components into a single array.
        [[x_grad1, y_grad1, z_grad1],
         [x_grad2, y_grad2, z_grad2],
         [x_grad3, y_grad3, z_grad3],...]
        
        :param numpy.ndarray x_grad: x-gradients
        :param numpy.ndarray y_grad: y-gradients
        :param numpy.ndarray z_grad: z-gradients
        
        :rtype: numpy.ndarray
        :return: gradient_vector: array where each row represents a model cell, 
                 and each column represents a component of the gradient.
        '''
        if z_grad.size == 0:
            gradient_vector = np.c_[Utils.mkvc(x_grad), Utils.mkvc(y_grad)]
        else:
            gradient_vector = np.c_[Utils.mkvc(x_grad), Utils.mkvc(y_grad), Utils.mkvc(z_grad)]
            
        return gradient_vector
        
    def construct_norm_vectors(self, m1, m2, reduced_space=False, fltr=True, fltr_per=0.05):
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
                norms_1: np.array containing reciprocal of the gradient norms for model1
                norms_2: np.array containing reciprocal of the gradient norms for model2
                tot_norms: np.array containing element-wise product of norms_1 and norms_2.
        '''
        model1, model2 = self.models_as_arrays([m1, m2])
        per = int(fltr_per*self.actvMap.nP)
        
        if self.mesh.dim == 2:
            Dx, Dy = self.build_difference_matrix()
            Dx_m1, Dy_m1 = self.calculate_gradient(model1, reduced_space=reduced_space)
            Dx_m2, Dy_m2 = self.calculate_gradient(model2, reduced_space=reduced_space)
            
            norms_1 = np.sqrt(Dx_m1**2 + Dy_m1**2)
            norms_1 = np.divide(1, norms_1, out=np.zeros_like(norms_1), where=norms_1>1e-10)
            # set lowest 5% of norms (largest 5% of 1/norms) to 0.0
            if fltr:
                inds1 = norms_1.argsort()[-per:]
                norms_1[inds1] = 0.0
            
            norms_2 = np.sqrt(Dx_m2**2 + Dy_m2**2)
            norms_2 = np.divide(1, norms_2, out=np.zeros_like(norms_2), where=norms_2>1e-10)
            # set lowest 5% of norms (largest 5% of 1/norms) to 0.0
            if fltr:
                inds2 = norms_2.argsort()[-per:]
                norms_2[inds2] = 0.0
            tot_norms = norms_1*norms_2
            
        elif self.mesh.dim == 3:
            Dx, Dy, Dz = self.build_difference_matrix()
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(model1, reduced_space=reduced_space)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(model2, reduced_space=reduced_space)
            
            norms_1 = np.sqrt(Dx_m1**2 + Dy_m1**2 + Dz_m1**2)
            norms_1 = np.divide(1, norms_1, out=np.zeros_like(norms_1), where=norms_1>1e-10)
            # set lowest 5% of norms (largest 5% of 1/norms) to 0.0
            if fltr:
                inds1 = norms_1.argsort()[-per:]
                norms_1[inds1] = 0.0
            
            norms_2 = np.sqrt(Dx_m2**2 + Dy_m2**2 + Dz_m2**2)
            norms_2 = np.divide(1, norms_2, out=np.zeros_like(norms_2), where=norms_2>1e-10)
            # set lowest 5% of norms (largest 5% of 1/norms) to 0.0
            if fltr:
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
            
###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################

class CrossGradient(GradientCoupling):
    '''
    The cross-gradient constraint for joint inversions.
    
    ..math::
        \phi_c(\mathbf{m_1},\mathbf{m_2}) = \lambda \sum_{i=1}^{M} \| 
        \nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2

    All methods assume that we are working with two models only.
    '''
    def __init__(self, mesh, actvMap, mapping, **kwargs):
        # For coupling using spatial gradients, assume non-active cells get value np.nan
        # Else, if we assign a value to the non-active cells, the gradients will be computed
        # using these values and will give incorrect gradients.
        self.as_super = super(CrossGradient, self)
        self.as_super.__init__(mesh, actvMap, mapping, **kwargs)
        self.map1, self.map2 = mapping.maps ### Assume a map has been passed for each model.
        
        assert self.mesh.dim in (2,3), 'Cross-Gradient is only defined for 2D or 3D'

    def calculate_cross_gradient(self, m1, m2, normalized=False):
        '''
        Calculates the cross-gradients of two models.
        
        :param numpy.ndarray m1: model1
        :param numpy.ndarray m2: model2
        :param bool normalized: normalizes gradients if True
        
        :rtype: numpy.ndarray
        :returns: array where at each location, we've computed the cross-product
                  of the gradients of two models.
        '''
        model1, model2 = self.models([m1,m2])
        
        ### Compute the gradients and concatenate components.
        if self.mesh.dim == 3:
            array_1, array_2 = self.models_as_arrays([model1, model2])
            x_grad_1, y_grad_1, z_grad_1 = self.calculate_gradient(array_1)
            x_grad_2, y_grad_2, z_grad_2 = self.calculate_gradient(array_2)
            
            grad_list_1 = self.gradient_list(x_grad_1, y_grad_1, z_grad_1)
            grad_list_2 = self.gradient_list(x_grad_2, y_grad_2, z_grad_2)
            
            if normalized:
                grad_list_1 = self.normalized_gradients(grad_list_1)
                grad_list_2 = self.normalized_gradients(grad_list_2)

        elif self.mesh.dim == 2:
            array_1, array_2 = self.models_as_arrays([model1, model2])
            x_grad_1, y_grad_1 = self.calculate_gradient(array_1)
            x_grad_2, y_grad_2 = self.calculate_gradient(array_2)

            grad_list_1 = self.gradient_list(x_grad_1, y_grad_1)
            grad_list_2 = self.gradient_list(x_grad_2, y_grad_2)

            if normalized:
                grad_list_1 = self.normalized_gradients(grad_list_1)
                grad_list_2 = self.normalized_gradients(grad_list_2)
                
        cross_prod_list = []
        num_cells = self.mesh.nC
        ### for each model cell, compute the cross product of the gradient vectors.
        for x in range(num_cells):
            if self.mesh.dim == 3:
                cross_prod_vector = np.cross(grad_list_1[x],grad_list_2[x])
                cross_prod = np.linalg.norm(cross_prod_vector)
            else:
                cross_prod = np.cross(grad_list_1[x],grad_list_2[x])
            cross_prod_list.append(cross_prod)
        cross_prod = np.array(cross_prod_list)
        
        return cross_prod

    def __call__(self, model, normalized=False):
        '''
        :param list of numpy.ndarray ind_models: [model1, model2,...]
        :param bool normalized: returns value of normalized cross-gradient if True
        
        :rtype: float
        :returns: the computed value of the cross-gradient term.
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        model1, model2 = self.models([m1,m2])
        
        if self.mesh.dim == 3:
            array_1, array_2 = self.models_as_arrays([model1, model2])
            x_grad_1, y_grad_1, z_grad_1 = self.calculate_gradient(array_1, reduced_space=True)
            x_grad_2, y_grad_2, z_grad_2 = self.calculate_gradient(array_2, reduced_space=True)
                
            temp1 = x_grad_1**2 + y_grad_1**2 + z_grad_1**2
            temp2 = x_grad_2**2 + y_grad_2**2 + z_grad_2**2
            term1 = temp1.dot(temp2)
            
            temp1 = x_grad_1*x_grad_2 + y_grad_1*y_grad_2 + z_grad_1*z_grad_2
            term2 = (np.linalg.norm(temp1))**2
            result = term1 - term2
            
            if normalized:
                norms1, norms2, norms = self.construct_norm_vectors(model1, model2, 
                                                    reduced_space=True, fltr=False)
                temp1 = (x_grad_1*norms1)**2 + (y_grad_1*norms1)**2 + (z_grad_1*norms1)**2
                temp2 = (x_grad_2*norms2)**2 + (y_grad_2*norms2)**2 + (z_grad_2*norms2)**2
                term1 = temp1.dot(temp2)

                temp1 = x_grad_1*x_grad_2*norms + y_grad_1*y_grad_2*norms + z_grad_1*z_grad_2*norms
                term2 = (np.linalg.norm(temp1))**2
                result = term1 - term2
                
        elif self.mesh.dim == 2:
            array_1, array_2 = self.models_as_arrays([model1, model2])
            x_grad_1, y_grad_1 = self.calculate_gradient(array_1, reduced_space=True)
            x_grad_2, y_grad_2 = self.calculate_gradient(array_2, reduced_space=True)
                
            temp1 = x_grad_1**2 + y_grad_1**2
            temp2 = x_grad_2**2 + y_grad_2**2
            term1 = temp1.dot(temp2)
            
            temp1 = x_grad_1*x_grad_2 + y_grad_1*y_grad_2
            term2 = (np.linalg.norm(temp1))**2
            result = term1 - term2
            
            if normalized:
                norms1, norms2, norms = self.construct_norm_vectors(model1, model2, 
                                                    reduced_space=True, fltr=False)
                temp1 = (x_grad_1*norms1)**2 + (y_grad_1*norms1)**2
                temp2 = (x_grad_2*norms2)**2 + (y_grad_2*norms2)**2
                term1 = temp1.dot(temp2)
                
                temp1 = x_grad_1*x_grad_2*norms + y_grad_1*y_grad_2*norms
                term2 = (np.linalg.norm(temp1))**2
                result = term1 - term2
                
        return 0.5*result
       
    def _func_jacobian1(self, D, v1, v2):
        '''
        Method for internal use only.
        Used for computing the Jacobian of the cross-gradient.
        Computes D.dot(v1*v2).
    
        :param scipy.sparse.csr_matrix: D: difference matrix
        :param numpy.ndarray v1: 1D array
        :param numpy.ndarray v2: 1D array
        
        :rtype: scipy.sparse.csr_matrix
        :return: result = D.dot(v1*v2)
        '''
        temp = v1*v2
        result = D.dot(temp)
        
        return result
    
    def deriv(self, model):
        '''
        Computes the Jacobian of the cross-gradient.
        
        :param list of numpy.ndarray ind_models: [model1, model2,...]
        
        :rtype: numpy.ndarray
        :return: result: gradient of the cross-gradient with respect to model1, model2
        '''
        ### For the derivatives, we work on the active-space.
        m1 = self.map1*model
        m2 = self.map2*model

        model1, model2 = self.models_as_arrays([m1, m2])

        func1 = self._func_jacobian1

        if self.mesh.dim == 2:
            Dx, Dy = self.build_difference_matrix(reduced_space=True)
            
            ### common terms for dt_dm1
            Dx_m1, Dy_m1 = self.calculate_gradient(model1, reduced_space=True)
            Dx_m2, Dy_m2 = self.calculate_gradient(model2, reduced_space=True)
            v1 = Dx_m2**2 + Dy_m2**2
            v2 = Dx_m1**2 + Dy_m1**2
            w = Dx_m1*Dx_m2 + Dy_m1*Dy_m2
            
            dt_dm1 = (func1(Dx.T, Dx_m1, v1) + func1(Dy.T, Dy_m1, v1) - 
                      func1(Dx.T, Dx_m2, w) - func1(Dy.T, Dy_m2, w))
            dt_dm2 = (func1(Dx.T, Dx_m2, v2) + func1(Dy.T, Dy_m2, v2) - 
                     func1(Dx.T, Dx_m1, w) - func1(Dy.T, Dy_m1, w))
            result = np.concatenate((dt_dm1,dt_dm2))
        elif self.mesh.dim == 3:
            Dx, Dy, Dz = self.build_difference_matrix(reduced_space=True)
            
            ### common terms for dt_dm1
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(model1, reduced_space=True)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(model2, reduced_space=True)
            v1 = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            v2 = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            w = Dx_m1*Dx_m2 + Dy_m1*Dy_m2 + Dz_m1*Dz_m2
            
            dt_dm1 = (func1(Dx.T, Dx_m1, v1) + func1(Dy.T, Dy_m1, v1) + func1(Dz.T, Dz_m1, v1) - 
                      func1(Dx.T, Dx_m2, w) - func1(Dy.T, Dy_m2, w) - func1(Dz.T, Dz_m2, w))
            dt_dm2 = (func1(Dx.T, Dx_m2, v2) + func1(Dy.T, Dy_m2, v2) + func1(Dz.T, Dz_m2, v2) -
                      func1(Dx.T, Dx_m1, w) - func1(Dy.T, Dy_m1, w) - func1(Dz.T, Dz_m1, w))
            result = np.concatenate((dt_dm1,dt_dm2))

        return result
    
    def _func_hessian1(self, D1, D2, *args):
        '''
        Method for internal use only.
        Used for computing the Hessian of the normalized cross-gradient.
        Computes D1.dot(Utils.sdiag(*args)).dot(D2)
        
        :param scipy.sparse.csr_matrix: D1: sparse matrix
        :param numpy.ndarray *args: vectors
        :param scipy.sparse.csr_matrix: D2: sparse matrix
        
        :rtype: scipy.sparse.csr_matrix
        :return: result = D1.dot(Utils.sdiag(*args)).dot(D2)
        '''    
        count = 0
        for vec in args:
            if count == 0:
                temp = vec.copy()
            else:
                temp *= vec.copy()
            count += 1
        
        temp = Utils.sdiag(temp)
        temp = temp.dot(D2)
        result = D1.dot(temp)
        result = csr_matrix(result)
        
        return result
    
    def _func_hessian2(self, D, v):
        '''
        Method for internal use only.
        Used for computing the Hessian of the cross-gradient.
        Computes D.dot(Utils.sdiag(v)).
        
        :param scipy.sparse.csr_matrix D: difference matrix
        :param numpy.ndarray v1: 1D array
        
        :rtype: scipy.sparse.csr_matrix
        :return: result = D.dot(Utils.sdiag(v))
        '''        
        temp = Utils.sdiag(v)
        result = D.dot(temp)
        result = csr_matrix(result)
        
        return result
    
    def _func_hessian3(self, D, grad1, grad2):
        '''
        Method for internal use only.
        Used for computing the Hessian of the cross-gradient.
        Computes the off-diagonals of the Hessian.
        
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
        func1 = self._func_hessian1
        D_result = np.zeros_like(D[0])
        for i in range(n):
            for j in range(n):
                if j==i:
                    continue
                else:
                    D_result += 2*func1(D[i].T, D[j], grad1[j], grad2[i])
                    D_result -= func1(D[i].T, D[i], grad1[j], grad2[j])
                    D_result -= func1(D[j].T, D[i], grad1[j], grad2[i])
        
        D_result = csr_matrix(D_result)
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
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        
        if v is not None:
            assert v.size == 2*m1.size, 'vector v must be of size 2*M'
            v1 = self.map1*v
            v2 = self.map2*v
            
        model1, model2 = self.models_as_arrays([m1, m2])
        
        func1 = self._func_hessian1
        func2 = self._func_hessian2
        func3 = self._func_hessian3

        if self.mesh.dim == 2:
            Dx, Dy = self.build_difference_matrix(reduced_space=True)
            ### define common terms
            Dx_m1, Dy_m1 = self.calculate_gradient(model1, reduced_space=True)
            Dx_m2, Dy_m2 = self.calculate_gradient(model2, reduced_space=True)
            a = Dx_m2**2 + Dy_m2**2
            b = Dx_m1**2 + Dy_m1**2
            A = func2(Dx.T, Dx_m2) + func2(Dy.T, Dy_m2)
            B = func2(Dx.T, Dx_m1) + func2(Dy.T, Dy_m1)
            
            d2t_dm1 = func1(Dx.T, Dx, a) + func1(Dy.T, Dy, a) - A.dot(A.T)
            d2t_dm2 = func1(Dx.T, Dx, b) + func1(Dy.T, Dy, b) - B.dot(B.T)
            d_dm2_dt_dm1 = func3((Dx, Dy), (Dx_m1, Dy_m1), (Dx_m2, Dy_m2))
            d_dm1_dt_dm2 = d_dm2_dt_dm1.T

            if v is not None:
                d2t_dm1 = d2t_dm1.dot(v1)
                d2t_dm2 = d2t_dm2.dot(v2)
                d_dm2_dt_dm1 = d_dm2_dt_dm1.dot(v1)
                d_dm1_dt_dm2 = d_dm1_dt_dm2.dot(v2)
                result = np.concatenate((d2t_dm1 + d_dm1_dt_dm2, d_dm2_dt_dm1 + d2t_dm2))
            else:
                temp1 = np.concatenate((d2t_dm1.A,d_dm2_dt_dm1.A))
                temp2 = np.concatenate((d_dm1_dt_dm2.A, d2t_dm2.A))
                result = np.concatenate((temp1,temp2), axis=1)
                result = csr_matrix(result)
        
        elif self.mesh.dim == 3:
            Dx, Dy, Dz = self.build_difference_matrix(reduced_space=True)
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(model1, reduced_space=True)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(model2, reduced_space=True)
            a = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            b = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            A = func2(Dx.T, Dx_m2) + func2(Dy.T, Dy_m2) + func2(Dz.T, Dz_m2)
            B = func2(Dx.T, Dx_m1) + func2(Dy.T, Dy_m1) + func2(Dz.T, Dz_m1)
            
            d2t_dm1 = func1(Dx.T, Dx, a) + func1(Dy.T, Dy, a) + func1(Dz.T, Dz, a) - A.dot(A.T)
            d2t_dm2 = func1(Dx.T, Dx, b) + func1(Dy.T, Dy, b) + func1(Dz.T, Dz, b) - B.dot(B.T)
            d_dm2_dt_dm1 = func3((Dx, Dy, Dz), (Dx_m1, Dy_m1, Dz_m1), (Dx_m2, Dy_m2, Dz_m2))
            d_dm1_dt_dm2 = d_dm2_dt_dm1.T
            
            if v is not None:
                d2t_dm1 = d2t_dm1.dot(v1)
                d2t_dm2 = d2t_dm2.dot(v2)
                d_dm2_dt_dm1 = d_dm2_dt_dm1.dot(v1)
                d_dm1_dt_dm2 = d_dm1_dt_dm2.dot(v2)
                result = np.concatenate((d2t_dm1 + d_dm1_dt_dm2, d_dm2_dt_dm1 + d2t_dm2))
            else:
                temp1 = np.concatenate((d2t_dm1.A,d_dm2_dt_dm1.A))
                temp2 = np.concatenate((d_dm1_dt_dm2.A, d2t_dm2.A))
                result = np.concatenate((temp1,temp2), axis=1)
                result = csr_matrix(result)

        return result
