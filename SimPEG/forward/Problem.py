import numpy as np
from SimPEG.utils import mkvc, sdiag
norm = np.linalg.norm


class Problem(object):
    """
        Problem is the base class for all geophysical forward problems in SimPEG.


        The problem is a partial differential equation of the form:

        .. math::
            c(m, u) = 0

        Here, m is the model and u is the field (or fields).
        Given the model, m, we can calculate the fields u(m),
        however, the data we collect is a subset of the fields,
        and can be defined by a linear projection, P.

        .. math::
            d_\\text{pred} = Pu(m)

        We are interested in how changing the model transforms the data,
        as such we can take write the Taylor expansion:

        .. math::
            Pu(m + hv) = Pu(m) + hP\\frac{\partial u(m)}{\partial m} v + \mathcal{O}(h^2 \left\| v \\right\| )

        We can linearize and define the sensitivity matrix as:

        .. math::
            J = P\\frac{\partial u}{\partial m}

        The sensitivity matrix, and it's transpose will be used in the inverse problem
        to (locally) find how model parameters change the data, and optimize!
    """

    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def RHS(self):
        """
            Source matrix.
        """
        return self._RHS
    @RHS.setter
    def RHS(self, value):
        self._RHS = value

    @property
    def W(self):
        """
            Standard deviation weighting matrix.
        """
        return self._W
    @W.setter
    def W(self, value):
        self._W = value

    @property
    def P(self):
        """
            Projection matrix.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        return self._P
    @P.setter
    def P(self, value):
        self._P = value


    @property
    def dobs(self):
        """
            Observed data.
        """
        return self._dobs
    @dobs.setter
    def dobs(self, value):
        self._dobs = value


    def J(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv

            Working with the general PDE, c(m, u) = 0, where m is the model and u is the field,
            the sensitivity is defined as:

            .. math::
                J = P\\frac{\partial u}{\partial m}

            We can take the derivative of the PDE:

            .. math::
                \\nabla_m c(m, u) \delta m + \\nabla_u c(m, u) \delta u = 0

            If the forward problem is invertible, then we can rearrange for du/dm:

            .. math::
                J = - P \left( \\nabla_u c(m, u) \\right)^{-1} \\nabla_m c(m, u)

            This can often be computed given a vector (i.e. J(v)) rather than stored, as J is a large dense matrix.

        """
        pass

    def Jt(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: JTv

            Transpose of J
        """
        pass

    def field(self, m):
        """
            The field given the model.

            .. math::
                u(m)

        """
        pass

    def dpred(self, m, u=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        if u is None:
            u = self.field(m)
        return self.P*u

    def modelTransform(self, m):
        """
            :param numpy.array m: model
            :rtype: numpy.array
            :return: transformed model

            The modelTransform changes the model into the physical property.

            A common example of this is to invert for electrical conductivity
            in log space. In this case, your model will be log(sigma) and to
            get back to sigma, you can take the exponential:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma
        """
        return np.exp(mkvc(m))

    def modelTransformDeriv(self, m):
        """
            :param numpy.array m: model
            :rtype: scipy.csr_matrix
            :return: derivative of transformed model

            The modelTransform changes the model into the physical property.
            The modelTransformDeriv provides the derivative of the modelTransform.

            If the model transform is:

            .. math::

                m = \log{\sigma}

                \exp{m} = \exp{\log{\sigma}} = \sigma

            Then the derivative is:

            .. math::

                \\frac{\partial \exp{m}}{\partial m} = \\text{sdiag}(\exp{m})
        """
        return sdiag(np.exp(mkvc(m)))

    def misfit(self, m, u=None):
        """
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: float
            :return: data misfit

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.
        """

        R = self.W*(self.dpred(m, u=u) - self.dobs)
        R = mkvc(R)
        return 0.5*R.dot(R)

    def misfitDeriv(self, m, u=None):
        """
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

            If the field, u, is provided, the calculation of the data is fast:

            .. math::

                \mathbf{d}_\\text{pred} = \mathbf{Pu(m)}

                \mathbf{R} = \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs})

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.

            The derivative of this, with respect to the model, is:

            .. math::

                \\frac{\partial \mu_\\text{data}}{\partial \mathbf{m}} = \mathbf{J}^\\top \mathbf{W \circ R}

        """
        if u is None:
            u = self.field(m)

        R = self.W*(self.dpred(m, u=u) - self.dobs)

        dmisfit = 0
        for i in range(self.RHS.shape[1]): # Loop over each right hand side
            dmisfit += self.Jt(m, self.W[:,i]*R[:,i], u=u[:,i])

        return dmisfit

    def misfitDerivDeriv(self, m, u=None):
        """
            :param numpy.array m: geophysical model
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: data misfit derivative

            The data misfit using an l_2 norm is:

            .. math::

                \mu_\\text{data} = {1\over 2}\left| \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}) \\right|_2^2

            If the field, u, is provided, the calculation of the data is fast:

            .. math::

                \mathbf{d}_\\text{pred} = \mathbf{Pu(m)}

                \mathbf{R} = \mathbf{W} \circ (\mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs})

            Where P is a projection matrix that brings the field on the full domain to the data measurement locations;
            u is the field of interest; d_obs is the observed data; and W is the weighting matrix.

            The derivative of this, with respect to the model, is:

            .. math::

                \\frac{\partial \mu_\\text{data}}{\partial \mathbf{m}} = \mathbf{J}^\\top \mathbf{W \circ R}

                \\frac{\partial^2 \mu_\\text{data}}{\partial^2 \mathbf{m}} = \mathbf{J}^\\top \mathbf{W \circ W J}

        """
        if u is None:
            u = self.field(m)

        R = self.W*(self.dpred(m, u=u) - self.dobs)

        dmisfit = 0
        for i in range(self.RHS.shape[1]): # Loop over each right hand side
            dmisfit += self.Jt(m, self.W[:,i]*R[:,i], u=u[:,i])

        return dmisfit


class SyntheticProblem(object):
    """
        Has helpful functions when dealing with synthetic problems

        To use this class, inherit to your problem::

            class mySyntheticExample(Problem, SyntheticProblem):
                pass
    """
    def createData(self, m, std=0.05):
        """
            :param numpy.array m: geophysical model
            :param numpy.array std: standard deviation
            :rtype: numpy.array, numpy.array
            :return: dobs, Wd

            Create synthetic data given a model, and a standard deviation.

            Returns the observed data with random Gaussian noise
            and Wd which is the same size as data, and can be used to weight the inversion.
        """
        dobs = self.dpred(m)
        dobs = dobs
        noise = std*abs(dobs)*np.random.randn(*dobs.shape)
        dobs = dobs+noise
        eps = np.linalg.norm(mkvc(dobs),2)*1e-5
        Wd = 1/(abs(dobs)*std+eps)
        return dobs, Wd
