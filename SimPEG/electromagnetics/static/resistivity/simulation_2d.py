import numpy as np
from scipy import sparse as sp
from scipy.optimize import minimize
import warnings


from ....utils import (
    mkvc,
    sdiag,
    Zero,
    validate_type,
    validate_string,
    validate_integer,
    validate_active_indices,
)
from ....base import BaseElectricalPDESimulation
from ....base.pde_simulation import __inner_mat_mul_op as inner_mat_mul_op
from ....data import Data

from .survey import Survey
from .fields_2d import Fields2D, Fields2DCellCentered, Fields2DNodal
from .fields import FieldsDC, Fields3DCellCentered, Fields3DNodal
from .utils import _mini_pole_pole
from scipy.special import k0e, k1e, k0
from discretize.utils import make_boundary_bool


class BaseDCSimulation2D(BaseElectricalPDESimulation):
    r"""Base simulation class for the 2D DC resistivity problem.

    This class is used to define properties and methods necessary for solving the
    DC resistivity problem, where the electrical conductivity :math:`\sigma` is
    spatially invariant along the y-direction. 2D DC resistivity simulation classes
    solve the full 3D DC resistivity problem efficiently by solving a set of 2D
    problems in the wave domain of y.

    For a full description of the mathematical formulation, see the *Notes* section
    below.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : None, SimPEG.electromagnetics.static.resistivity.survey.Survey
        The DC resisitivity survey.
    nky : int
        Number of evaluations of the 2D problem in the wave domain.
    storeJ : bool
        Whether to construct and store the sensitivity matrix.
    miniaturize : bool
        If ``True``, we compute the fields for each unique source electrode location.
        We avoid computing the fields for repeated electrode locations and the
        fields for dipole sources can be constructed using superposition.
    do_trap : bool
        Use trap method to find the optimum set of quadrature points and weights
        in the wave domain for evaluating the set of 2D problems.
    fix_Jmatrix : bool
        Permanently fix the sensitivity matrix once constructed.
    surface_faces : None, numpy.ndarray of bool
        Array defining which faces to interpret as surfaces of the Neumann boundary.

    Notes
    -----
    For current :math:`I` injected at point :math:`r_s`, the full DC resistivity
    problem is expressed as:

    .. math::
        \nabla \cdot \sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where :math:`\nabla = [\partial_x, \; \partial_y, \; \partial_z \, ]^T` is the
    gradient operator, :math:`\sigma` is the electrical conductivity, and we wish
    to solve for the electric potential :math:`\phi`.

    The the 2D simulation, we assume that electrical conductivity is invariant
    along the y-direction. This allows us to take the Laplace transform along
    the y-direction. In the wave domain of y, the solution for the electric potential
    :math:`\Phi` for a wavenumber :math:`k_y` is a 2D problem of the form:

    .. math::
        \nabla_{\! xz} \cdot \sigma \nabla_{\! xz} \Phi + k_y^2 \sigma \Phi
        = - I \delta (x-x_s) \delta (z-z_s)

    where :math:`\nabla_{\! xz} = [\partial_x, \; \partial_z \, ]^T` is a partial
    gradient operator that acts along the x and z directions.

    For an optimum set of wavenumbers :math:`k_y^{(i)}` and coefficients :math:`\alpha^{(i)}`,
    2D simulation classes solve a set of 2D problems in the wave domain using
    mimetic finite volume. And the full 3D solution for a specified y-coordinate
    location is computed according to:

    .. math::
        \phi = \sum_{i=1}^{nk_y} \alpha^{(i)} \, \Phi \! \left ( k_y^{(i)} \right )

    where :math:`nk_y` is the number of wavenumbers used to compute the solution.
    """

    fieldsPair = Fields2D  # SimPEG.EM.Static.Fields2D
    fieldsPair_fwd = FieldsDC  # SimPEG.EM.Static.FieldsDC
    # there's actually nT+1 fields, so we don't need to store the last one
    _mini_survey = None

    def __init__(
        self,
        mesh,
        survey=None,
        nky=11,
        storeJ=False,
        miniaturize=False,
        do_trap=False,
        fix_Jmatrix=False,
        surface_faces=None,
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.nky = nky
        self.storeJ = storeJ
        self.fix_Jmatrix = fix_Jmatrix
        self.surface_faces = surface_faces

        do_trap = validate_type("do_trap", do_trap, bool)
        if not do_trap:
            # try to find an optimal set of quadrature points and weights
            def get_phi(r):
                e = np.ones_like(r)

                def phi(k):
                    # use log10 transform to enforce positivity
                    k = 10**k
                    A = r[:, None] * k0(r[:, None] * k)
                    v_i = A @ np.linalg.solve(A.T @ A, A.T @ e)
                    dv = (e - v_i) / len(r)
                    return np.linalg.norm(dv)

                def g(k):
                    A = r[:, None] * k0(r[:, None] * k)
                    return np.linalg.solve(A.T @ A, A.T @ e)

                return phi, g

            # find the minimum cell spacing, and the maximum side of the mesh
            min_r = min(self.mesh.edge_lengths)
            max_r = max(
                np.max(self.mesh.nodes, axis=0) - np.min(self.mesh.nodes, axis=0)
            )
            # generate test points log spaced between these two end members
            rs = np.logspace(np.log10(min_r / 4), np.log10(max_r * 4), 100)

            min_rinv = -np.log10(rs).max()
            max_rinv = -np.log10(rs).min()
            # a decent initial guess of the k_i's for the optimization = 1/rs
            k_i = np.linspace(min_rinv, max_rinv, self.nky)

            # these functions depend on r, so grab them
            func, g_func = get_phi(rs)

            # just use scipy's minimize for ease
            out = minimize(func, k_i)
            if self.verbose:
                print(f"optimized ks converged? : {out['success']}")
                print(f"Estimated transform Error: {out['fun']}")
            # transform the solution back to normal points
            points = 10 ** out["x"]
            # transform has a 2/pi and we want 1/pi, so divide by 2
            weights = g_func(points) / 2
            if not out["success"]:
                warnings.warn(
                    "Falling back to trapezoidal for integration. "
                    "You may need to change nky.",
                    stacklevel=2,
                )
                do_trap = True
        if do_trap:
            if self.verbose:
                print("doing trap")
            y = 0.0

            points = np.logspace(-4, 1, self.nky)
            dky = np.diff(points) / 2
            weights = np.r_[dky, 0] + np.r_[0, dky]
            weights *= np.cos(points * y)  # *(1.0/np.pi)
            # assume constant value at 0 frequency?
            weights[0] += points[0] / 2 * (1.0 + np.cos(points[0] * y))
            weights /= np.pi

        self._quad_weights = weights
        self._quad_points = points

        self.Ainv = [None for i in range(self.nky)]
        self.nT = self.nky - 1  # Only for using TimeFields

        # Do stuff to simplify the forward and JTvec operation if number of dipole
        # sources is greater than the number of unique pole sources
        miniaturize = validate_type("miniaturize", miniaturize, bool)
        if miniaturize:
            self._dipoles, self._invs, self._mini_survey = _mini_pole_pole(self.survey)

    @property
    def survey(self):
        """The DC survey object.

        Returns
        -------
        SimPEG.electromagnetics.static.resistivity.survey.Survey
            The DC survey object.
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value

    @property
    def _Pxz_from_xyz(self):
        """Projection matrix to extract x and z conductivities from
        axial anisotropy vector [sig_x, sig_y, sig_z].
        """

        if getattr(self, "__Pxz_from_xyz", None) is None:
            nC = self.mesh.nC
            P = sp.vstack(
                [
                    sp.diags(np.ones(nC), 0, shape=(nC, 3 * nC)),
                    sp.diags(np.ones(nC), 2 * nC, shape=(nC, 3 * nC)),
                ]
            )
            self.__Pxz_from_xyz = P

        return self.__Pxz_from_xyz

    @property
    def _Py_from_xyz(self):
        """Projection matrix to extract y conductivities from
        axial anisotropy vector [sig_x, sig_y, sig_z].
        """

        if getattr(self, "__Py_from_xyz", None) is None:
            nC = self.mesh.nC
            P = sp.diags(np.ones(nC), nC, shape=(nC, 3 * nC))
            self.__Py_from_xyz = P

        return self.__Py_from_xyz

    @property
    def nky(self):
        """Number of wavenumbers :math:`k_y` used to solve the problem.

        Returns
        -------
        int
            Number of wavenumbers :math:`k_y` used to solve the problem.
        """
        return self._nky

    @nky.setter
    def nky(self, value):
        self._nky = validate_integer("nky", value, min_val=3)

    @property
    def storeJ(self):
        """Whether to store the sensitivity matrix.

        Returns
        -------
        bool
        """
        return self._storeJ

    @storeJ.setter
    def storeJ(self, value):
        self._storeJ = validate_type("storeJ", value, bool)

    @property
    def fix_Jmatrix(self):
        """Permanently fix the sensitivity matrix once constructed.

        Returns
        -------
        bool
        """
        return self._fix_Jmatrix

    @fix_Jmatrix.setter
    def fix_Jmatrix(self, value):
        self._fix_Jmatrix = validate_type("fix_Jmatrix", value, bool)

    @property
    def surface_faces(self):
        """Array defining which faces to interpret as surfaces of Neumann boundary.

        DC problems will always enforce a Neumann boundary on surface interfaces.
        The default (available on semi-structured grids) assumes the top interface
        is the surface.

        Returns
        -------
        None or (n_bf, ) numpy.ndarray of bool
        """
        return self._surface_faces

    @surface_faces.setter
    def surface_faces(self, value):
        if value is not None:
            n_bf = self.mesh.boundary_faces.shape[0]
            value = validate_active_indices("surface_faces", value, n_bf)
        self._surface_faces = value

    def fields(self, m=None):
        """Solve for the fields for all 2D problems in the wave domain.

        Parameters
        ----------
        m : None, (n_param,) numpy.ndarray
            The model.

        Returns
        -------
        SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D
            The fields for all 2D problems in the wave domain.
        """
        if self.verbose:
            print(">> Compute fields")
        if m is not None:
            self.model = m
        if self.Ainv[0] is not None:
            for i in range(self.nky):
                self.Ainv[i].clean()
        f = self.fieldsPair(self)
        kys = self._quad_points
        f._quad_weights = self._quad_weights
        for iky, ky in enumerate(kys):
            A = self.getA(ky)
            if self.Ainv[iky] is not None:
                self.Ainv[iky].clean()
            self.Ainv[iky] = self.solver(A, **self.solver_opts)
            RHS = self.getRHS(ky)
            u = self.Ainv[iky] * RHS
            f[:, self._solutionType, iky] = u
        return f

    def fields_to_space(self, f, y=0.0):
        """Solve for the fields at the y-coordinate location specified.

        Parameters
        ----------
        f : SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D
            The fields for all 2D problems solved in the wave domain.
        y : float
            Y-coordinate location for which we want the 3D solution.

        Returns
        -------
        SimPEG.electromagnetics.static.resistivity.fields.FieldsDC
            The fields at the y-coordinate location specified.
        """
        f_fwd = self.fieldsPair_fwd(self)
        phi = f[:, self._solutionType, :].dot(self._quad_weights)
        f_fwd[:, self._solutionType] = phi
        return f_fwd

    def dpred(self, m=None, f=None):
        """Predict the data for a given model.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray, optional
            The model parameters.
        f : SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D, optional
            The 2D fields solved in the wave domain.

        Returns
        -------
        (n_data,) numpy.ndarray
            The predicted data array.
        """
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        weights = self._quad_weights
        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        temp = np.empty(survey.nD)
        count = 0
        for src in survey.source_list:
            for rx in src.receiver_list:
                d = rx.eval(src, self.mesh, f).dot(weights)
                temp[count : count + len(d)] = d
                count += len(d)

        return self._mini_survey_data(temp)

    def getJ(self, m, f=None):
        r"""Generate the full sensitivity matrix.

        This method generates and stores the full sensitivity matrix for the
        model provided. I.e.:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        where :math:`\mathbf{d}` are the data and :math:`\mathbf{m}` are the model parameters.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters
        f : SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D, optional
            2D fields solved in the wave domain.

        Returns
        -------
        (n_data, n_param) numpy.ndarray
            The full sensitivity matrix.
        """
        if getattr(self, "_Jmatrix", None) is None:
            if self.verbose:
                print("Calculating J and storing")
            self.model = m
            if f is None:
                f = self.fields(m)
            self._Jmatrix = (self._Jtvec(m, v=None, f=f)).T
        return self._Jmatrix

    def Jvec(self, m, v, f=None):
        r"""Compute the sensitivity matrix times a vector.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the matrix-vector product:

        .. math::
            \mathbf{J v}

        for a given vector :math:`v`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        v : (n_param,) numpy.ndarray
            The vector.
        f : SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D, optional
            2D fields solved in the wave domain.

        Returns
        -------
        (n_data,) numpy.ndarray
            The sensitivity matrix times a vector.
        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = mkvc(np.dot(J, v))
            return Jv

        self.model = m

        if f is None:
            f = self.fields(m)

        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        kys = self._quad_points
        weights = self._quad_weights

        Jv = np.zeros(survey.nD)
        # Assume y=0.
        # This needs some thoughts to implement in general when src is dipole

        # TODO: this loop is pretty slow .. (Parellize)
        for iky, ky in enumerate(kys):
            u_ky = f[:, self._solutionType, iky]
            count = 0
            for i_src, src in enumerate(survey.source_list):
                u_src = u_ky[:, i_src]
                dA_dm_v = self.getADeriv(ky, u_src, v, adjoint=False)
                # dRHS_dm_v = self.getRHSDeriv(ky, src, v) = 0
                du_dm_v = self.Ainv[iky] * (-dA_dm_v)  # + dRHS_dm_v)
                for rx in src.receiver_list:
                    df_dmFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                    df_dm_v = df_dmFun(iky, src, du_dm_v, v, adjoint=False)
                    Jv1_temp = rx.evalDeriv(src, self.mesh, f, df_dm_v)
                    # Trapezoidal intergration
                    Jv[count : count + len(Jv1_temp)] += weights[iky] * Jv1_temp
                    count += len(Jv1_temp)

        return self._mini_survey_data(Jv)

    def Jtvec(self, m, v, f=None):
        r"""Compute the adjoint sensitivity matrix times a vector.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the matrix-vector product:

        .. math::
            \mathbf{J^T v}

        for a given vector :math:`v`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        v : (n_data,) numpy.ndarray
            The vector.
        f : SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D, optional
            2D fields solved in the wave domain

        Returns
        -------
        (n_param,) numpy.ndarray
            The adjoint sensitivity matrix times a vector.
        """
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = mkvc(np.dot(J.T, v))
            return Jtv

        self.model = m

        if f is None:
            f = self.fields(m)

        return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        Full J matrix can be computed by inputing v=None
        """
        kys = self._quad_points
        weights = self._quad_weights
        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        if v is not None:
            # Ensure v is a data object.
            if isinstance(v, Data):
                v = v.dobs
            v = self._mini_survey_dataT(v)
            Jtv = np.zeros(m.size, dtype=float)

            for iky, ky in enumerate(kys):
                u_ky = f[:, self._solutionType, iky]
                count = 0
                for i_src, src in enumerate(survey.source_list):
                    u_src = u_ky[:, i_src]
                    df_duT_sum = 0
                    df_dmT_sum = 0
                    for rx in src.receiver_list:
                        my_v = v[count : count + rx.nD]
                        count += rx.nD
                        # wrt f, need possibility wrt m
                        PTv = rx.evalDeriv(src, self.mesh, f, my_v, adjoint=True)
                        df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                        df_duT, df_dmT = df_duTFun(iky, src, None, PTv, adjoint=True)
                        df_duT_sum += df_duT
                        df_dmT_sum += df_dmT

                    ATinvdf_duT = self.Ainv[iky] * df_duT_sum

                    dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT, adjoint=True)
                    # dRHS_dmT = self.getRHSDeriv(ky, src, ATinvdf_duT,
                    #                            adjoint=True)
                    du_dmT = -dA_dmT  # + dRHS_dmT=0
                    Jtv += weights[iky] * (df_dmT + du_dmT).astype(float)
            return mkvc(Jtv)

        else:
            # This is for forming full sensitivity matrix
            Jt = np.zeros((self.model.size, survey.nD), order="F")
            for iky, ky in enumerate(kys):
                u_ky = f[:, self._solutionType, iky]
                istrt = 0
                for i_src, src in enumerate(survey.source_list):
                    u_src = u_ky[:, i_src]
                    for rx in src.receiver_list:
                        # wrt f, need possibility wrt m
                        PT = rx.evalDeriv(src, self.mesh, f).toarray().T
                        ATinvdf_duT = self.Ainv[iky] * PT

                        dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT, adjoint=True)
                        Jtv = -weights[iky] * dA_dmT  # RHS=0
                        iend = istrt + rx.nD
                        if rx.nD == 1:
                            Jt[:, istrt] += Jtv
                        else:
                            Jt[:, istrt:iend] += Jtv
                        istrt += rx.nD
            return (self._mini_survey_data(Jt.T)).T

    def getSourceTerm(self, ky):
        """Compute the discrete source terms (right-hand sides) for the wavenumber provided.

        Parameters
        ----------
        ky : float
            The wavenumber.

        Returns
        -------
        (nC or nN, nSrc) numpy.ndarray
            The array containing the right-hand sides for all sources for
            the wavenumber specified.
        """

        if self._mini_survey is not None:
            Srcs = self._mini_survey.source_list
        else:
            Srcs = self.survey.source_list

        if self._formulation == "EB":
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == "HJ":
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)), order="F")

        for i, src in enumerate(Srcs):
            q[:, i] = src.eval(self)
        return q

    @property
    def deleteTheseOnModelUpdate(self):
        """Returns the properties to delete when model is updated.

        Returns
        -------
        list of str
            The properties to delete when model the is updated.
        """
        toDelete = super().deleteTheseOnModelUpdate
        if self.fix_Jmatrix:
            return toDelete
        return toDelete + ["_Jmatrix"]

    def _mini_survey_data(self, d_mini):
        if self._mini_survey is not None:
            out = d_mini[self._invs[0]]  # AM
            out[self._dipoles[0]] -= d_mini[self._invs[1]]  # AN
            out[self._dipoles[1]] -= d_mini[self._invs[2]]  # BM
            out[self._dipoles[0] & self._dipoles[1]] += d_mini[self._invs[3]]  # BN
        else:
            out = d_mini
        return out

    def _mini_survey_dataT(self, v):
        if self._mini_survey is not None:
            out = np.zeros(self._mini_survey.nD)
            # Need to use ufunc.at because there could be repeated indices
            # That need to be properly handled.
            np.add.at(out, self._invs[0], v)  # AM
            np.subtract.at(out, self._invs[1], v[self._dipoles[0]])  # AN
            np.subtract.at(out, self._invs[2], v[self._dipoles[1]])  # BM
            np.add.at(out, self._invs[3], v[self._dipoles[0] & self._dipoles[1]])  # BN
            return out
        else:
            out = v
        return out


class Simulation2DCellCentered(BaseDCSimulation2D):
    r"""Cell centered 2D DC resistivity simulation.

    Simulation class which solves the 2D DC resistivity problem for electric
    potentials at cell centers. For a full description of the numerical approach,
    see the *Notes* section below.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The mesh.
    survey : None, Survey
        The DC resisitivity survey.
    nky : int
        Number of evaluations of the 2D problem in the wave domain.
    storeJ : bool
        Whether to construct and store the sensitivity matrix.
    miniaturize : bool
        If ``True``, we compute the fields for each unique source electrode location.
        We avoid computing the fields for repeated electrode locations and the
        fields for dipole sources can be constructed using superposition.
    do_trap : bool
        Use trap method to find the optimum set of quadrature points and weights
        in the wave domain for evaluating the set of 2D problems.
    fix_Jmatrix : bool
        Permanently fix the sensitivity matrix once constructed.
    surface_faces : None, numpy.ndarray of bool
        Array defining which faces to interpret as surfaces of the Neumann boundary.

    Notes
    -----
    For current :math:`I` injected at point :math:`r_s`, the full DC resistivity
    problem is expressed as:

    .. math::
        \nabla \cdot \sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where :math:`\nabla = [\partial_x, \; \partial_y, \; \partial_z \, ]^T` is the
    gradient operator, :math:`\sigma` is the electrical conductivity, and we wish
    to solve for the electric potential :math:`\phi`.

    The the 2D simulation, we assume that electrical conductivity is invariant
    along the y-direction. This allows us to take the Laplace transform along
    the y-direction. In the wave domain of y, the solution for the electric potential
    :math:`\Phi` for a wavenumber :math:`k_y` is a 2D problem of the form:

    .. math::
        \nabla_{\! xz} \cdot \sigma \nabla_{\! xz} \Phi + k_y^2 \sigma \Phi
        = - I \delta (x-x_s) \delta (z-z_s)

    where :math:`\nabla_{\! xz} = [\partial_x, \; \partial_z \, ]^T` is a partial
    gradient operator that acts along the x and z directions.

    Using mimetic finite volume, the 2D problem can be solved numerically by solving
    the following linear system:

    .. math::
        \big [ \mathbf{D \, M_{f\rho}^{-1} \, G} + k_y^2 \, \mathbf{M_{c\sigma}} \big ]
        \boldsymbol{\Phi} = \mathbf{q}

    where

    * :math:`\boldsymbol{\Phi}` are the discrete electric potentials defined at cell centers
    * :math:`\mathbf{D}` is the 2D face-divergence operator with imposed boundary conditions
    * :math:`G` is the 2D cell-gradient operator
    * :math:`M_{f\rho}` is the resistivity inner-product matrix on cell faces; note :math:`\rho = 1/\sigma`
    * :math:`M_{c\sigma}` is the conductivity inner-product matrix at cell centers

    For an optimum set of wavenumbers :math:`k_y^{(i)}` and coefficients :math:`\alpha^{(i)}`,
    we solve a set of discrete 2D problems in the wave domain. And the full 3D solution
    for a specified y-coordinate location is computed according to:

    .. math::
        \boldsymbol{\phi} = \sum_{i=1}^{nk_y}
        \alpha^{(i)} \, \boldsymbol{\Phi} \! \left ( k_y^{(i)} \right )

    where :math:`nk_y` is the number of wavenumbers used to compute the solution.

    **Axial Anisotropy:**

    In this case, the DC resistivity problem is defined according to:

    .. math::
        \nabla \cdot \Sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where

    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_x & 0 & 0 \\ 0 & \sigma_y & 0 \\ 0 & 0 & \sigma_z
        \end{bmatrix}

    The discrete 2D problem solved in the wave domain of y still takes the form:

    .. math::
        \big [ \mathbf{D \, M_{f\rho}^{-1} \, G} + k_y^2 \, \mathbf{M_{c\sigma}} \big ]
        \boldsymbol{\Phi} = \mathbf{q}

    However,

    * :math:`M_{f\rho}` is a resistivity inner-product matrix on cell faces constructed using axial resistivities :math:`\rho_x = 1/\sigma_x` and :math:`\rho_z = 1/\sigma_z`
    * :math:`M_{c\sigma}` is the conductivity inner-product matrix at cell centers constructed using axial conductivity :math:`\sigma_y`
    """

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields2DCellCentered
    fieldsPair_fwd = Fields3DCellCentered

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):
        super().__init__(mesh, survey=survey, **kwargs)
        V = sdiag(self.mesh.cell_volumes)
        self.Div = V @ self.mesh.face_divergence
        self.Grad = self.Div.T
        self.bc_type = bc_type

    @property
    def bc_type(self):
        """Type of boundary condition to use for simulation.

        The boundary conditions supported by the :class:`Simulation2DCellCentered` are:

        * "Dirichlet": Zero Dirichlet on the boundary (natural boundary conditions)
        * "Neumann": Zero Neumann on the boundary
        * "Robin" or "Mixed": Mix of Robin and zero Neumann boundary conditions.

        Returns
        -------
        {"Dirichlet", "Neumann", "Robin", "Mixed"}
        """
        return self._bc_type

    @property
    def MfRhoI(self):
        r"""Inverse of the resistivity inner-product matrix on faces.

        Where the inner-product between a vector :math:`\vec{u}` and the electrical
        resistivity :math:`\rho` times a vector :math:`\vec{v}` can be approximated by a
        discrete operation on cell faces:

        .. math::
            \langle \vec{u}, \, \rho \vec{v} \rangle \approx
            \mathbf{u^T M_{f \rho} \, v}

        this property returns the inverse of the inner-product matrix :math:`\mathbf{M_{f \rho}}`.

        Returns
        -------
        (n_faces, n_faces) scipy.sparse.csr_matrix
            The inverse of the inner-product matrix.
        """

        # Isotropic case
        if self.rho.size == self.mesh.nC:
            return super().MfRhoI

        # Override in the case of anisotropy
        stash_name = "_MfI_Rho"
        if getattr(self, stash_name, None) is None:
            if self.rho.size == 3 * self.mesh.nC:
                model = self._Pxz_from_xyz * self.rho
                M_prop = self.mesh.get_face_inner_product(
                    model=model, invert_matrix=True
                )
                setattr(self, stash_name, M_prop)
            else:
                raise NotImplementedError(
                    "Only isotropic and linear isotropic resistivities implemented."
                )

        return getattr(self, stash_name)

    def MfRhoDeriv(self, u, v=None, adjoint=False):
        r"""Derivative operation for the inverse of the resistivity inner-product matrix.

        Let :math:`\mathbf{M_{f\rho}}` represent the resistivity inner-product matrix on mesh faces.
        Where :math:`\mathbf{u}` and :math:`\mathbf{v}` are vectors, this method returns

        .. math::
            \frac{\partial (\mathbf{M_{f\rho}^{-1} \, u} )}{\partial \mathbf{m}} \, \mathbf{v}

        or the adjoint operation

        .. math::
            \left ( \frac{\partial (\mathbf{M_{f\rho}^{-1} \, u})}{\partial \mathbf{m}} \, \right )^T \mathbf{v}

        If the input argument *v* is ``None``, the method will return a function handle that
        acts on an input argument *v*.

        Parameters
        ----------
        u : (n_cells,) numpy.ndarray
            A vector.
        v : numpy.ndarray, optional
            A vector. (n_param,) for the standard operation. (n_faces,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative operation for the inverse of the resistivity inner-product matrix.
            (n_faces,) for the standard operation. (n_param,) for the adjoint operation.
        """

        # Isotropic case
        if self.rho.size == self.mesh.nC:
            return super().MfRhoDeriv(u, v, adjoint)

        # Override for anisotropic case
        if self.rhoMap is None:
            return Zero()
        if isinstance(u, Zero) or isinstance(v, Zero):
            return Zero()

        stash_name = "_Mf_Rho_deriv"
        if getattr(self, stash_name, None) is None:
            rho = self.rho
            nC = self.mesh.nC

            if rho.size == 3 * nC:
                if self.mesh._meshType.lower() in (
                    "cyl",
                    "tensor",
                    "tree",
                ):
                    # Edge inner product derivative for x and z axial conductivitites
                    M_deriv_func = self.mesh.get_face_inner_product_deriv(
                        model=self._Pxz_from_xyz * rho
                    )

                    # Derivative wrt all axial conductivities
                    prop_deriv = self.rhoDeriv

                    M_prop_deriv = (
                        M_deriv_func(np.ones(self.mesh.n_faces))
                        @ self._Pxz_from_xyz
                        @ prop_deriv
                    )
                    setattr(self, stash_name, M_prop_deriv)
                else:
                    raise NotImplementedError(
                        "Only implemented for 'CylindricalMesh', 'TensorMesh' and 'TreeMesh'."
                    )

            else:
                raise NotImplementedError(
                    "Only isotropic and axial anisotropic conductivities implemented."
                )

        return inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

    @property
    def MccSigma(self):
        r"""Conductivity inner-product matrix at cell centers.

        Where the inner-product between a scalar :math:`\phi` and the electrical
        conductivity :math:`\sigma` times a scalar :math:`\psi` can be approximated by a
        discrete operation at cell centers:

        .. math::
            \langle \phi, \, \sigma \psi \rangle \approx
            \mathbf{\phi^T M_{c \sigma} \, \psi}

        this property returns the inner-product matrix :math:`\mathbf{M_{c \sigma}}`.

        Returns
        -------
        (n_cells, n_cells) scipy.sparse.csr_matrix
            The the inner-product matrix.
        """

        # Isotropic case
        if self.sigma.size == self.mesh.nC:
            return super().MccSigma

        # Override in the case of anisotropy
        stash_name = "_Mcc_Sigma"
        if getattr(self, stash_name, None) is None:
            if self.sigma.size == 3 * self.mesh.nC:
                vol = self.mesh.cell_volumes
                M_prop = sp.diags(vol * (self._Py_from_xyz * self.sigma), format="csr")
                setattr(self, stash_name, M_prop)
            else:
                raise NotImplementedError(
                    "Only isotropic and linear isotropic conductivities implemented."
                )

        return getattr(self, stash_name)

    def MccSigmaDeriv(self, u, v=None, adjoint=False):
        r"""Derivative operation for the conductivity inner-product matrix at cell centers.

        Let :math:`\mathbf{M_{c\sigma}}` represent the conductivity inner-product matrix at cell centers.
        Where :math:`\mathbf{u}` and :math:`\mathbf{v}` are vectors, this method returns

        .. math::
            \frac{\partial (\mathbf{M_{c\sigma} \, u} )}{\partial \mathbf{m}} \, \mathbf{v}

        or the adjoint operation

        .. math::
            \left ( \frac{\partial (\mathbf{M_{c\sigma} \, u})}{\partial \mathbf{m}} \, \right )^T \mathbf{v}

        If the input argument *v* is ``None``, the method will return a function handle that
        acts on an input argument *v*.

        Parameters
        ----------
        u : (n_cells,) numpy.ndarray
            A vector.
        v : numpy.ndarray, optional
            A vector. (n_param,) for the standard operation. (n_cells,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative operation for the conductivity inner-product matrix. (n_cells,) for the
            standard operation. (n_param,) for the adjoint operation.
        """

        # Isotropic case
        if self.sigma.size == self.mesh.nC:
            return super().MccSigmaDeriv(u, v, adjoint)

        # Override for anisotropic case
        if self.sigmaMap is None:
            return Zero()
        if isinstance(u, Zero) or isinstance(v, Zero):
            return Zero()

        stash_name = "_Mcc_Sigma_deriv"
        if getattr(self, stash_name, None) is None:
            sigma = self.sigma
            nC = self.mesh.nC

            if sigma.size == 3 * nC:
                M_prop_deriv = (
                    sp.diags(self.mesh.cell_volumes)
                    * self._Py_from_xyz
                    * self.sigmaDeriv
                )
                setattr(self, stash_name, M_prop_deriv)

            else:
                raise NotImplementedError(
                    "Only isotropic and axial anisotropic conductivities implemented."
                )

        return inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

    @bc_type.setter
    def bc_type(self, value):
        self._bc_type = validate_string(
            "bc_type", value, ["Dirichlet", "Neumann", ("Robin", "Mixed")]
        )

    def getA(self, ky):
        r"""Compute the system matrix for the wavenumber provided.

        The discrete solution to the 2D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\Phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\Phi` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. This method returns the system matrix
        for the cell-centered formulation for the wavenumber :math:`k_y` provided, i.e.:

        .. math::
            \mathbf{A} = \mathbf{D \, M_{f\rho}^{-1} \, G} + k_y^2 \mathbf{M_{c\sigma}}

        where :math:`\mathbf{D}` is the face divergence operator, :math:`\mathbf{G}` is the
        cell gradient operator with imposed boundary conditions, :math:`\mathbf{M_{f\rho}}`
        is the inner product matrix for resistivities projected to faces,
        and :math:`\mathbf{M_{c\sigma}}` is the inner product matrix for conductivities
        at cell centers.

        Parameters
        ----------
        ky : float
            The wavenumber.

        Returns
        -------
        (n_cells, n_cells) scipy.sparse.csr_matrix
            The sparse system matrix.
        """
        # To handle Mixed boundary condition
        self.setBC(ky=ky)
        D = self.Div
        G = self.Grad
        if self.bc_type != "Dirichlet":
            G = G - self._MBC[ky]
        MfRhoI = self.MfRhoI
        # Get resistivity rho
        A = D * MfRhoI * G + ky**2 * self.MccSigma
        if self.bc_type == "Neumann":
            A[0, 0] = A[0, 0] + 1.0
        return A

    def getADeriv(self, ky, u, v, adjoint=False):
        r"""Derivative operation for the system matrix times a vector.

        The discrete solution to the 2D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A \, u} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\mathbf{u}` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. For a vector :math:`\mathbf{v}`, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, u})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, u})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        ky : float
            The wavenumber.
        u : (n_cells,) numpy.ndarray
            The solution for the fields for the current model; i.e. electric potentials at cell centers.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_cells,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_cells,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        D = self.Div
        G = self.Grad
        if self.bc_type != "Dirichlet":
            G = G - self._MBC[ky]
        if adjoint:
            return self.MfRhoIDeriv(
                G * u, D.T * v, adjoint=adjoint
            ) + ky**2 * self.MccSigmaDeriv(u, v, adjoint=adjoint)
        else:
            return D * self.MfRhoIDeriv(
                G * u, v, adjoint=adjoint
            ) + ky**2 * self.MccSigmaDeriv(u, v, adjoint=adjoint)

    def getRHS(self, ky):
        r"""Compute the source terms for the wavenumber provided.

        For a single source, the discrete solution to the 2D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\phi` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side corresponding to the source term.
        This method computes and returns an array :math:`\mathbf{Q}`, whose columns are
        the right-hand sides for all sources.

        Parameters
        ----------
        ky : float
            The wavenumber.

        Returns
        -------
        (n_cells, nSrc) numpy.ndarray
            The array containing the right-hand sides for all sources for
            the wavenumber specified.
        """

        RHS = self.getSourceTerm(ky)
        return RHS

    def getRHSDeriv(self, ky, src, v, adjoint=False):
        r"""Derivative of the source term with respect to the model times a vector.

        The discrete solution to the 2D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\Phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\boldsymbol{\Phi}` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side. This method returns the derivative
        of the right-hand side with respect to the model times a vector, i.e.:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}} \mathbf{v}

        or the adjoint operation:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}}^T \mathbf{v}

        Parameters
        ----------
        ky : float
            The wavenumber.
        src : SimPEG.electromagnetic.static.resistivity.sources.BaseSrc
            The source object.
        v : numpy.ndarray
            The vector. Has shape (n_param,) when performing the standard derivative operation.
            Has shape (n_cells,) when performing the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        Zero or numpy.ndarray
            Returns :py:class:`Zero` if the derivative with respect to the model is zero.
            Returns (n_cells,) :class:`numpy.ndarray` when computing the standard
            derivative operation. Returns (n_param,) :class:`numpy.ndarray` when performing
            the adjoint.
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, ky, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def setBC(self, ky=None):
        """Sets the boundary conditions on the cell gradient operator.

        This method will set the boundary conditions on the cell gradient
        operator based on the value of the :py:attr:`bc_type` property.
        The options are:

        * "Dirichlet": Zero Dirichlet on the boundary (natural boundary conditions)
        * "Neumann": Zero Neumann on the boundary
        * "Robin" or "Mixed": Mix of zero Dirichlet and zero Neumann. Faces that use the Neumann boundary conditions are set using the :py:attr:`surface_faces` property.

        Parameters
        ----------
        ky : float (optional)
            The wavenumber.
        """
        if self.bc_type == "Dirichlet":
            return
        if getattr(self, "_MBC", None) is None:
            self._MBC = {}
        if ky in self._MBC:
            # I have already created the BC matrix for this wavenumber
            return
        if self.bc_type == "Neumann":
            alpha, beta, gamma = 0, 1, 0
        else:
            mesh = self.mesh
            boundary_faces = mesh.boundary_faces
            boundary_normals = mesh.boundary_face_outward_normals
            n_bf = len(boundary_faces)

            # Top gets 0 Neumann
            alpha = np.zeros(n_bf)
            beta = np.ones(n_bf)
            gamma = 0

            # assume a source point at the middle of the top of the mesh
            middle = np.median(mesh.nodes, axis=0)
            top_v = np.max(mesh.nodes[:, -1])
            source_point = np.r_[middle[:-1], top_v]

            r_vec = boundary_faces - source_point
            r = np.linalg.norm(r_vec, axis=-1)
            r_hat = r_vec / r[:, None]
            r_dot_n = np.einsum("ij,ij->i", r_hat, boundary_normals)

            if self.surface_faces is None:
                # determine faces that are on the sides and bottom of the mesh...
                if mesh._meshType.lower() == "tree":
                    not_top = boundary_faces[:, -1] != top_v
                elif mesh._meshType.lower() in ["tensor", "curv"]:
                    # mesh faces are ordered, faces_x, faces_y, faces_z so...
                    is_b = make_boundary_bool(mesh.shape_faces_y)
                    is_t = np.zeros(mesh.shape_faces_y, dtype=bool, order="F")
                    is_t[:, -1] = True
                    is_t = is_t.reshape(-1, order="F")[is_b]
                    not_top = np.ones(boundary_faces.shape[0], dtype=bool)
                    not_top[-len(is_t) :] = ~is_t
                else:
                    raise NotImplementedError(
                        f"Unable to infer surface boundaries for {type(mesh)}, please "
                        f"set the `surface_faces` property."
                    )
            else:
                not_top = ~self.surface_faces

            # use the exponentialy scaled modified bessel function of second kind,
            # (the division will cancel out the scaling)
            # This is more stable for large values of ky * r
            # actual ratio is k1/k0...
            alpha[not_top] = (ky * k1e(ky * r) / k0e(ky * r) * r_dot_n)[not_top]

        B, bc = self.mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
        # bc should always be 0 because gamma was always 0 above
        self._MBC[ky] = B


class Simulation2DNodal(BaseDCSimulation2D):
    r"""Nodal 2D DC resistivity simulation.

    Simulation class which solves the 2D DC resistivity problem for electric
    potentials on mesh nodes. For a full description of the numerical approach,
    see the *Notes* section below.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The mesh.
    survey : None, Survey
        The DC resisitivity survey.
    nky : int
        Number of evaluations of the 2D problem in the wave domain.
    storeJ : bool
        Whether to construct and store the sensitivity matrix.
    miniaturize : bool
        If ``True``, we compute the fields for each unique source electrode location.
        We avoid computing the fields for repeated electrode locations and the
        fields for dipole sources can be constructed using superposition.
    do_trap : bool
        Use trap method to find the optimum set of quadrature points and weights
        in the wave domain for evaluating the set of 2D problems.
    fix_Jmatrix : bool
        Permanently fix the sensitivity matrix once constructed.
    surface_faces : None, numpy.ndarray of bool
        Array defining which faces to interpret as surfaces of the Neumann boundary.

    Notes
    -----
    For current :math:`I` injected at point :math:`r_s`, the full DC resistivity
    problem is expressed as:

    .. math::
        \nabla \cdot \sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where :math:`\nabla = [\partial_x, \; \partial_y, \; \partial_z \, ]^T` is the
    gradient operator, :math:`\sigma` is the electrical conductivity, and we wish
    to solve for the electric potential :math:`\phi`.

    The the 2D simulation, we assume that electrical conductivity is invariant
    along the y-direction. This allows us to take the Laplace transform along
    the y-direction. In the wave domain of y, the solution for the electric potential
    :math:`\Phi` for a wavenumber :math:`k_y` is a 2D problem of the form:

    .. math::
        \nabla_{\! xz} \cdot \sigma \nabla_{\! xz} \Phi + k_y^2 \sigma \Phi
        = - I \delta (x-x_s) \delta (z-z_s)

    where :math:`\nabla_{\! xz} = [\partial_x, \; \partial_z \, ]^T` is a partial
    gradient operator that acts along the x and z directions.

    Using mimetic finite volume, the 2D problem can be solved numerically by solving
    the following linear system:

    .. math::
        \big [ \mathbf{G^T \, M_{e\sigma} \, G} + k_y^2 \, \mathbf{M_{n\sigma}} \big ]
        \boldsymbol{\Phi} = \mathbf{q}

    where

    * :math:`\boldsymbol{\Phi}` are the discrete electric potentials defined on mesh nodes
    * :math:`G` is the 2D nodal gradient operator
    * :math:`M_{e\sigma}` is the conductivity inner-product matrix on mesh edges
    * :math:`M_{n\sigma}` is the conductivity inner-product matrix on mesh nodes

    For an optimum set of wavenumbers :math:`k_y^{(i)}` and coefficients :math:`\alpha^{(i)}`,
    we solve a set of discrete 2D problems in the wave domain. And the full 3D solution
    for a specified y-coordinate location is computed according to:

    .. math::
        \boldsymbol{\phi} = \sum_{i=1}^{nk_y}
        \alpha^{(i)} \, \boldsymbol{\Phi} \! \left ( k_y^{(i)} \right )

    where :math:`nk_y` is the number of wavenumbers used to compute the solution.

    **Axial Anisotropy:**

    In this case, the DC resistivity problem is defined according to:

    .. math::
        \nabla \cdot \Sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where

    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_x & 0 & 0 \\ 0 & \sigma_y & 0 \\ 0 & 0 & \sigma_z
        \end{bmatrix}

    The discrete 2D problem solved in the wave domain of y still takes the form:

    .. math::
        \big [ \mathbf{G^T \, M_{e\sigma} \, G} + k_y^2 \, \mathbf{M_{n\sigma}} \big ]
        \boldsymbol{\Phi} = \mathbf{q}

    However,

    * :math:`M_{e\sigma}` is a conductivity inner-product matrix on cell edges constructed using axial conductivities :math:`\sigma_x` and :math:`\sigma_z`
    * :math:`M_{n\sigma}` is the conductivity inner-product matrix at mesh nodes constructed using axial conductivity :math:`\sigma_y`
    """

    _solutionType = "phiSolution"
    _formulation = "EB"  # CC potentials means J is on faces
    fieldsPair = Fields2DNodal
    fieldsPair_fwd = Fields3DNodal
    _gradT = None

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.solver_opts["is_symmetric"] = True
        self.solver_opts["is_positive_definite"] = True
        self.bc_type = bc_type

    @property
    def bc_type(self):
        """Type of boundary condition to use for simulation.

        The boundary conditions supported by the :class:`Simulation2DNodal` are:

        * "Neumann": Zero Neumann on the boundary
        * "Robin" or "Mixed": Mix of zero Dirichlet and zero Neumann.

        Returns
        -------
        {"Neumann", "Robin", "Mixed"}

        Notes
        -----
        "Robin" and "Mixed" are equivalent.
        """
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value):
        self._bc_type = validate_string(
            "bc_type", value, ["Neumann", ("Robin", "Mixed")]
        )

    @property
    def MeSigma(self):
        r"""Conductivity inner-product matrix on edges.

        Where the inner product between a vector :math:`\vec{u}` and the electrical
        conductivity :math:`\sigma` times a vector :math:`\vec{v}` can be approximated by a
        discrete operation on mesh edges:

        .. math::
            \langle \vec{u}, \sigma \vec{v} \rangle \approx
            \mathbf{u^T M_{e \sigma} \, v}

        this property returns the the inner-product matrix :math:`\mathbf{M_{e \sigma}}`.

        Returns
        -------
        (n_edges, n_edges) scipy.sparse.csr_matrix
            The the inner-product matrix.
        """

        # Isotropic case
        if self.sigma.size == self.mesh.nC:
            return super().MeSigma

        # Override for anisotropic case
        stash_name = "_Me_Sigma"
        if getattr(self, stash_name, None) is None:
            if self.sigma.size == 3 * self.mesh.nC:
                model = self._Pxz_from_xyz * self.sigma
                M_prop = self.mesh.get_edge_inner_product(model=model)
                setattr(self, stash_name, M_prop)
            else:
                raise NotImplementedError(
                    "Only isotropic and linear isotropic conductivities implemented."
                )

        return getattr(self, stash_name)

    def MeSigmaDeriv(self, u, v=None, adjoint=False):
        r"""Derivative operation for the conductivity inner-product matrix on edges.

        Let :math:`\mathbf{M_{e\sigma}}` represent the conductivity inner-product matrix on mesh edges.
        Where :math:`\mathbf{u}` and :math:`\mathbf{v}` are vectors, this method returns

        .. math::
            \frac{\partial (\mathbf{M_{e\sigma} \, u} )}{\partial \mathbf{m}} \, \mathbf{v}

        or the adjoint operation

        .. math::
            \left ( \frac{\partial (\mathbf{M_{e\sigma} \, u})}{\partial \mathbf{m}} \, \right )^T \mathbf{v}

        If the input argument *v* is ``None``, the method will return a function handle that
        acts on an input argument *v*.

        Parameters
        ----------
        u : (n_nodes,) numpy.ndarray
            A vector.
        v : numpy.ndarray, optional
            A vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative operation for the conductivity inner-product matrix. (n_edges,) for the
            standard operation. (n_param,) for the adjoint operation.
        """

        # Isotropic case
        if self.sigma.size == self.mesh.nC:
            return super().MeSigmaDeriv(u, v, adjoint)

        # Override for anisotropic case
        if self.sigmaMap is None:
            return Zero()
        if isinstance(u, Zero) or isinstance(v, Zero):
            return Zero()
        stash_name = "_Me_Sigma_deriv"

        if getattr(self, stash_name, None) is None:
            sigma = self.sigma
            nC = self.mesh.nC

            if sigma.size == 3 * nC:
                if self.mesh._meshType.lower() in (
                    "cyl",
                    "tensor",
                    "tree",
                ):
                    # Edge inner product derivative for x and z axial conductivitites
                    M_deriv_func = self.mesh.get_edge_inner_product_deriv(
                        model=self._Pxz_from_xyz * sigma
                    )

                    # Derivative wrt all axial conductivities
                    prop_deriv = self.sigmaDeriv

                    M_prop_deriv = (
                        M_deriv_func(np.ones(self.mesh.n_edges))
                        @ self._Pxz_from_xyz
                        @ prop_deriv
                    )
                    setattr(self, stash_name, M_prop_deriv)
                else:
                    raise NotImplementedError(
                        "Only implemented for 'CylindricalMesh', 'TensorMesh' and 'TreeMesh'."
                    )

            else:
                raise NotImplementedError(
                    "Only isotropic and axial anisotropic conductivities implemented."
                )

        return inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

    @property
    def MnSigma(self):
        r"""Conductivity inner-product matrix at mesh nodes.

        Where the inner-product between a scalar :math:`\phi` and the electrical
        conductivity :math:`\sigma` times a scalar :math:`\psi` can be approximated by a
        discrete operation at mesh nodes:

        .. math::
            \langle \phi, \, \sigma \psi \rangle \approx
            \mathbf{phi^T M_{n \sigma} \, \psi}

        this property returns the inner-product matrix :math:`\mathbf{M_{n \sigma}}`.

        Returns
        -------
        (n_nodes, n_nodes) scipy.sparse.csr_matrix
            The the inner-product matrix.
        """

        # Isotropic case
        if self.sigma.size == self.mesh.nC:
            return super().MnSigma

        # Override if anisotropic case
        stash_name = "_Mn_Sigma"
        if getattr(self, stash_name, None) is None:
            if self.sigma.size == 3 * self.mesh.nC:
                vol = self.mesh.cell_volumes
                M_prop = sp.diags(
                    self.mesh.aveN2CC.T * (vol * (self._Py_from_xyz * self.sigma)),
                    format="csr",
                )
                setattr(self, stash_name, M_prop)
            else:
                raise NotImplementedError(
                    "Only isotropic and linear isotropic conductivities implemented."
                )

        return getattr(self, stash_name)

    def MnSigmaDeriv(self, u, v=None, adjoint=False):
        r"""Derivative operation for the conductivity inner-product matrix on nodes.

        Let :math:`\mathbf{M_{n\sigma}}` represent the conductivity inner-product matrix on mesh nodes.
        Where :math:`\mathbf{u}` and :math:`\mathbf{v}` are vectors, this method returns

        .. math::
            \frac{\partial (\mathbf{M_{n\sigma} \, u} )}{\partial \mathbf{m}} \, \mathbf{v}

        or the adjoint operation

        .. math::
            \left ( \frac{\partial (\mathbf{M_{n\sigma} \, u})}{\partial \mathbf{m}} \, \right )^T \mathbf{v}

        If the input argument *v* is ``None``, the method will return a function handle that
        acts on an input argument *v*.

        Parameters
        ----------
        u : (n_nodes,) numpy.ndarray
            A vector.
        v : numpy.ndarray, optional
            A vector. (n_param,) for the standard operation. (n_nodes,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative operation for the conductivity inner-product matrix. (n_nodes,) for the
            standard operation. (n_param,) for the adjoint operation.
        """

        # Isotropic case
        if self.sigma.size == self.mesh.nC:
            return super().MnSigmaDeriv(u, v, adjoint)

        # Override for anisotropic case
        if self.sigmaMap is None:
            return Zero()
        if isinstance(u, Zero) or isinstance(v, Zero):
            return Zero()

        stash_name = "_Mn_Sigma_deriv"
        if getattr(self, stash_name, None) is None:
            sigma = self.sigma
            nC = self.mesh.nC

            if sigma.size == 3 * nC:
                M_prop_deriv = (
                    self.mesh.aveN2CC.T
                    * sp.diags(self.mesh.cell_volumes)
                    * self._Py_from_xyz
                    * self.sigmaDeriv
                )
                setattr(self, stash_name, M_prop_deriv)

            else:
                raise NotImplementedError(
                    "Only isotropic and axial anisotropic conductivities implemented."
                )

        return inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

    def getA(self, ky):
        r"""Compute the system matrix for the wavenumber provided.

        The discrete solution to the 2D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\Phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\boldsymbol{\Phi}` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. This method returns the system matrix
        for the nodal formulation for the wavenumber :math:`k_y` provided, i.e.:

        .. math::
            \mathbf{A} = \mathbf{G^T \, M_{e\sigma} \, G} + k_y^2 \mathbf{M_{n\sigma}}

        where :math:`\mathbf{G}` is the nodal gradient operator with imposed boundary conditions,
        :math:`\mathbf{M_{e\sigma}}` is the inner product matrix for conductivities projected to edges,
        and :math:`\mathbf{M_{n\sigma}}` is the inner product matrix for conductivities
        at projected to nodes.

        Parameters
        ----------
        ky : float
            The wavenumber.

        Returns
        -------
        (n_nodes, n_nodes) scipy.sparse.csr_matrix
            The sparse system matrix.
        """
        # To handle Mixed boundary condition
        self.setBC(ky=ky)

        MeSigma = self.MeSigma
        MnSigma = self.MnSigma
        Grad = self.mesh.nodal_gradient
        if self._gradT is None:
            self._gradT = Grad.T.tocsr()  # cache the .tocsr()
        GradT = self._gradT
        A = GradT * MeSigma * Grad + ky**2 * MnSigma

        if self.bc_type != "Neumann":
            try:
                A = A + sdiag(self._AvgBC[ky] @ self.sigma)
            except ValueError as err:
                if len(self.sigma) != len(self.mesh):
                    raise NotImplementedError(
                        "Anisotropic conductivity is not supported for Robin boundary "
                        "conditions, please use 'Neumann'."
                    )
                else:
                    raise err
        return A

    def getADeriv(self, ky, u, v, adjoint=False):
        r"""Derivative operation for the system matrix times a vector.

        The discrete solution to the 2D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A \, u} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\mathbf{u}` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. For a vector :math:`\mathbf{v}`, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A \, u})}{\partial \mathbf{m}} \, \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A \, u})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        ky : float
            The wavenumber.
        u : (n_nodes,) numpy.ndarray
            The solution for the fields for the current model; i.e. electric potentials at cell nodes.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_nodes,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_nodes,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        Grad = self.mesh.nodal_gradient

        if adjoint:
            out = self.MeSigmaDeriv(
                Grad * u, Grad * v, adjoint=adjoint
            ) + ky**2 * self.MnSigmaDeriv(u, v, adjoint=adjoint)

        else:
            out = Grad.T * self.MeSigmaDeriv(
                Grad * u, v, adjoint=adjoint
            ) + ky**2 * self.MnSigmaDeriv(u, v, adjoint=adjoint)

        if self.bc_type != "Neumann" and self.sigmaMap is not None:
            if getattr(self, "_MBC_sigma", None) is None:
                self._MBC_sigma = {}
            if ky not in self._MBC_sigma:
                self._MBC_sigma[ky] = self._AvgBC[ky] @ self.sigmaDeriv
            if not isinstance(u, Zero):
                u = u.flatten()
                if v.ndim > 1:
                    u = u[:, None]
                if not adjoint:
                    out += u * (self._MBC_sigma[ky] @ v)
                else:
                    out += self._MBC_sigma[ky].T @ (u * v)
        return out

    def getRHS(self, ky):
        r"""Compute the source terms for the wavenumber provided.

        For a single source, the discrete solution to the 2D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\phi` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side corresponding to the source term.
        This method computes and returns an array :math:`\mathbf{Q}`, whose columns are
        the right-hand sides for all sources.

        Parameters
        ----------
        ky : float
            The wavenumber.

        Returns
        -------
        (n_nodes, nSrc) numpy.ndarray
            The array containing the right-hand sides for all sources for
            the wavenumber specified.
        """

        RHS = self.getSourceTerm(ky)
        return RHS

    def getRHSDeriv(self, ky, src, v, adjoint=False):
        r"""Derivative of the source term with respect to the model times a vector.

        The discrete solution to the 2D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\Phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\Phi` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side. This method returns the derivative
        of the right-hand side with respect to the model times a vector, i.e.:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}} \mathbf{v}

        or the adjoint operation:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}}^T \mathbf{v}

        Parameters
        ----------
        ky : float
            The wavenumber.
        src : SimPEG.electromagnetic.static.resistivity.sources.BaseSrc
            The source object.
        v : numpy.ndarray
            The vector. Has shape (n_param,) when performing the standard derivative operation.
            Has shape (n_nodes,) when performing the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        Zero or numpy.ndarray
            Returns :py:class:`Zero` if the derivative with respect to the model is zero.
            Returns (n_nodes,) :class:`numpy.ndarray` when computing the standard
            derivative operation. Returns (n_param,) :class:`numpy.ndarray` when performing
            the adjoint.
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = src.evalDeriv(self, ky, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def setBC(self, ky=None):
        """Sets the boundary conditions on the nodal gradient operator.

        This method will set the boundary conditions on the nodal gradient
        operator based on the value of the :py:attr:`bc_type` property.
        The options are:

        * "Neumann": Zero Neumann on the boundary (natural boundary conditions).
        * "Robin" or "Mixed": Mix of Robin and zero Neumann boundary conditions.

        Parameters
        ----------
        ky : float (optional)
            The wavenumber.
        """
        if self.bc_type == "Dirichlet":
            # do nothing
            raise ValueError(
                "Dirichlet conditions are not supported in the Nodal formulation"
            )
        elif self.bc_type == "Neumann":
            if self.verbose:
                print(
                    "Homogeneous Neumann is the natural BC for this Nodal discretization."
                )
            return
        else:
            if getattr(self, "_AvgBC", None) is None:
                self._AvgBC = {}
            if ky in self._AvgBC:
                return
            mesh = self.mesh
            # calculate alpha, beta, gamma at the boundary faces
            boundary_faces = mesh.boundary_faces
            boundary_normals = mesh.boundary_face_outward_normals
            n_bf = len(boundary_faces)

            alpha = np.zeros(n_bf)

            # assume a source point at the middle of the top of the mesh
            middle = np.median(mesh.nodes, axis=0)
            top_v = np.max(mesh.nodes[:, -1])
            source_point = np.r_[middle[:-1], top_v]

            r_vec = boundary_faces - source_point
            r = np.linalg.norm(r_vec, axis=-1)
            r_hat = r_vec / r[:, None]
            r_dot_n = np.einsum("ij,ij->i", r_hat, boundary_normals)

            if self.surface_faces is None:
                # determine faces that are on the sides and bottom of the mesh...
                if mesh._meshType.lower() == "tree":
                    not_top = boundary_faces[:, -1] != top_v
                elif mesh._meshType.lower() in ["tensor", "curv"]:
                    # mesh faces are ordered, faces_x, faces_y, faces_z so...
                    is_b = make_boundary_bool(mesh.shape_faces_y)
                    is_t = np.zeros(mesh.shape_faces_y, dtype=bool, order="F")
                    is_t[:, -1] = True
                    is_t = is_t.reshape(-1, order="F")[is_b]
                    not_top = np.ones(boundary_faces.shape[0], dtype=bool)
                    not_top[-len(is_t) :] = ~is_t
                else:
                    raise NotImplementedError(
                        f"Unable to infer surface boundaries for {type(mesh)}, please "
                        f"set the `surface_faces` property."
                    )
            else:
                not_top = ~self.surface_faces

            # use the exponentiall scaled modified bessel function of second kind,
            # (the division will cancel out the scaling)
            # This is more stable for large values of ky * r
            # actual ratio is k1/k0...
            alpha[not_top] = (ky * k1e(ky * r) / k0e(ky * r) * r_dot_n)[not_top]

            P_bf = self.mesh.project_face_to_boundary_face

            AvgN2Fb = P_bf @ self.mesh.average_node_to_face
            AvgCC2Fb = P_bf @ self.mesh.average_cell_to_face

            AvgCC2Fb = sdiag(alpha * (P_bf @ self.mesh.face_areas)) @ AvgCC2Fb
            self._AvgBC[ky] = AvgN2Fb.T @ AvgCC2Fb


Simulation2DCellCentred = Simulation2DCellCentered  # UK and US
