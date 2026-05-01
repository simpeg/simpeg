"""3D DC resistivity simulation classes."""

import numpy as np
import scipy.sparse as sp

from ....utils import (
    mkvc,
    Zero,
    validate_type,
    validate_string,
    validate_active_indices,
)
from ....data import Data
from ....base import BaseElectricalPDESimulation, BaseHierarchicalElectricalSimulation
from ....base.pde_simulation import _inner_mat_mul_op
from .survey import Survey
from .fields import Fields3DCellCentered, Fields3DNodal
from .utils import _mini_pole_pole
from discretize.utils import make_boundary_bool


class BaseDCSimulation(BaseElectricalPDESimulation):
    r"""Base class for 3D DC resistivity simulation.

    This class is used to define properties and methods necessary for solving the
    3D direct current resistivity problem using mimetic finite volume. The PDE we are
    solving is given by:

    .. math::
        \nabla \cdot \sigma \nabla \phi = - I \delta (r)

    where we are solving for the electric potential :math:`\phi` is the electric potential.
    The electrical conductivity is given by :math:`\sigma`, and :math:`I \delta (r)`
    represents current *I* injected at point *r*.
    Child classes of ``BaseDCSimulation`` solve the above expression numerically
    for various cases using mimetic finite volume.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : .resistivity.survey.Survey
        The DC resistivity survey.
    """

    _mini_survey = None

    Ainv = None

    def __init__(  # noqa D107
        self,
        mesh,
        survey=None,
        storeJ=False,
        miniaturize=False,
        surface_faces=None,
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.storeJ = storeJ
        self.surface_faces = surface_faces
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
        simpeg.electromagnetics.static.resistivity.survey.Survey
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
    def surface_faces(self):
        """Return which boundary faces to interpret as surfaces of Neumann boundary.

        DC problems will always enforce a Neumann boundary on surface interfaces.
        The default (available on semi-structured grids) assumes the top interface
        is the surface.

        Returns
        -------
        None or (n_bf,) numpy.ndarray of bool
        """
        return self._surface_faces

    @surface_faces.setter
    def surface_faces(self, value):
        if value is not None:
            n_bf = self.mesh.boundary_faces.shape[0]
            value = validate_active_indices("surface_faces", value, n_bf)
        self._surface_faces = value

    def fields(self, m=None, calcJ=True):
        """Compute and return the fields for the model provided.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model.

        Returns
        -------
        .resistivity.fields.FieldsDC
            The DC fields object.
        """
        if m is not None:
            self.model = m

        f = self.fieldsPair(self)
        if self.Ainv is not None:
            self.Ainv.clean()
        A = self.getA()
        self.Ainv = self.solver(A, **self.solver_opts)
        RHS = self.getRHS()

        f[:, self._solutionType] = self.Ainv * RHS

        return f

    def getJ(self, m, f=None):
        r"""Compute the sensitivity matrix for a given model.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the sensitivity matrix.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        f : .resistivity.fields.FieldsDC, optional
            Fields solved for all sources.

        Returns
        -------
        (n_data, n_param) numpy.ndarray
            The sensitivity matrix times.
        """
        self.model = m
        if getattr(self, "_Jmatrix", None) is None:
            if f is None:
                f = self.fields(m)
            self._Jmatrix = self._Jtvec(m, v=None, f=f).T
        return self._Jmatrix

    def dpred(self, m=None, f=None):  # noqa D102
        # Docstring inherited from parent class.
        if self._mini_survey is not None:
            # Temporarily set self.survey to self._mini_survey
            survey = self.survey
            self.survey = self._mini_survey

        data = super().dpred(m=m, f=f)

        if self._mini_survey is not None:
            # reset survey
            self.survey = survey

        return self._mini_survey_data(data)

    def getJtJdiag(self, m, W=None, f=None):
        r"""Return the diagonal of JtJ.

        Where :math:`\mathbf{d}` are the data, :math:`\mathbf{m}` are the model parameters,
        and the sensitivity matrix is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        this method computes and returns the diagnal elements
        of :math:`\mathbf{J^T J}`.

        Parameters
        ----------
        m : (n_param,) numpy.ndarray
            The model parameters.
        W : (n_param,) numpy.ndarray, optional
            Cell weights.
        f : .resistivity.fields.FieldsDC, optional
            Fields solved for all sources.

        Returns
        -------
        (n_param,) numpy.ndarray
            The diagonal of :math:`\mathbf{J^T J}`.
        """
        if getattr(self, "_gtgdiag", None) is None:
            J = self.getJ(m, f=f)

            if W is None:
                W = np.ones(J.shape[0])
            else:
                W = W.diagonal() ** 2

            diag = np.zeros(J.shape[1])
            for i in range(J.shape[0]):
                diag += (W[i]) * (J[i] * J[i])

            self._gtgdiag = diag
        return self._gtgdiag

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
        f : .resistivity.fields.FieldsDC, optional
            Fields solved for all sources.

        Returns
        -------
        (n_data,) numpy.ndarray
            The sensitivity matrix times a vector.
        """
        if f is None:
            f = self.fields(m)

        self.model = m

        if self.storeJ:
            J = self.getJ(m, f=f)
            return J.dot(v)

        self.model = m

        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        Jv = []
        for source in survey.source_list:
            u_source = f[source, self._solutionType]  # solution vector
            dA_dm_v = self.getADeriv(u_source, v)
            dRHS_dm_v = self.getRHSDeriv(source, v)
            du_dm_v = self.Ainv * (-dA_dm_v + dRHS_dm_v)
            for rx in source.receiver_list:
                df_dmFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                df_dm_v = df_dmFun(source, du_dm_v, v, adjoint=False)
                Jv.append(rx.evalDeriv(source, self.mesh, f, df_dm_v))
        Jv = np.hstack(Jv)
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
        f : .resistivity.fields.FieldsDC, optional
            Fields solved for all sources.

        Returns
        -------
        (n_param,) numpy.ndarray
            The adjoint sensitivity matrix times a vector.
        """
        if f is None:
            f = self.fields(m)

        self.model = m

        if self.storeJ:
            J = self.getJ(m, f=f)
            return np.asarray(J.T.dot(v))

        return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        This method does the actual computation of J-transpose times a vector.
        Or when *v* = ``None``, it returns the full transpose of the sensitivity matrix.
        """
        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        if v is not None:
            if isinstance(v, Data):
                v = v.dobs
            v = self._mini_survey_dataT(v)
            Jtv = np.zeros(m.size)
        else:
            # This is for forming full sensitivity matrix
            Jtv = np.zeros((self.model.size, survey.nD), order="F")
            istrt = int(0)
            iend = int(0)

        # Get dict of flat array slices for each source-receiver pair in the survey
        survey_slices = survey.get_all_slices()

        for source in survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                # wrt f, need possibility wrt m
                if v is not None:
                    src_rx_slice = survey_slices[source, rx]
                    PTv = rx.evalDeriv(
                        source, self.mesh, f, v[src_rx_slice], adjoint=True
                    )
                else:
                    PTv = rx.evalDeriv(source, self.mesh, f).toarray().T

                df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                df_duT, df_dmT = df_duTFun(source, None, PTv, adjoint=True)

                ATinvdf_duT = self.Ainv * df_duT

                dA_dmT = self.getADeriv(u_source, ATinvdf_duT, adjoint=True)
                dRHS_dmT = self.getRHSDeriv(source, ATinvdf_duT, adjoint=True)
                du_dmT = -dA_dmT + dRHS_dmT
                if v is not None:
                    Jtv += (df_dmT + du_dmT).astype(float)
                else:
                    iend = istrt + rx.nD
                    if rx.nD == 1:
                        Jtv[:, istrt] = df_dmT + du_dmT
                    else:
                        Jtv[:, istrt:iend] = df_dmT + du_dmT
                    istrt += rx.nD

        if v is not None:
            return mkvc(Jtv)
        else:
            return (self._mini_survey_data(Jtv.T)).T

    def getSourceTerm(self):
        r"""Return the discrete source term for all sources.

        Returns
        -------
        numpy.ndarray
            The source terms for number of unique source locations.
            (n_cells, n_sources) for cell centered formulations.
            (n_nodes, n_sources) for nodal formulations.
        """
        if getattr(self, "_q", None) is None:
            if self._mini_survey is not None:
                Srcs = self._mini_survey.source_list
            else:
                Srcs = self.survey.source_list

            if self._formulation == "EB":
                n = self.mesh.nN

            elif self._formulation == "HJ":
                n = self.mesh.nC

            q = np.zeros((n, len(Srcs)), order="F")

            for i, source in enumerate(Srcs):
                q[:, i] = source.eval(self)
            self._q = q
        return self._q

    @property
    def _delete_on_model_update(self):
        """List of model-dependent attributes to clean upon model update.

        Some of the simulation's attributes are model-dependent. This property specifies
        the model-dependent attributes that much be cleared when the model is updated.

        Returns
        -------
        list of str
            List of the model-dependent attributes to clean upon model update.
        """
        toDelete = super()._delete_on_model_update
        return toDelete + ["_Jmatrix", "_gtgdiag"]

    def _mini_survey_data(self, d_mini):
        """Get mini survey data."""
        if self._mini_survey is not None:
            out = d_mini[self._invs[0]]  # AM
            out[self._dipoles[0]] -= d_mini[self._invs[1]]  # AN
            out[self._dipoles[1]] -= d_mini[self._invs[2]]  # BM
            out[self._dipoles[0] & self._dipoles[1]] += d_mini[self._invs[3]]  # BN
        else:
            out = d_mini
        return out

    def _mini_survey_dataT(self, v):
        """Get transpose of mini survey data."""
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


class Simulation3DCellCentered(BaseDCSimulation):
    r"""3D cell centered DC resistivity simulation class.

    Where :math:`\phi` is the electric potential, :math:`\rho` is the electrical resistivity,
    and :math:`I \delta (r)` represents a grounded source which injects static current *I* at
    location *r*, the 3D DC resistivity problem is define according to the following PDE:

    .. math::
        \nabla \cdot \frac{1}{\rho} \nabla \phi = - I \delta (r)

    The ``Simulation3DCellCentered`` class uses the mimetic finite volume approach to solve for
    the discrete electric potentials at cell centers. See the *Notes* section for a
    comprehensive description of the formulation.

    Notes
    -----
    To derive the discrete solution, we start by considering Ampere's law and Faraday's law.
    In the static regime, all time-derivative are zero. By also taking the divergence of
    Ampere's law, we obtain:

    .. math::
        &\nabla \times \vec{e} = 0\\
        &\nabla \cdot \vec{j} = - \nabla \cdot \vec{j}_s

    where :math:`\vec{e}` is the electric field, :math:`\vec{j}` is the current density outside
    the grounded source and and :math:`\vec{j}_s` is the source current density. The constitutive
    relation between the electric field and current density is given by Ohm's law:

    .. math::
        \vec{e} = \rho \vec{j}

    where :math:`\rho` is the electrical resistivity.

    Faraday's law implies the electric field can be defined as the gradient of a scalar
    potential :math:`\phi` as follows:

    .. math::
        \phi = -\nabla \phi

    For a source representing current *I* injected at point *r*, the static form of Ampere's
    law becomes:

    .. math::
        \nabla \cdot \vec{j} = - I \delta (r)

    For a vector test function :math:`\vec{u}` and a scalar test function :math:`\psi`, we
    take the inner products with the three previous equations. Through vector calculus
    identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega \vec{u} \cdot \vec{e} \, dv
        = \int_\Omega \vec{u} \cdot \rho \vec{j} \, dv \\
        & \int_\Omega \psi ( \nabla \cdot \vec{j} ) \, dv =
        - I \int_\Omega \psi \, \delta (r) \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{e} \, dv =
        \int_\Omega (\nabla \cdot \vec{u}) \phi \, dv
        - \oint_{\partial \Omega} (\vec{u} \cdot \hat{n} ) \, \phi \, da

    where the surface integral defines the boundary conditions.

    The above expressions are discretized in space according to the finite volume method.
    The discrete current densities :math:`\mathbf{j}` are defined on mesh faces,
    and the discrete electric potential :math:`\boldsymbol{\phi}` are defined at cell centers.
    This implies :math:`\mathbf{u}` and :math:`\mathbf{e}` must be defined on mesh faces,
    and :math:`\boldsymbol{\psi}` must be defined at cell centers. We obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u^T M_f \, e} = \mathbf{u^T M_{f\rho} \, j} \\
        &\boldsymbol{\psi^T} \mathbf{M_c D j} = \boldsymbol{\psi^T} \mathbf{q} \\
        &\mathbf{u^T M_f e} = \mathbf{u^T (D^T - B ) M_c} \boldsymbol{\phi}

    where

    * :math:`\mathbf{D}` is the divergence operator (faces to cell centers)
    * :math:`\mathbf{q}` is an integrated source term
    * :math:`\mathbf{B}` is a matrix that implements boundary conditions
    * :math:`\mathbf{M_c}` is the cell inner-product matrix
    * :math:`\mathbf{M_f}` is the face inner-product matrix
    * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities
      projected to faces

    By combining the discrete equations to form a discrete solution in terms of the
    discrete electric potential on mesh nodes, we obtain:

    .. math::
        \big [ \mathbf{M_c D M_{r\rho}^{-1} (D^T - B ) M_c} \big ] \boldsymbol{\phi}
        = \mathbf{q}

    Note that :math:`\mathbf{D^T - B}` is effectively a gradient operator that has been
    modified to implement the boundary conditions.

    """

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):  # noqa D107
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.bc_type = bc_type
        self.setBC()

    @property
    def bc_type(self):
        """Type of boundary condition to use for simulation.

        Returns
        -------
        {"Dirichlet", "Neumann", "Robin", "Mixed"}

        Notes
        -----
        Robin and Mixed are equivalent.
        """
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value):
        self._bc_type = validate_string(
            "bc_type", value, ["Dirichlet", "Neumann", ("Robin", "Mixed")]
        )

    def getA(self, resistivity=None):
        r"""Return the system matrix.

        This method generates and returns the system matrix for the cell-centered DC
        resistivity problem. The system matrix is given by:

        .. math::
            \mathbf{A} = \mathbf{M_c D M_{f\rho} (G - B) M_c}

        where

        * :math:`\mathbf{D}` is the divergence operator (faces to cell centers)
        * :math:`\mathbf{B}` is a matrix that implements boundary conditions
        * :math:`\mathbf{M_c}` is the cell inner-product matrix
        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities
          projected to faces

        Parameters
        ----------
        resistivity : (n_cells,) numpy.ndarray
            Electrical resistivities defined at cell centers

        Returns
        -------
        (n_cells, n_cells) sp.sparse.csr_matrix
            The system matrix.
        """
        D = self.Div
        G = self.Grad
        if resistivity is None:
            MfRhoI = self.MfRhoI
        else:
            MfRhoI = self.mesh.get_face_inner_product(resistivity, invert_matrix=True)
        A = D @ MfRhoI @ G

        if self.bc_type == "Neumann":
            if self.verbose:
                print("Perturbing first row of A to remove nullspace for Neumann BC.")

            # Handling Null space of A
            I, J, V = sp.find(A[0, :])
            for jj in J:
                A[0, jj] = 0.0
            A[0, 0] = 1.0

        return A

    def getADeriv(self, u, v, adjoint=False):
        r"""Get cnductivity derivative operation for the system matrix times a vector.

        The system matrix is given by:

        .. math::
            \mathbf{A} = \mathbf{M_c D M_{f\rho} (G - B) M_c}

        where

        * :math:`\mathbf{D}` is the divergence operator (faces to cell centers)
        * :math:`\mathbf{B}` is a matrix that implements boundary conditions
        * :math:`\mathbf{M_c}` is the cell inner-product matrix
        * :math:`\mathbf{M_{f\rho}}` is the inner-product matrix for resistivities
          projected to faces

        See the *Notes* section of the doc strings for :class:`Simulation3DCellCentered`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters defining the
        electromagnetic properties :math:`\mathbf{v}` is a vector and
        :math:`\boldsymbol{\phi}` is the discrete electric potential solution, this
        method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\phi})}{\partial \mathbf{m}} \,
            \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\phi})}{\partial \mathbf{m}}^T
            \, \mathbf{v}

        Parameters
        ----------
        u : (n_cells,) numpy.ndarray
            The solution for the electric potentials.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_cells,) for the adjoint
            operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_cells,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        if self.rhoMap is not None:
            D = self.Div
            G = self.Grad
            MfRhoIDeriv = self.MfRhoIDeriv

            if adjoint:
                return MfRhoIDeriv(G @ u, D.T @ v, adjoint)

            return D * (MfRhoIDeriv(G @ u, v, adjoint))
        return Zero()

    def getRHS(self):
        """Compute and return right-hand sides for all sources.

        For This method computes and returns the right-hand sides used to solve the
        discrete finite volume solution for all sources.

        Returns
        -------
        (n_cells, n_sources) numpy.ndarray
            The right-hand sides.
        """
        RHS = self.getSourceTerm()

        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        """Get derivative of the right-hand side with respect to the model.

        For ``Simulation3DCellCentered``, the derivative of the right-hand side with respect
        to the model is zero.

        Returns
        -------
        simpeg.utils.Zero
            The SimPEG zero operator.
        """
        return Zero()

    def setBC(self):
        """Set the boundary conditions for the gradient operator."""
        mesh = self.mesh
        V = sp.diags(mesh.cell_volumes)
        self.Div = V @ mesh.face_divergence
        self.Grad = self.Div.T

        if self.bc_type == "Dirichlet":
            if self.verbose:
                print(
                    "Homogeneous Dirichlet is the natural BC for this CC discretization."
                )
            # do nothing
            return
        elif self.bc_type == "Neumann":
            alpha, beta, gamma = 0, 1, 0
        else:
            boundary_faces = mesh.boundary_faces
            boundary_normals = mesh.boundary_face_outward_normals
            n_bf = len(boundary_faces)

            # Top gets 0 Nuemann
            alpha = np.zeros(n_bf)
            beta = np.ones(n_bf)
            gamma = 0

            # assume a source point at the middle of the top of the mesh
            middle = np.median(mesh.nodes, axis=0)
            top_v = np.max(mesh.nodes[:, -1])
            source_point = np.r_[middle[:-1], top_v]

            # Others: Robin: alpha * phi + d phi dn = 0
            # where alpha = 1 / r  * r_hat_dot_n
            # TODO: Implement Zhang et al. (1995)

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
                    if mesh.dim == 2:
                        is_b = make_boundary_bool(mesh.shape_faces_y)
                        is_t = np.zeros(mesh.shape_faces_y, dtype=bool, order="F")
                        is_t[:, -1] = True
                    else:
                        is_b = make_boundary_bool(mesh.shape_faces_z)
                        is_t = np.zeros(mesh.shape_faces_z, dtype=bool, order="F")
                        is_t[:, :, -1] = True
                    is_t = is_t.reshape(-1, order="F")[is_b]
                    not_top = np.ones(boundary_faces.shape[0], dtype=bool)
                    not_top[-len(is_t) :] = ~is_t  # noqa E203
                    self.surface_faces = ~not_top
                else:
                    raise NotImplementedError(
                        f"Unable to infer surface boundaries for {type(mesh)}, please "
                        f"set the `surface_faces` property."
                    )
            else:
                not_top = ~self.surface_faces
            alpha[not_top] = (r_dot_n / r)[not_top]

        B, bc = mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
        # bc should always be 0 because gamma was always 0 above
        self.Grad = self.Grad - B


class Simulation3DNodal(BaseDCSimulation):
    r"""3D nodal DC resistivity simulation class.

    Where :math:`\phi` is the electric potential, :math:`\sigma` is the electrical conductivity,
    and :math:`I \delta (r)` represents a grounded source which injects static current *I* at
    location *r*, the DC resistivity problem is given by:

    .. math::
        \nabla \cdot \sigma \nabla \phi = - I \delta (r)

    The ``Simulation3DNodal`` class uses the mimetic finite volume approach to solve for
    the discrete electric potentials at mesh nodes. See the *Notes* section for a
    comprehensive description of the formulation.

    Notes
    -----
    To derive the discrete solution, we start by considering Ampere's law and Faraday's law.
    In the static regime, all time-derivative are zero. By also taking the divergence of
    Ampere's law, we obtain:

    .. math::
        &\nabla \times \vec{e} = 0\\
        &\nabla \cdot \vec{j} = - \nabla \cdot \vec{j}_s

    where :math:`\vec{e}` is the electric field, :math:`\vec{j}` is the current density outside
    the grounded source and and :math:`\vec{j}_s` is the source current density. The constitutive
    relation between the electric field and current density is given by Ohm's law:

    .. math::
        \vec{j} = \sigma \vec{e}

    where :math:`\sigma` is the electrical conductivity.

    Faraday's law implies the electric field can be defined as the gradient of a scalar
    potential :math:`\phi` as follows:

    .. math::
        \phi = -\nabla \phi

    For a source representing current *I* injected at point *r*, the static form of Ampere's
    law becomes:

    .. math::
        \nabla \cdot \vec{j} = - I \delta (r)

    For a vector test function :math:`\vec{u}` and a scalar test function :math:`\psi`, we
    take the inner products with the three previous equations. Through vector calculus
    identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega \vec{u} \cdot \vec{j} \, dv =
        \int_\Omega \vec{u} \cdot \sigma \vec{e} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{e} \, dv =
        - \int_\Omega \vec{u} \cdot \nabla \phi \, dv \\
        & -\int_\Omega \nabla \psi \cdot \vec{j} \, dv
        +\oint_{\partial \Omega} \psi (\vec{j} \cdot \hat{n}) \, da =
        -I \int_\Omega \psi \, \delta (r) dv

    where the surface integral defines the boundary conditions.

    The above expressions are discretized in space according to the finite volume method.
    The discrete electric fields :math:`\mathbf{e}` are defined on mesh edges,
    and the discrete electric potentials :math:`\boldsymbol{\phi}` are defined on nodes.
    This implies :math:`\mathbf{j}` and :math:`\mathbf{u}` must be defined on mesh edges,
    and :math:`\boldsymbol{\psi}` must be defined on nodes. We obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u^T M_e \, j} = \mathbf{u^T M_{e\sigma} \, e} \\
        &\mathbf{u^T M_e \, e} = -\mathbf{u^T M_e G} \boldsymbol{\phi} \\
        &-\boldsymbol{\psi} \mathbf{(G^T - B) M_e \, j} = \boldsymbol{\psi} \mathbf{q}

    where

    * :math:`\mathbf{G}` is the nodal gradient operator
    * :math:`\mathbf{q}` is an integrated source term
    * :math:`\mathbf{B}` is a matrix that applies boundary conditions
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities
      projected to edges

    By combining the discrete equations to form a discrete solution in terms of the
    discrete electric potential on mesh nodes, we obtain:

    .. math::
        \big [ \mathbf{(G^T - B) M_{e\sigma} G} \big ] \boldsymbol{\phi} = \mathbf{q}

    Note that :math:`\mathbf{G^T - B}` effectively acts as a divergence operator in which
    boundary conditions have been applied.

    """

    _solutionType = "phiSolution"
    _formulation = "EB"  # N potentials means B is on faces
    fieldsPair = Fields3DNodal

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):  # noqa 107
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        # Not sure why I need to do this
        # To evaluate mesh.aveE2CC, this is required....
        if mesh._meshType == "TREE":
            mesh.nodal_gradient
        elif mesh._meshType == "CYL":
            bc_type = "Neumann"
        self.bc_type = bc_type
        self.setBC()

    @property
    def bc_type(self):
        """Get type of boundary condition to use for simulation.

        Returns
        -------
        {"Neumann", "Robin", "Mixed"}

        Notes
        -----
        Robin and Mixed are equivalent.
        """
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value):
        self._bc_type = validate_string(
            "bc_type", value, ["Neumann", ("Robin", "Mixed")]
        )

    def getA(self, resistivity=None):
        r"""Return the system matrix.

        This method generates and returns the system matrix for the nodal DC
        resistivity problem. The system matrix is given by:

        .. math::
            \mathbf{A} = \mathbf{(G^T - B) M_{e\sigma} G}

        where

        * :math:`\mathbf{G}` is the nodal gradient operator
        * :math:`\mathbf{B}` is a matrix that implements boundary conditions
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities
          projected to edges

        Parameters
        ----------
        resistivity : (n_cells,) numpy.ndarray
            Electrical resistivities defined at cell centers

        Returns
        -------
        (n_nodes, n_nodes) sp.sparse.csr_matrix
            The system matrix.
        """
        if resistivity is None:
            MeSigma = self.MeSigma
        else:
            MeSigma = self.mesh.get_edge_inner_product(1.0 / resistivity)
        Grad = self.mesh.nodal_gradient
        A = Grad.T.tocsr() @ MeSigma @ Grad

        if self.bc_type == "Neumann":
            # Handling Null space of A
            I, J, V = sp.find(A[0, :])
            for jj in J:
                A[0, jj] = 0.0
            A[0, 0] = 1.0
        else:
            # Dirichlet BC type should already have failed
            # Also, this will fail if sigma is anisotropic
            try:
                A = A + sp.diags(self._AvgBC @ self.sigma, format="csr")
            except ValueError as err:
                if len(self.sigma) != len(self.mesh):
                    raise NotImplementedError(
                        "Anisotropic conductivity is not supported for Robin boundary "
                        "conditions, please use 'Neumann'."
                    )
                else:
                    raise err

        return A

    def getADeriv(self, u, v, adjoint=False):
        r"""Get cnductivity derivative operation for the system matrix times a vector.

        The system matrix is given by:

        .. math::
            \mathbf{A} = \mathbf{(G^T - B) M_{e\sigma} G}

        where

        * :math:`\mathbf{G}` is the nodal gradient operator
        * :math:`\mathbf{B}` is a matrix that implements boundary conditions
        * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities
          projected to edges

        See the *Notes* section of the doc strings for :class:`Simulation3DNodal`
        for a full description of the formulation.

        Where :math:`\mathbf{m}` are the set of model parameters defining the
        electromagnetic properties :math:`\mathbf{v}` is a vector and
        :math:`\boldsymbol{\phi}` is the discrete electric potential solution, this
        method assumes the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\phi})}{\partial \mathbf{m}} \,
            \mathbf{v}

        Or the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\phi})}{\partial \mathbf{m}}^T
            \, \mathbf{v}

        Parameters
        ----------
        u : (n_nodes,) numpy.ndarray
            The solution for the electric potentials.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_nodes,) for the adjoint
            operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_nodes,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        Grad = self.mesh.nodal_gradient
        if not adjoint:
            out = Grad.T @ self.MeSigmaDeriv(Grad @ u, v, adjoint)
        else:
            out = self.MeSigmaDeriv(Grad @ u, Grad @ v, adjoint)
        if self.bc_type != "Neumann" and self.sigmaMap is not None:
            if getattr(self, "_MBC_sigma", None) is None:
                self._MBC_sigma = self._AvgBC @ self.sigmaDeriv
            out += _inner_mat_mul_op(self._MBC_sigma, u, v, adjoint)
        return out

    def setBC(self):
        """Set boundary conditions for the divergence operation."""
        if self.bc_type == "Dirichlet":
            # do nothing
            raise ValueError(
                "Dirichlet conditions are not supported in the Nodal formulation"
            )
        elif self.bc_type == "Neumann":
            if self.verbose:
                print(
                    "Homogeneous Neumann is the natural BC for this nodal discretization."
                )
            return
        else:
            mesh = self.mesh
            # calculate alpha, beta, gamma at the boundary faces
            boundary_faces = mesh.boundary_faces
            boundary_normals = mesh.boundary_face_outward_normals
            n_bf = len(boundary_faces)

            # Top gets 0 Nuemann
            alpha = np.zeros(n_bf)
            # beta = np.ones(n_bf) = 1.0

            # not top get Robin condition
            # assume a source point at the middle of the top of the mesh
            middle = np.median(mesh.nodes, axis=0)
            top_v = np.max(mesh.nodes[:, -1])
            source_point = np.r_[middle[:-1], top_v]

            # Others: Robin: alpha * phi + d phi dn = 0
            # where alpha = 1 / r  * r_hat_dot_n
            # TODO: Implement Zhang et al. (1995)

            r_vec = boundary_faces - source_point
            r = np.linalg.norm(r_vec, axis=-1)
            r_hat = r_vec / r[:, None]
            r_dot_n = np.einsum("ij,ij->i", r_hat, boundary_normals)

            # determine faces that are on the sides and bottom of the mesh...
            if self.surface_faces is None:
                if mesh._meshType.lower() == "tree":
                    not_top = boundary_faces[:, -1] != top_v
                elif mesh._meshType.lower() in ["tensor", "curv"]:
                    # mesh faces are ordered, faces_x, faces_y, faces_z so...
                    if mesh.dim == 2:
                        is_b = make_boundary_bool(mesh.shape_faces_y)
                        is_t = np.zeros(mesh.shape_faces_y, dtype=bool, order="F")
                        is_t[:, -1] = True
                    else:
                        is_b = make_boundary_bool(mesh.shape_faces_z)
                        is_t = np.zeros(mesh.shape_faces_z, dtype=bool, order="F")
                        is_t[:, :, -1] = True
                    is_t = is_t.reshape(-1, order="F")[is_b]
                    not_top = np.ones(boundary_faces.shape[0], dtype=bool)
                    not_top[-len(is_t) :] = ~is_t  # noqa E203
                else:
                    raise NotImplementedError(
                        f"Unable to infer surface boundaries for {type(mesh)}, please "
                        f"set the `surface_faces` property."
                    )
            else:
                not_top = ~self.surface_faces

            alpha[not_top] = (r_dot_n / r)[not_top]

            P_bf = self.mesh.project_face_to_boundary_face

            AvgN2Fb = P_bf @ self.mesh.average_node_to_face
            AvgCC2Fb = P_bf @ self.mesh.average_cell_to_face

            AvgCC2Fb = sp.diags(alpha * (P_bf @ self.mesh.face_areas)) @ AvgCC2Fb
            self._AvgBC = AvgN2Fb.T @ AvgCC2Fb

    def getRHS(self):
        """Compute and return right-hand sides for all sources.

        For This method computes and returns the right-hand sides used to solve the
        discrete finite volume solution for all sources.

        Returns
        -------
        (n_nodes, n_sources) numpy.ndarray
            The right-hand sides.
        """
        RHS = self.getSourceTerm()
        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        """Get derivative of the right-hand side with respect to the model.

        For ``Simulation3DNodal``, the derivative of the right-hand side with respect
        to the model is zero.

        Returns
        -------
        simpeg.utils.Zero
            The SimPEG zero operator.
        """
        return Zero()

    @property
    def _clear_on_sigma_update(self):
        """Add items to be cleared upon updating the `sigma` property.

        Returns
        -------
        list
            All of the items to be cleared up updating the `sigma` property.
        """
        return super()._clear_on_sigma_update + ["_MBC_sigma"]


class Simulation3DHierarchicalNodal(
    BaseHierarchicalElectricalSimulation, Simulation3DNodal
):
    r"""3D hierarchical nodal DC resistivity simulation class.

    Where :math:`\phi` is the electric potential, :math:`\sigma` is the electrical conductivity,
    and :math:`I \delta (r)` represents a grounded source which injects static current *I* at
    location *r*, the DC resistivity problem is given by:

    .. math::
        \nabla \cdot \sigma \nabla \phi = - I \delta (r)

    The ``Simulation3DHierarchicalNodal`` class uses the mimetic finite volume approach to
    solve for the discrete electric potentials at mesh nodes. This simulation adopts the
    hierarchical framework wherein:

    * Thick structures are parameterized as conductivities at cell centers. This property
      and the corresponding mapping are set with `sigma` and `sigmaMap`.
    * Sheet-like structures can be parameterized as conductances on mesh faces. This property
      and the corresponding mapping are set with `tau` and `tauMap`.
    * Wire-like structures can be parameterized as area-integrated conductivities on
      mesh edges. This property and the corresponding mapping as set with `kappa` and
      `kappaMap`.

    See the *Notes* section for a comprehensive description of the formulation.

    Notes
    -----
    To derive the discrete solution, we start by considering Ampere's law and Faraday's law.
    In the static regime, all time-derivative are zero. By also taking the divergence of
    Ampere's law, we obtain:

    .. math::
        &\nabla \times \vec{e} = 0\\
        &\nabla \cdot \vec{j} = - \nabla \cdot \vec{j}_s

    where :math:`\vec{e}` is the electric field, :math:`\vec{j}` is the current density outside
    the grounded source and and :math:`\vec{j}_s` is the source current density. The constitutive
    relation between the electric field and current density is given by Ohm's law:

    .. math::
        \vec{j} = \sigma \vec{e}

    where :math:`\sigma` is the electrical conductivity.

    Faraday's law implies the electric field can be defined as the gradient of a scalar
    potential :math:`\phi` as follows:

    .. math::
        \phi = -\nabla \phi

    For a source representing current *I* injected at point *r*, the static form of Ampere's
    law becomes:

    .. math::
        \nabla \cdot \vec{j} = - I \delta (r)

    For a vector test function :math:`\vec{u}` and a scalar test function :math:`\psi`, we
    take the inner products with the three previous equations. Through vector calculus
    identities and the divergence theorem, we obtain:

    .. math::
        & \int_\Omega \vec{u} \cdot \vec{j} \, dv =
        \int_\Omega \vec{u} \cdot \sigma \vec{e} \, dv \\
        & \int_\Omega \vec{u} \cdot \vec{e} \, dv =
        - \int_\Omega \vec{u} \cdot \nabla \phi \, dv \\
        & -\int_\Omega \nabla \psi \cdot \vec{j} \, dv
        +\oint_{\partial \Omega} \psi (\vec{j} \cdot \hat{n}) \, da =
        -I \int_\Omega \psi \, \delta (r) dv

    where the surface integral defines the boundary conditions.

    The hierarchical approach assumes the existence of infinitessimally thin plate-like
    regions between adjacent mesh faces and infinitessimally thin wire-like regions
    between adjacent mesh edges. We re-express the inner-product with Ohm's law as follows:

    .. math::
        \int_\Omega \vec{u} \cdot \vec{j} \, dv =&
        \sum_{n}^{nc} \int \vec{u} \cdot \sigma_n \vec{e} \, dv \\
        &+ \sum_{n}^{nf} \int \vec{u} \cdot \tau_n \vec{e} \, da \\
        &+ \sum_{n}^{ne} \int \vec{u} \cdot \kappa_n \vec{e} \, d\ell

    where :math:`\sigma_n` is the conductivity in cell *n*, :math:`\tau_n` is the face
    conductance on face *n*, and :math:`\kappa_n` is the area-integrated conductivity
    on edge *n*.

    The above expressions are discretized in space according to the finite volume method.
    The discrete electric fields :math:`\mathbf{e}` are defined on mesh edges,
    and the discrete electric potentials :math:`\boldsymbol{\phi}` are defined on nodes.
    This implies :math:`\mathbf{j}` and :math:`\mathbf{u}` must be defined on mesh edges,
    and :math:`\boldsymbol{\psi}` must be defined on nodes. We obtain the following
    set of discrete inner-products:

    .. math::
        &\mathbf{u^T M_e \, j} = \mathbf{u^T M_{e\Sigma} \, e} \\
        &\mathbf{u^T M_e \, e} = -\mathbf{u^T M_e G} \boldsymbol{\phi} \\
        &-\boldsymbol{\psi} \mathbf{(G^T - B) M_e \, j} = \boldsymbol{\psi} \mathbf{q}

    where:

    .. math::
        \mathbf{M_{e\Sigma}} = \mathbf{M_{e\sigma} + M_{e\tau} + M_{e\kappa}}

    and

    * :math:`\mathbf{G}` is the nodal gradient operator
    * :math:`\mathbf{q}` is an integrated source term
    * :math:`\mathbf{B}` is a matrix that applies boundary conditions
    * :math:`\mathbf{M_e}` is the edge inner-product matrix
    * :math:`\mathbf{M_{e\sigma}}` is the inner-product matrix for conductivities
      projected to edges
    * :math:`\mathbf{M_{e\tau}}` is the inner-product matrix for conductances
      projected to edges
    * :math:`\mathbf{M_{e\kappa}}` is the inner-product matrix for area integrated
      conductivities projected to edges

    By combining the discrete equations to form a discrete solution in terms of the
    discrete electric potential on mesh nodes, we obtain:

    .. math::
        \big [ \mathbf{(G^T - B) M_{e\Sigma} G} \big ] \boldsymbol{\phi} = \mathbf{q}

    """

    pass


Simulation3DCellCentred = Simulation3DCellCentered  # UK and US!
