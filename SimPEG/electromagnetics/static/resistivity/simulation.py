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
from ....base import BaseElectricalPDESimulation
from .survey import Survey
from .fields import Fields3DCellCentered, Fields3DNodal
from .utils import _mini_pole_pole
from discretize.utils import make_boundary_bool


class BaseDCSimulation(BaseElectricalPDESimulation):
    r"""Base simulation class for the 3D DC resistivity problem.

    This class is used to define properties and methods necessary for solving the
    3D DC resistivity problem. For current :math:`I` injected at point :math:`r_s`,
    the 3D DC resistivity problem is expressed as:

    .. math::
        \nabla \cdot \sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where :math:`\nabla` is the gradient operator, :math:`\sigma` is the electrical
    conductivity, and we wish to solve for the electric potential :math:`\phi`.
    Child classes of ``BaseDCSimulation`` solve the above expression numerically
    using mimetic finite volume.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh.
    survey : None, SimPEG.electromagnetics.static.resistivity.survey.Survey
        The DC resisitivity survey.
    storeJ : bool
        Whether to construct and store the sensitivity matrix.
    miniaturize : bool
        If ``True``, we compute the fields for each unique source electrode location.
        We avoid computing the fields for repeated electrode locations and the
        fields for dipole sources can be constructed using superposition.
    fix_Jmatrix : bool
        Whether to fix the sensitivity matrix during Newton iterations.
    surface_faces : None, numpy.ndarray of bool
        Array defining which faces to interpret as surfaces of the Neumann boundary.
    """

    _mini_survey = None

    Ainv = None

    def __init__(
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
        """Array defining which boundary faces to interpret as surfaces of Neumann boundary

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

    def fields(self, m=None, calcJ=True):
        """Solve for the fields for the model provided.

        Parameters
        ----------
        m : None, (nP,) numpy.ndarray
            The model.
        calcJ : bool
            Whether to calculate the sensitivities.

        Returns
        -------
        SimPEG.electromagnetics.static.resistivity.fields.FieldsDC
            The fields solved for all sources.
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
        r"""Generate the full sensitivity matrix.

        This method generates and stores the full sensitivity matrix for the
        model provided. I.e.:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        where :math:`\mathbf{d}` are the data and :math:`\mathbf{m}` are the model parameters.

        Parameters
        ----------
        m : (nP,) numpy.ndarray
            The model parameters.
        f : None, SimPEG.electromagnetics.static.resistivity.fields.FieldsDC
            Fields solved for all sources.

        Returns
        -------
        (nD, nP) numpy.ndarray
            The full sensitivity matrix.
        """
        if getattr(self, "_Jmatrix", None) is None:
            if f is None:
                f = self.fields(m)
            self._Jmatrix = self._Jtvec(m, v=None, f=f).T
        return self._Jmatrix

    def dpred(self, m=None, f=None):
        """Predict the data for a given model.

        Parameters
        ----------
        m : None, (nP,) numpy.ndarray
            The model parameters.
        f : None, SimPEG.electromagnetics.static.resistivity.fields.FieldsDC
            The fields object containing solution to the DC resistivity problem
            for all sources.

        Returns
        -------
        (nD,) numpy.ndarray
            The predicted data array.
        """
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
        r"""Return the diagonal of :math:`\mathbf{J^T J}`.

        Where :math:`\mathbf{d}` are the data and :math:`\mathbf{m}` are the model parameters,
        the sensitivity matrix :math:`\mathbf{J}` is defined as:

        .. math::
            \mathbf{J} = \dfrac{\partial \mathbf{d}}{\partial \mathbf{m}}

        This method returns the diagonals of :math:`\mathbf{J^T J}`. When the
        *W* input argument is used to include a diagonal weighting matrix
        :math:`\mathbf{W}`, this method returns the diagonal of
        :math:`\mathbf{W^T J^T J W}`.

        Parameters
        ----------
        m : (nP,) numpy.ndarray
            The model parameters.
        W : (nP, nP) scipy.sparse.csr_matrix
            A diagonal weighting matrix.
        f : None, SimPEG.electromagnetics.static.resistivity.fields.FieldsDC
            Fields solved for all sources.

        Returns
        -------
        (nP,) numpy.ndarray
            The diagonals.
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
        m : (nP,) numpy.ndarray
            The model parameters.
        v : (nP,) numpy.ndarray
            The vector.
        f : None, SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D
            2D fields solved in the wave domain.

        Returns
        -------
        (nD,) numpy.ndarray
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
        m : (nP,) numpy.ndarray
            The model parameters.
        v : (nD,) numpy.ndarray
            The vector.
        f : None, SimPEG.electromagnetics.static.resistivity.fields_2d.Fields2D
            2D fields solved in the wave domain

        Returns
        -------
        (nP,) numpy.ndarray
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
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        Full J matrix can be computed by inputing v=None
        """

        if self._mini_survey is not None:
            survey = self._mini_survey
        else:
            survey = self.survey

        if v is not None:
            if isinstance(v, Data):
                v = v.dobs
            v = self._mini_survey_dataT(v)
            v = Data(survey, v)
            Jtv = np.zeros(m.size)
        else:
            # This is for forming full sensitivity matrix
            Jtv = np.zeros((self.model.size, survey.nD), order="F")
            istrt = int(0)
            iend = int(0)

        for source in survey.source_list:
            u_source = f[source, self._solutionType].copy()
            for rx in source.receiver_list:
                # wrt f, need possibility wrt m
                if v is not None:
                    PTv = rx.evalDeriv(
                        source, self.mesh, f, v[source, rx], adjoint=True
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
        """Compute the discrete source terms (right-hand sides).

        Returns
        -------
        (nC or nN, nSrc) numpy.ndarray
            The array containing the right-hand sides for all sources.
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
    def deleteTheseOnModelUpdate(self):
        """Returns the properties to delete when model is updated.

        Returns
        -------
        list of str
            The properties to delete when model the is updated.
        """
        toDelete = super().deleteTheseOnModelUpdate
        return toDelete + ["_Jmatrix", "_gtgdiag"]

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


class Simulation3DCellCentered(BaseDCSimulation):
    r"""Cell centered 3D DC resistivity simulation.

    Simulation class which solves the 3D DC resistivity problem for electric
    potentials at cell centers. For a full description of the numerical approach,
    see the *Notes* section below.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The mesh.
    survey : None, Survey
        The DC resisitivity survey.
    storeJ : bool
        Whether to construct and store the sensitivity matrix.
    miniaturize : bool
        If ``True``, we compute the fields for each unique source electrode location.
        We avoid computing the fields for repeated electrode locations and the
        fields for dipole sources can be constructed using superposition.
    fix_Jmatrix : bool
        Whether to fix the sensitivity matrix during Newton iterations.
    surface_faces : None, numpy.ndarray of bool
        Array defining which faces to interpret as surfaces of the Neumann boundary.

    Notes
    -----
    For current :math:`I` injected at point :math:`r_s`, the 3D DC resistivity
    problem is expressed as:

    .. math::
        \nabla \cdot \sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where :math:`\sigma` is the electrical conductivity, and we wish
    to solve for the electric potential :math:`\phi`.

    Using mimetic finite volume, the 3D problem can be solved numerically by solving
    the following linear system:

    .. math::
        \mathbf{D \, M_{f\rho}^{-1} \, G} \, \boldsymbol{\phi} = \mathbf{q}

    where

    * :math:`\boldsymbol{\phi}` are the discrete electric potentials defined at cell centers
    * :math:`\mathbf{D}` is the face-divergence operator with imposed boundary conditions
    * :math:`G` is the cell-gradient operator
    * :math:`M_{f\rho}` is the resistivity inner-product matrix on cell faces; note :math:`\rho = 1/\sigma`

    **Axial Anisotropy:**

    In this case, the DC resistivity problem is defined according to:

    .. math::
        \nabla \cdot \Sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where

    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_x & 0 & 0 \\ 0 & \sigma_y & 0 \\ 0 & 0 & \sigma_z
        \end{bmatrix}

    The discrete 3D problem still takes the form:

    .. math::
        \mathbf{D \, M_{f\Rho}^{-1} \, G} \, \boldsymbol{\phi} = \mathbf{q}

    However :math:`M_{f\rho}` is a resistivity inner-product matrix on cell faces constructed using
    axial resistivities :math:`\rho_x = 1/\sigma_x`, :math:`\rho_y = 1/\sigma_y` and :math:`\rho_z = 1/\sigma_z`
    """

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.bc_type = bc_type
        self.setBC()

    @property
    def bc_type(self):
        """Type of boundary condition to use for simulation.

        The boundary conditions supported by the :class:`Simulation2DCellCentered` are:

        * "Dirichlet": Zero Dirichlet on the boundary (natural boundary conditions)
        * "Neumann": Zero Neumann on the boundary
        * "Robin" or "Mixed": Mix of zero Dirichlet and zero Neumann.

        Returns
        -------
        {"Dirichlet", "Neumann", "Robin", "Mixed"}
        """
        return self._bc_type

    @bc_type.setter
    def bc_type(self, value):
        self._bc_type = validate_string(
            "bc_type", value, ["Dirichlet", "Neumann", ("Robin", "Mixed")]
        )

    def getA(self, resistivity=None):
        r"""Compute the system matrix.

        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\phi` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. This method returns the system matrix
        for the cell-centered formulation, i.e.:

        .. math::
            \mathbf{A} = \mathbf{D \, M_{f\rho}^{-1} \, G}

        where :math:`\mathbf{D}` is the face divergence operator, :math:`\mathbf{G}` is the
        cell gradient operator with imposed boundary conditions, and :math:`\mathbf{M_{f\rho}}`
        is the inner product matrix for resistivities projected to faces.

        Parameters
        ----------
        resistivity : None, (n_cells,) numpy.ndarray
            The resistivities for all mesh cells (:math:`\Omega m`).

        Returns
        -------
        (n_cells, n_cells) scipy.sparse.csr_matrix
            The sparse system matrix.
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
        r"""Derivative of system matrix times a vector.

        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\phi` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. For a vector :math:`v`, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\phi})}{\partial \mathbf{m}} \, \mathbf{v}

        Or when set to do so, the method returns the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\Phi})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        u : (n_cells,) numpy.ndarray
            The solution for the fields for the current model; i.e. electric potentials at cell centers.
        v : numpy.ndarray
            The vector. (nP,) for the standard operation. (n_cells,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_cells,) for the standard operation.
            (nP,) for the adjoint operation.
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
        r"""Compute the source terms (right-hand sides).

        For a single source, the discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\phi` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side corresponding to the source term.
        This method computes and returns an array :math:`\mathbf{Q}`, whose columns are
        the right-hand sides for all sources.

        Returns
        -------
        (n_cells, nSrc) numpy.ndarray
            The array containing the right-hand sides for all sources.
        """

        RHS = self.getSourceTerm()

        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        r"""Derivative of the source term with respect to the model times a vector.

        The discrete solution to the 3D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\boldsymbol{\phi}` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side. This method returns the derivative
        of the right-hand side with respect to the model times a vector, i.e.:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}} \mathbf{v}

        or the adjoint operation:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}}^T \mathbf{v}

        Parameters
        ----------
        source : SimPEG.electromagnetic.static.resistivity.sources.BaseSrc
            The source object.
        v : numpy.ndarray
            The vector. Has shape (nP,) when performing the standard derivative operation.
            Has shape (n_cells,) when performing the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        Zero or numpy.ndarray
            Returns :py:class:`Zero` if the derivative with respect to the model is zero.
            Returns (n_cells,) :class:`numpy.ndarray` when computing the standard
            derivative operation. Returns (nP,) :class:`numpy.ndarray` when performing
            the adjoint.
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = source.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def setBC(self):
        """Sets the boundary conditions on the cell gradient operator.

        This method will set the boundary conditions on the cell gradient
        operator based on the value of the :py:attr:`bc_type` property.
        The options are:

        * "Dirichlet": Zero Dirichlet on the boundary (natural boundary conditions)
        * "Neumann": Zero Neumann on the boundary
        * "Robin" or "Mixed": Mix of zero Dirichlet and zero Neumann. Faces that use the Neumann boundary conditions are set using the :py:attr:`surface_faces` property.
        """
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
                    not_top[-len(is_t) :] = ~is_t
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
    r"""Nodal 3D DC resistivity simulation.

    Simulation class which solves the 3D DC resistivity problem for electric
    potentials on mesh nodes. For a full description of the numerical approach,
    see the *Notes* section below.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The mesh.
    survey : None, Survey
        The DC resisitivity survey.
    storeJ : bool
        Whether to construct and store the sensitivity matrix.
    miniaturize : bool
        If ``True``, we compute the fields for each unique source electrode location.
        We avoid computing the fields for repeated electrode locations and the
        fields for dipole sources can be constructed using superposition.
    fix_Jmatrix : bool
        Whether to fix the sensitivity matrix during Newton iterations.
    surface_faces : None, numpy.ndarray of bool
        Array defining which faces to interpret as surfaces of the Neumann boundary.

    Notes
    -----
    For current :math:`I` injected at point :math:`r_s`, the 3D DC resistivity
    problem is expressed as:

    .. math::
        \nabla \cdot \sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where :math:`\sigma` is the electrical conductivity, and we wish
    to solve for the electric potential :math:`\phi`.

    Using mimetic finite volume, the 3D problem can be solved numerically by solving
    the following linear system:

    .. math::
        \mathbf{G \, M_{e\sigma} \, G} \, \boldsymbol{\phi} = \mathbf{q}

    where

    * :math:`\boldsymbol{\phi}` are the discrete electric potentials defined on mesh nodes
    * :math:`G` is the nodal gradient operator
    * :math:`M_{e\sigma}` is the conductivity inner-product matrix on mesh edges

    **Axial Anisotropy:**

    In this case, the DC resistivity problem is defined according to:

    .. math::
        \nabla \cdot \Sigma \, \nabla \phi = - I \, \delta (r-r_s)

    where

    .. math::
        \Sigma = \begin{bmatrix}
        \sigma_x & 0 & 0 \\ 0 & \sigma_y & 0 \\ 0 & 0 & \sigma_z
        \end{bmatrix}

    The discrete 3D problem still takes the form:

    .. math::
        \mathbf{G^T \, M_{e\Sigma} \, G} \, \boldsymbol{\phi} = \mathbf{q}

    However :math:`M_{e\Sigma}` is a conductivity inner-product matrix on edges constructed using
    axial conductivities :math:`\sigma_x`, :math:`\sigma_y` and :math:`\sigma_z`.
    """

    _solutionType = "phiSolution"
    _formulation = "EB"  # N potentials means B is on faces
    fieldsPair = Fields3DNodal

    def __init__(self, mesh, survey=None, bc_type="Robin", **kwargs):
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

    def getA(self, resistivity=None):
        r"""Compute the system matrix.

        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\boldsymbol{\phi}` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. This method returns the system matrix
        for the nodal formulation, i.e.:

        .. math::
            \mathbf{A} = \mathbf{G^T \, M_{e\sigma} \, G}

        where :math:`\mathbf{G}` is the nodal gradient operator with imposed boundary conditions,
        and :math:`\mathbf{M_{e\sigma}}` is the inner product matrix for conductivities projected to edges.

        Parameters
        ----------
        resistivity : None, (n_cells,) numpy.ndarray
            The electrical resistivities defined at cell centers (:math:`\Omega m`).

        Returns
        -------
        (n_nodes, n_nodes) scipy.sparse.csr_matrix
            The sparse system matrix.
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
        r"""Derivative of system matrix times a vector.

        The discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\Phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\boldsymbol{\phi}` is the discrete solution,
        and :math:`\mathbf{q}` is the source term. For a vector :math:`v`, this method assumes
        the discrete solution is fixed and returns

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\phi})}{\partial \mathbf{m}} \, \mathbf{v}

        Or when set to do so, the method returns the adjoint operation

        .. math::
            \frac{\partial (\mathbf{A} \, \boldsymbol{\phi})}{\partial \mathbf{m}}^T \, \mathbf{v}

        Parameters
        ----------
        u : (n_nodes,) numpy.ndarray
            The solution for the fields for the current model; i.e. electric potentials at cell nodes.
        v : numpy.ndarray
            The vector. (nP,) for the standard operation. (n_nodes,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_nodes,) for the standard operation.
            (nP,) for the adjoint operation.
        """
        Grad = self.mesh.nodal_gradient
        if not adjoint:
            out = Grad.T @ self.MeSigmaDeriv(Grad @ u, v, adjoint)
        else:
            out = self.MeSigmaDeriv(Grad @ u, Grad @ v, adjoint)
        if self.bc_type != "Neumann" and self.sigmaMap is not None:
            if getattr(self, "_MBC_sigma", None) is None:
                self._MBC_sigma = self._AvgBC @ self.sigmaDeriv
            if not isinstance(u, Zero):
                u = u.flatten()
                if v.ndim > 1:
                    u = u[:, None]
                if not adjoint:
                    out += u * (self._MBC_sigma @ v)
                else:
                    out += self._MBC_sigma.T @ (u * v)
        return out

    def setBC(self):
        """Sets the boundary conditions on the nodal gradient operator.

        This method will set the boundary conditions on the nodal gradient
        operator based on the value of the :py:attr:`bc_type` property.
        The options are:

        * "Neumann": Zero Neumann on the boundary (natural boundary conditions).
        * "Robin" or "Mixed": Mix of Robin and zero Neumann boundary conditions.
        """
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

            # Top gets 0 Neumann
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
                    not_top[-len(is_t) :] = ~is_t
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
        r"""Compute the source terms (right-hand sides).

        For a single source, the discrete solution to the 3D DC resistivity problem is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\phi` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side corresponding to the source term.
        This method computes and returns an array :math:`\mathbf{Q}`, whose columns are
        the right-hand sides for all sources.

        Returns
        -------
        (n_nodes, nSrc) numpy.ndarray
            The array containing the right-hand sides for all sources.
        """

        RHS = self.getSourceTerm()
        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        r"""Derivative of the source term with respect to the model times a vector.

        The discrete solution to the 3D DC resistivity problem in the wave domain
        is expressed as:

        .. math::
            \mathbf{A}\,\boldsymbol{\phi} = \mathbf{q}

        where :math:`\mathbf{A}` is the system matrix, :math:`\boldsymbol{\phi}` is the discrete solution,
        and :math:`\mathbf{q}` is the right-hand side. This method returns the derivative
        of the right-hand side with respect to the model times a vector, i.e.:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}} \mathbf{v}

        or the adjoint operation:

        .. math::
            \frac{\partial \mathbf{q}}{\partial \mathbf{m}}^T \mathbf{v}

        Parameters
        ----------
        source : SimPEG.electromagnetic.static.resistivity.sources.BaseSrc
            The source object.
        v : numpy.ndarray
            The vector. Has shape (nP,) when performing the standard derivative operation.
            Has shape (n_nodes,) when performing the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        Zero or numpy.ndarray
            Returns :py:class:`Zero` if the derivative with respect to the model is zero.
            Returns (n_nodes,) :class:`numpy.ndarray` when computing the standard
            derivative operation. Returns (nP,) :class:`numpy.ndarray` when performing
            the adjoint.
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = source.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()

    @property
    def _clear_on_sigma_update(self):
        """
        These matrices are deleted if there is an update to the conductivity
        model
        """
        return super()._clear_on_sigma_update + ["_MBC_sigma"]


Simulation3DCellCentred = Simulation3DCellCentered  # UK and US!
