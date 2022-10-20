import numpy as np
import scipy.sparse as sp
import properties

from ....utils import mkvc, Zero
from ....data import Data
from ....base import BaseElectricalPDESimulation
from .survey import Survey
from .fields import Fields3DCellCentered, Fields3DNodal
from .utils import _mini_pole_pole
from discretize.utils import make_boundary_bool


class BaseDCSimulation(BaseElectricalPDESimulation):
    """
    Base DC Problem
    """

    survey = properties.Instance("a DC survey object", Survey, required=True)

    storeJ = properties.Bool("store the sensitivity matrix?", default=False)

    _mini_survey = None

    Ainv = None
    _Jmatrix = None
    gtgdiag = None

    def __init__(self, *args, **kwargs):
        miniaturize = kwargs.pop("miniaturize", False)
        super().__init__(*args, **kwargs)
        # Do stuff to simplify the forward and JTvec operation if number of dipole
        # sources is greater than the number of unique pole sources
        if miniaturize:
            self._dipoles, self._invs, self._mini_survey = _mini_pole_pole(self.survey)

    def fields(self, m=None, calcJ=True):
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
        if self._Jmatrix is None:
            if f is None:
                f = self.fields(m)
            self._Jmatrix = self._Jtvec(m, v=None, f=f).T
        return self._Jmatrix

    def dpred(self, m=None, f=None):
        if self._mini_survey is not None:
            # Temporarily set self.survey to self._mini_survey
            survey = self.survey
            self.survey = self._mini_survey

        data = super().dpred(m=m, f=f)

        if self._mini_survey is not None:
            # reset survey
            self.survey = survey

        return self._mini_survey_data(data)

    def getJtJdiag(self, m, W=None):
        """
        Return the diagonal of JtJ
        """
        if self.gtgdiag is None:
            J = self.getJ(m)

            if W is None:
                W = np.ones(J.shape[0])
            else:
                W = W.diagonal() ** 2

            diag = np.zeros(J.shape[1])
            for i in range(J.shape[0]):
                diag += (W[i]) * (J[i] * J[i])

            self.gtgdiag = diag
        return self.gtgdiag

    def Jvec(self, m, v, f=None):
        """
        Compute sensitivity matrix (J) and vector (v) product.
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
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
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
        """
        Evaluates the sources, and puts them in matrix form
        :rtype: tuple
        :return: q (nC or nN, nSrc)
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
        toDelete = super().deleteTheseOnModelUpdate
        if self._Jmatrix is not None:
            toDelete = toDelete + ["_Jmatrix"]
        if self.gtgdiag is not None:
            toDelete = toDelete + ["gtgdiag"]
        return toDelete

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
    """
    3D cell centered DC problem
    """

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered

    bc_type = properties.StringChoice(
        "Type of boundary condition to use for simulation. Note that Robin and Mixed "
        "are equivalent.",
        choices=["Dirichlet", "Neumann", "Robin", "Mixed"],
        default="Robin",
    )

    def __init__(self, mesh, **kwargs):

        BaseDCSimulation.__init__(self, mesh, **kwargs)
        self.setBC()

    def getA(self, resistivity=None):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = D MfRhoI G
        """

        D = self.Div
        G = self.Grad
        if resistivity is None:
            MfRhoI = self.MfRhoI
        else:
            MfRhoI = self.mesh.getFaceInnerProduct(resistivity, invMat=True)
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
        D = self.Div
        G = self.Grad
        MfRhoIDeriv = self.MfRhoIDeriv

        if adjoint:
            return MfRhoIDeriv(G @ u, D.T @ v, adjoint)

        return D * (MfRhoIDeriv(G @ u, v, adjoint))

    def getRHS(self):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm()

        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
        """
        # TODO: add qDeriv for RHS depending on m
        # qDeriv = source.evalDeriv(self, adjoint=adjoint)
        # return qDeriv
        return Zero()

    def setBC(self):
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
        else:  # self.bc_type == "Mixed":
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

            # determine faces that are on the sides and bottom of the mesh...
            if mesh._meshType.lower() == "tree":
                not_top = boundary_faces[:, -1] != top_v
            else:
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
                not_top = np.zeros(boundary_faces.shape[0], dtype=bool)
                not_top[-len(is_t) :] = ~is_t
            alpha[not_top] = (r_dot_n / r)[not_top]

        B, bc = mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
        # bc should always be 0 because gamma was always 0 above
        self.Grad = self.Grad - B


class Simulation3DNodal(BaseDCSimulation):
    """
    3D nodal DC problem
    """

    _solutionType = "phiSolution"
    _formulation = "EB"  # N potentials means B is on faces
    fieldsPair = Fields3DNodal

    bc_type = properties.StringChoice(
        "Type of boundary condition to use for simulation. Note that Robin and Mixed "
        "are equivalent.",
        choices=["Neumann", "Robin", "Mixed"],
        default="Robin",
    )

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        # Not sure why I need to do this
        # To evaluate mesh.aveE2CC, this is required....
        if mesh._meshType == "TREE":
            mesh.nodalGrad
        elif mesh._meshType == "CYL":
            self.bc_type == "Neumann"
        self.setBC()

    def getA(self, resistivity=None):
        """
        Make the A matrix for the cell centered DC resistivity problem
        A = G.T MeSigma G
        """
        if resistivity is None:
            MeSigma = self.MeSigma
        else:
            MeSigma = self.mesh.getEdgeInnerProduct(1.0 / resistivity)
        Grad = self.mesh.nodalGrad
        A = Grad.T @ MeSigma @ Grad

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
                A = A + sp.diags(self._AvgBC @ self.sigma)
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
        """
        Product of the derivative of our system matrix with respect to the
        model and a vector
        """
        Grad = self.mesh.nodalGrad
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
            if mesh._meshType.lower() == "tree":
                not_top = boundary_faces[:, -1] != top_v
            else:
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
                not_top = np.zeros(boundary_faces.shape[0], dtype=bool)
                not_top[-len(is_t) :] = ~is_t
            alpha[not_top] = (r_dot_n / r)[not_top]

            P_bf = self.mesh.project_face_to_boundary_face

            AvgN2Fb = P_bf @ self.mesh.average_node_to_face
            AvgCC2Fb = P_bf @ self.mesh.average_cell_to_face

            AvgCC2Fb = sp.diags(alpha * (P_bf @ self.mesh.face_areas)) @ AvgCC2Fb
            self._AvgBC = AvgN2Fb.T @ AvgCC2Fb

    def getRHS(self):
        """
        RHS for the DC problem
        q
        """

        RHS = self.getSourceTerm()
        return RHS

    def getRHSDeriv(self, source, v, adjoint=False):
        """
        Derivative of the right hand side with respect to the model
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
