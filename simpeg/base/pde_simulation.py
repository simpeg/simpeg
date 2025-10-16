import inspect
import warnings
import numpy as np
import pymatsolver
import scipy.sparse as sp
from discretize.utils import Zero, TensorType
import discretize.base
from ..simulation import BaseSimulation
from .. import props
from scipy.constants import mu_0

from ..utils import validate_type, get_default_solver, get_logger, PerformanceWarning


def _inner_mat_mul_op(M, u, v=None, adjoint=False):
    u = np.squeeze(u)
    if sp.issparse(M):
        if v is not None:
            if v.ndim > 1:
                v = np.squeeze(v)
            if u.ndim > 1:
                # u has multiple fields
                if v.ndim == 1:
                    v = v[:, None]
                if adjoint and v.shape[1] != u.shape[1] and v.shape[1] > 1:
                    # make sure v is a good shape
                    v = v.reshape(u.shape[0], -1, u.shape[1])
            else:
                if v.ndim > 1:
                    u = u[:, None]
            if v.ndim > 2:
                u = u[:, None, :]
            if adjoint:
                if u.ndim > 1 and u.shape[-1] > 1:
                    return M.T * (u * v).sum(axis=-1)
                return M.T * (u * v)
            if u.ndim > 1 and u.shape[1] > 1:
                return np.squeeze(u[:, None, :] * (M * v)[:, :, None])
            return u * (M * v)
        else:
            if u.ndim > 1:
                UM = sp.vstack([sp.diags(u[:, i]) @ M for i in range(u.shape[1])])
            else:
                U = sp.diags(u, format="csr")
                UM = U @ M
            if adjoint:
                return UM.T
            return UM
    elif isinstance(M, tuple):
        # assume it was a tuple of M_func, prop_deriv
        M_deriv_func, prop_deriv = M
        if u.ndim > 1:
            Mu = [M_deriv_func(u[:, i]) for i in range(u.shape[1])]
            if v is None:
                Mu = sp.vstack([M @ prop_deriv for M in Mu])
                if adjoint:
                    return Mu.T
                return Mu
            elif v.ndim > 1:
                v = np.squeeze(v)
            if adjoint:
                return sum(
                    [prop_deriv.T @ (Mu[i].T @ v[..., i]) for i in range(u.shape[1])]
                )
            pv = prop_deriv @ v
            return np.stack([M @ pv for M in Mu], axis=-1)
        else:
            Mu = M_deriv_func(u)
            if v is None:
                Mu = Mu @ prop_deriv
                if adjoint:
                    return Mu.T
                return Mu
            elif v.ndim > 1:
                v = np.squeeze(v)
            if adjoint:
                return prop_deriv.T @ (Mu.T @ v)
            return Mu @ (prop_deriv @ v)
    else:
        raise TypeError(
            "The stashed property derivative is an unexpected type. Expected either a `tuple` or a "
            f"sparse matrix. Received a {type(M)}."
        )


def with_property_mass_matrices(property_name):
    """
    This decorator will automatically populate all of the property mass matrices.

    Given the property "prop", this will add properties and functions to the class
    representing all of the possible mass matrix operations on the mesh.

    For a given property, "prop", they will be named:

    * MccProp
    * MccPropDeriv
    * MccPropI
    * MccPropIDeriv

    and so on for each "Mcc", "Mn", "Mf", and "Me".
    """

    def decorator(cls):
        arg = property_name.lower()
        arg = arg[0].upper() + arg[1:]

        @property
        def Mcc_prop(self):
            """
            Cell center property inner product matrix.
            """
            stash_name = f"_Mcc_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = sp.diags(self.mesh.cell_volumes * prop, format="csr")
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mcc{arg}", Mcc_prop)

        @property
        def Mn_prop(self):
            """
            Node property inner product matrix.
            """
            stash_name = f"_Mn_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                vol = self.mesh.cell_volumes
                M_prop = sp.diags(self.mesh.aveN2CC.T * (vol * prop), format="csr")
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mn{arg}", Mn_prop)

        @property
        def Mf_prop(self):
            """
            Face property inner product matrix.
            """
            stash_name = f"_Mf_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = self.mesh.get_face_inner_product(model=prop)
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mf{arg}", Mf_prop)

        @property
        def Me_prop(self):
            """
            Edge property inner product matrix.
            """
            stash_name = f"_Me_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = self.mesh.get_edge_inner_product(model=prop)
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Me{arg}", Me_prop)

        @property
        def MccI_prop(self):
            """
            Cell center property inner product inverse matrix.
            """
            stash_name = f"_MccI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = sp.diags(1.0 / (self.mesh.cell_volumes * prop), format="csr")
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mcc{arg}I", MccI_prop)

        @property
        def MnI_prop(self):
            """
            Node property inner product inverse matrix.
            """
            stash_name = f"_MnI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                vol = self.mesh.cell_volumes
                M_prop = sp.diags(
                    1.0 / (self.mesh.aveN2CC.T * (vol * prop)), format="csr"
                )
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mn{arg}I", MnI_prop)

        @property
        def MfI_prop(self):
            """
            Face property inner product inverse matrix.
            """
            stash_name = f"_MfI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = self.mesh.get_face_inner_product(
                    model=prop, invert_matrix=True
                )
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Mf{arg}I", MfI_prop)

        @property
        def MeI_prop(self):
            """
            Edge property inner product inverse matrix.
            """
            stash_name = f"_MeI_{arg}"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                M_prop = self.mesh.get_edge_inner_product(
                    model=prop, invert_matrix=True
                )
                setattr(self, stash_name, M_prop)
            return getattr(self, stash_name)

        setattr(cls, f"Me{arg}I", MeI_prop)

        def MccDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MccProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mcc_{arg}_deriv"

            if getattr(self, stash_name, None) is None:
                M_prop_deriv = sp.diags(self.mesh.cell_volumes) * getattr(
                    self, f"{arg.lower()}Deriv"
                )
                setattr(self, stash_name, M_prop_deriv)
            return _inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

        setattr(cls, f"Mcc{arg}Deriv", MccDeriv_prop)

        def MnDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MnProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mn_{arg}_deriv"
            if getattr(self, stash_name, None) is None:
                M_prop_deriv = (
                    self.mesh.aveN2CC.T
                    * sp.diags(self.mesh.cell_volumes)
                    * getattr(self, f"{arg.lower()}Deriv")
                )
                setattr(self, stash_name, M_prop_deriv)
            return _inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

        setattr(cls, f"Mn{arg}Deriv", MnDeriv_prop)

        def MfDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MfProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mf_{arg}_deriv"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                t_type = TensorType(self.mesh, prop)

                M_deriv_func = self.mesh.get_face_inner_product_deriv(model=prop)
                prop_deriv = getattr(self, f"{arg.lower()}Deriv")
                # t_type == 3 for full tensor model, t_type < 3 for scalar, isotropic, or axis-aligned anisotropy.
                if t_type < 3 and self.mesh._meshType.lower() in (
                    "cyl",
                    "tensor",
                    "tree",
                ):
                    M_prop_deriv = M_deriv_func(np.ones(self.mesh.n_faces)) @ prop_deriv
                    setattr(self, stash_name, M_prop_deriv)
                else:
                    setattr(self, stash_name, (M_deriv_func, prop_deriv))

            return _inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

        setattr(cls, f"Mf{arg}Deriv", MfDeriv_prop)

        def MeDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MeProperty` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Me_{arg}_deriv"
            if getattr(self, stash_name, None) is None:
                prop = getattr(self, arg.lower())
                t_type = TensorType(self.mesh, prop)

                M_deriv_func = self.mesh.get_edge_inner_product_deriv(model=prop)
                prop_deriv = getattr(self, f"{arg.lower()}Deriv")
                # t_type == 3 for full tensor model, t_type < 3 for scalar, isotropic, or axis-aligned anisotropy.
                if t_type < 3 and self.mesh._meshType.lower() in (
                    "cyl",
                    "tensor",
                    "tree",
                ):
                    M_prop_deriv = M_deriv_func(np.ones(self.mesh.n_edges)) @ prop_deriv
                    setattr(self, stash_name, M_prop_deriv)
                else:
                    setattr(self, stash_name, (M_deriv_func, prop_deriv))
            return _inner_mat_mul_op(getattr(self, stash_name), u, v=v, adjoint=adjoint)

        setattr(cls, f"Me{arg}Deriv", MeDeriv_prop)

        def MccIDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MccPropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Mcc{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Mcc{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Mcc{arg}IDeriv", MccIDeriv_prop)

        def MnIDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MnPropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Mn{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Mn{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Mn{arg}IDeriv", MnIDeriv_prop)

        def MfIDeriv_prop(self, u, v=None, adjoint=False):
            """I
            Derivative of `MfPropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Mf{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Mf{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Mf{arg}IDeriv", MfIDeriv_prop)

        def MeIDeriv_prop(self, u, v=None, adjoint=False):
            """
            Derivative of `MePropertyI` with respect to the model.
            """
            if getattr(self, f"{arg.lower()}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"Me{arg}I")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"Me{arg}Deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        setattr(cls, f"Me{arg}IDeriv", MeIDeriv_prop)

        @property
        def _clear_on_prop_update(self):
            items = [
                f"_Mcc_{arg}",
                f"_Mn_{arg}",
                f"_Mf_{arg}",
                f"_Me_{arg}",
                f"_MccI_{arg}",
                f"_MnI_{arg}",
                f"_MfI_{arg}",
                f"_MeI_{arg}",
                f"_Mcc_{arg}_deriv",
                f"_Mn_{arg}_deriv",
                f"_Mf_{arg}_deriv",
                f"_Me_{arg}_deriv",
            ]
            return items

        setattr(cls, f"_clear_on_{arg.lower()}_update", _clear_on_prop_update)
        return cls

    return decorator


class BasePDESimulation(BaseSimulation):
    """Base simulation for PDE solutions.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh on which the forward problem is discretized.
    solver : type[pymatsolver.base.Base], optional
        Numerical solver used to solve the forward problem. If ``None``,
        an appropriate solver specific to the simulation class is set by default.
    solver_opts : dict, optional
        Solver-specific parameters. If ``None``, default parameters are used for
        the solver set by ``solver``. Otherwise, the ``dict`` must contain appropriate
        pairs of keyword arguments and parameter values for the solver. Please visit
        `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`__ to learn more
        about solvers and their parameters.
    """

    def __init__(self, mesh, solver=None, solver_opts=None, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)
        if solver is None:
            solver = get_default_solver()
            get_logger().info(
                f"Setting the default solver '{solver.__name__}' for the "
                f"'{type(self).__name__}'.\n"
                "To avoid receiving this message, pass a solver to the simulation. "
                "For example:"
                "\n\n"
                "  from simpeg.utils import get_default_solver\n"
                "\n"
                "  solver = get_default_solver()\n"
                f"  simulation = {type(self).__name__}(solver=solver, ...)"
            )
        self.solver = solver
        if solver_opts is None:
            solver_opts = {}
        self.solver_opts = solver_opts

    @property
    def mesh(self):
        """Mesh for the simulation.

        For more on meshes, visit :py:class:`discretize.base.BaseMesh`.

        Returns
        -------
        discretize.base.BaseMesh
            Mesh on which the forward problem is discretized.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = validate_type("mesh", value, discretize.base.BaseMesh, cast=False)

    @property
    def solver(self):
        r"""Numerical solver used in the forward simulation.

        Many forward simulations in SimPEG require solutions to discrete linear
        systems of the form:

        .. math::
            \mathbf{A}(\mathbf{m}) \, \mathbf{u} = \mathbf{q}

        where :math:`\mathbf{A}` is an invertible matrix that depends on the
        model :math:`\mathbf{m}`. The numerical solver can be set using the
        ``solver`` property. In SimPEG, the
        `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`__ package
        is used to create solver objects. Parameters specific to each solver
        can be set manually using the ``solver_opts`` property.

        Returns
        -------
        type[pymatsolver.solvers.Base]
            Numerical solver used to solve the forward problem.
        """
        return self._solver

    @solver.setter
    def solver(self, cls):
        if cls is not None:
            if not inspect.isclass(cls):
                raise TypeError(f"{type(self).__qualname__}.solver must be a class")
            if not issubclass(cls, pymatsolver.solvers.Base):
                raise TypeError(
                    f"{cls.__qualname__} is not a subclass of pymatsolver.base.BaseSolver"
                )
        if cls in (pymatsolver.SolverLU, pymatsolver.Solver):
            warnings.warn(
                f"The 'pymatsolver.{cls.__name__}' solver might lead to high "
                "computation times. "
                "We recommend using a faster alternative such as 'pymatsolver.Pardiso' "
                "or 'pymatsolver.Mumps'.",
                PerformanceWarning,
                stacklevel=2,
            )
        self._solver = cls

    @property
    def solver_opts(self):
        """Solver-specific parameters.

        The parameters specific to the solver set with the ``solver`` property are set
        upon instantiation. The ``solver_opts`` property is used to set solver-specific properties.
        This is done by providing a ``dict`` that contains appropriate pairs of keyword arguments
        and parameter values. Please visit `pymatsolver <https://pymatsolver.readthedocs.io/en/latest/>`__
        to learn more about solvers and their parameters.

        Returns
        -------
        dict
            keyword arguments and parameters passed to the solver.
        """
        return self._solver_opts

    @solver_opts.setter
    def solver_opts(self, value):
        self._solver_opts = validate_type("solver_opts", value, dict, cast=False)

    @property
    def Vol(self):
        return self.Mcc

    @property
    def Mcc(self):
        """
        Cell center inner product matrix.
        """
        if getattr(self, "_Mcc", None) is None:
            self._Mcc = sp.diags(self.mesh.cell_volumes, format="csr")
        return self._Mcc

    @property
    def Mn(self):
        """
        Node inner product matrix.
        """
        if getattr(self, "_Mn", None) is None:
            vol = self.mesh.cell_volumes
            self._Mn = sp.diags(self.mesh.aveN2CC.T * vol, format="csr")
        return self._Mn

    @property
    def Mf(self):
        """
        Face inner product matrix.
        """
        if getattr(self, "_Mf", None) is None:
            self._Mf = self.mesh.get_face_inner_product()
        return self._Mf

    @property
    def Me(self):
        """
        Edge inner product matrix.
        """
        if getattr(self, "_Me", None) is None:
            self._Me = self.mesh.get_edge_inner_product()
        return self._Me

    @property
    def MccI(self):
        if getattr(self, "_MccI", None) is None:
            self._MccI = sp.diags(1.0 / self.mesh.cell_volumes, format="csr")
        return self._MccI

    @property
    def MnI(self):
        """
        Node inner product inverse matrix.
        """
        if getattr(self, "_MnI", None) is None:
            vol = self.mesh.cell_volumes
            self._MnI = sp.diags(1.0 / (self.mesh.aveN2CC.T * vol), format="csr")
        return self._MnI

    @property
    def MfI(self):
        """
        Face inner product inverse matrix.
        """
        if getattr(self, "_MfI", None) is None:
            self._MfI = self.mesh.get_face_inner_product(invert_matrix=True)
        return self._MfI

    @property
    def MeI(self):
        """
        Edge inner product inverse matrix.
        """
        if getattr(self, "_MeI", None) is None:
            self._MeI = self.mesh.get_edge_inner_product(invert_matrix=True)
        return self._MeI


@with_property_mass_matrices("sigma")
@with_property_mass_matrices("rho")
class BaseElectricalPDESimulation(BasePDESimulation):
    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    def __init__(
        self, mesh, sigma=None, sigmaMap=None, rho=None, rhoMap=None, **kwargs
    ):
        super().__init__(mesh=mesh, **kwargs)
        self.sigma = sigma
        self.rho = rho
        self.sigmaMap = sigmaMap
        self.rhoMap = rhoMap

    @property
    def _delete_on_model_update(self):
        """
        matrices to be deleted if the model for conductivity/resistivity is updated
        """
        toDelete = super()._delete_on_model_update
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete = (
                toDelete + self._clear_on_sigma_update + self._clear_on_rho_update
            )
        return toDelete

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["sigma", "rho"]:
            for mat in self._clear_on_sigma_update + self._clear_on_rho_update:
                if hasattr(self, mat):
                    delattr(self, mat)


@with_property_mass_matrices("mu")
@with_property_mass_matrices("mui")
class BaseMagneticPDESimulation(BasePDESimulation):
    mu, muMap, muDeriv = props.Invertible(
        "Magnetic Permeability (H/m)",
    )
    mui, muiMap, muiDeriv = props.Invertible("Inverse Magnetic Permeability (m/H)")
    props.Reciprocal(mu, mui)

    def __init__(self, mesh, mu=mu_0, muMap=None, mui=None, muiMap=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.mu = mu
        self.mui = mui
        self.muMap = muMap
        self.muiMap = muiMap

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["mu", "mui"]:
            for mat in self._clear_on_mu_update + self._clear_on_mui_update:
                if hasattr(self, mat):
                    delattr(self, mat)

    @property
    def _delete_on_model_update(self):
        """
        items to be deleted if the model for Magnetic Permeability is updated
        """
        toDelete = super()._delete_on_model_update
        if self.muMap is not None or self.muiMap is not None:
            toDelete = toDelete + self._clear_on_mu_update + self._clear_on_mui_update
        return toDelete
