import functools
import inspect
import numpy as np
import pymatsolver
import scipy.sparse as sp
from discretize.utils import Zero, TensorType
import discretize.base

from .physical_property import ElectricalConductivity, MagneticPermeability
from ..simulation import BaseSimulation

from ..utils import validate_type
from ..utils.solver_utils import get_default_solver


def __inner_mat_mul_op(M, u, v=None, adjoint=False):
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


def _Mcc_getter(arg, obj):
    """
    Cell center property inner product matrix.
    """
    stash_name = f"_Mcc_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        M_prop = sp.diags(obj.mesh.cell_volumes * prop, format="csr")
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _Mn_getter(arg, obj):
    """
    Node property inner product matrix.
    """
    stash_name = f"_Mn_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        vol = obj.mesh.cell_volumes
        M_prop = sp.diags(obj.mesh.aveN2CC.T * (vol * prop), format="csr")
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _Mf_getter(arg, obj):
    """
    Face property inner product matrix.
    """
    stash_name = f"_Mf_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        M_prop = obj.mesh.get_face_inner_product(model=prop)
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _Me_getter(arg, obj):
    """
    Edge property inner product matrix.
    """
    stash_name = f"_Me_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        M_prop = obj.mesh.get_edge_inner_product(model=prop)
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _MccI_getter(arg, obj):
    """
    Cell center property inner product inverse matrix.
    """
    stash_name = f"_MccI_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        M_prop = sp.diags(1.0 / (obj.mesh.cell_volumes * prop), format="csr")
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _MnI_getter(arg, obj):
    """
    Node property inner product inverse matrix.
    """
    stash_name = f"_MnI_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        vol = obj.mesh.cell_volumes
        M_prop = sp.diags(1.0 / (obj.mesh.aveN2CC.T * (vol * prop)), format="csr")
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _MfI_getter(arg, obj):
    """
    Face property inner product inverse matrix.
    """
    stash_name = f"_MfI_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        M_prop = obj.mesh.get_face_inner_product(model=prop, invert_matrix=True)
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _MeI_getter(arg, obj):
    """
    Edge property inner product inverse matrix.
    """
    stash_name = f"_MeI_{arg}"
    if getattr(obj, stash_name, None) is None:
        prop = getattr(obj, arg.lower())
        M_prop = obj.mesh.get_edge_inner_product(model=prop, invert_matrix=True)
        setattr(obj, stash_name, M_prop)
    return getattr(obj, stash_name)


def _MccDeriv_func(obj, arg, u, v=None, adjoint=False):
    """
    Derivative of `MccProperty` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()
    stash_name = f"_Mcc_{arg}_deriv"

    if getattr(obj, stash_name, None) is None:
        prop_deriv = obj._prop_deriv(arg.lower())
        if not isinstance(prop_deriv, Zero):
            prop_deriv = sp.diags(obj.mesh.cell_volumes) * prop_deriv
        setattr(obj, stash_name, prop_deriv)

    M_prop_deriv = getattr(obj, stash_name)
    if isinstance(M_prop_deriv, Zero):
        return Zero()
    return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)


def _MnDeriv_func(obj, arg, u, v=None, adjoint=False):
    """
    Derivative of `MnProperty` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()

    stash_name = f"_Mn_{arg}_deriv"
    if getattr(obj, stash_name, None) is None:
        prop_deriv = obj._prop_deriv(arg.lower())
        if not isinstance(prop_deriv, Zero):
            prop_deriv = (
                obj.mesh.aveN2CC.T * sp.diags(obj.mesh.cell_volumes) * prop_deriv
            )
        setattr(obj, stash_name, prop_deriv)

    M_prop_deriv = getattr(obj, stash_name)
    if isinstance(M_prop_deriv, Zero):
        return Zero()
    return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)


def _MfDeriv_func(obj, arg, u, v=None, adjoint=False):
    """
    Derivative of `MfProperty` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()
    stash_name = f"_Mf_{arg}_deriv"
    if getattr(obj, stash_name, None) is None:
        prop_deriv = obj._prop_deriv(arg.lower())
        if isinstance(prop_deriv, Zero):
            setattr(obj, stash_name, prop_deriv)
        else:
            prop = getattr(obj, arg.lower())
            t_type = TensorType(obj.mesh, prop)

            M_deriv_func = obj.mesh.get_face_inner_product_deriv(model=prop)
            # t_type == 3 for full tensor model, t_type < 3 for scalar, isotropic, or axis-aligned anisotropy.
            if t_type < 3 and obj.mesh._meshType.lower() in (
                "cyl",
                "tensor",
                "tree",
            ):
                M_prop_deriv = M_deriv_func(np.ones(obj.mesh.n_faces)) @ prop_deriv
                setattr(obj, stash_name, M_prop_deriv)
            else:
                setattr(obj, stash_name, (M_deriv_func, prop_deriv))

    M_prop_deriv = getattr(obj, stash_name)
    if isinstance(M_prop_deriv, Zero):
        return Zero()
    return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)


def _MeDeriv_func(obj, arg, u, v=None, adjoint=False):
    """
    Derivative of `MeProperty` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()
    stash_name = f"_Me_{arg}_deriv"
    if getattr(obj, stash_name, None) is None:
        prop_deriv = obj._prop_deriv(arg.lower())
        if isinstance(prop_deriv, Zero):
            setattr(obj, stash_name, prop_deriv)
        else:
            prop = getattr(obj, arg.lower())
            t_type = TensorType(obj.mesh, prop)

            M_deriv_func = obj.mesh.get_edge_inner_product_deriv(model=prop)
            # t_type == 3 for full tensor model, t_type < 3 for scalar, isotropic, or axis-aligned anisotropy.
            if t_type < 3 and obj.mesh._meshType.lower() in (
                "cyl",
                "tensor",
                "tree",
            ):
                M_prop_deriv = M_deriv_func(np.ones(obj.mesh.n_edges)) @ prop_deriv
                setattr(obj, stash_name, M_prop_deriv)
            else:
                setattr(obj, stash_name, (M_deriv_func, prop_deriv))

    M_prop_deriv = getattr(obj, stash_name)
    if isinstance(M_prop_deriv, Zero):
        return Zero()
    return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)


def _MccIDeriv_func(obj, arg, u, v=None, adjoint=False):
    """
    Derivative of `MccPropertyI` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()

    MI_prop = getattr(obj, f"Mcc{arg}I")
    u = MI_prop @ (MI_prop @ -u)
    M_prop_deriv = getattr(obj, f"Mcc{arg}Deriv")
    return M_prop_deriv(u, v, adjoint=adjoint)


def _MnIDeriv_func(obj, arg, u, v=None, adjoint=False):
    """
    Derivative of `MnPropertyI` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()

    MI_prop = getattr(obj, f"Mn{arg}I")
    u = MI_prop @ (MI_prop @ -u)
    M_prop_deriv = getattr(obj, f"Mn{arg}Deriv")
    return M_prop_deriv(u, v, adjoint=adjoint)


def _MfIDeriv_func(obj, arg, u, v=None, adjoint=False):
    """I
    Derivative of `MfPropertyI` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()

    MI_prop = getattr(obj, f"Mf{arg}I")
    u = MI_prop @ (MI_prop @ -u)
    M_prop_deriv = getattr(obj, f"Mf{arg}Deriv")
    return M_prop_deriv(u, v, adjoint=adjoint)


def _MeIDeriv_func(obj, arg, u, v=None, adjoint=False):
    """
    Derivative of `MePropertyI` with respect to the model.
    """
    if isinstance(u, Zero) or isinstance(v, Zero):
        return Zero()

    MI_prop = getattr(obj, f"Me{arg}I")
    u = MI_prop @ (MI_prop @ -u)
    M_prop_deriv = getattr(obj, f"Me{arg}Deriv")
    return M_prop_deriv(u, v, adjoint=adjoint)


def _clear_on_fset(items, fset, obj, value):
    fset(obj, value)
    for item in items:
        if hasattr(obj, item):
            delattr(obj, item)


def _clear_on_fdel(items, fdel, obj):
    fdel(obj)
    for item in items:
        if hasattr(obj, item):
            delattr(obj, item)


def _delete_on_model_update_getter(items, prop, fget, obj):
    other_items = fget(obj)
    if prop.is_mapped(obj):
        other_items += items
    return other_items


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
    arg = property_name.lower()
    arg = arg[0].upper() + arg[1:]

    cached_items = [
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

    def decorator(cls):
        prop = getattr(cls, property_name)
        new_prop = prop.shallow_copy()
        new_prop.reciprocal = prop.reciprocal

        new_prop.fset = functools.partial(_clear_on_fset, cached_items, prop.fset)
        new_prop.fdel = functools.partial(_clear_on_fdel, cached_items, prop.fdel)
        new_prop.__set_name__(cls, prop.__name__)
        setattr(cls, prop.__name__, new_prop)

        old_deleter = cls._delete_on_model_update

        new_deleter = property(
            functools.partial(
                _delete_on_model_update_getter, cached_items, new_prop, old_deleter.fget
            )
        )
        try:
            new_deleter.__name__ = old_deleter.__name__
        except AttributeError:
            pass

        cls._delete_on_model_update = new_deleter

        setattr(cls, f"Mcc{arg}", property(functools.partial(_Mcc_getter, arg)))
        setattr(cls, f"Mn{arg}", property(functools.partial(_Mn_getter, arg)))
        setattr(cls, f"Mf{arg}", property(functools.partial(_Mf_getter, arg)))
        setattr(cls, f"Me{arg}", property(functools.partial(_Me_getter, arg)))
        setattr(cls, f"Mcc{arg}I", property(functools.partial(_MccI_getter, arg)))
        setattr(cls, f"Mn{arg}I", property(functools.partial(_MnI_getter, arg)))
        setattr(cls, f"Mf{arg}I", property(functools.partial(_MfI_getter, arg)))
        setattr(cls, f"Me{arg}I", property(functools.partial(_MeI_getter, arg)))
        setattr(cls, f"Mcc{arg}Deriv", functools.partialmethod(_MccDeriv_func, arg))
        setattr(cls, f"Mn{arg}Deriv", functools.partialmethod(_MnDeriv_func, arg))
        setattr(cls, f"Mf{arg}Deriv", functools.partialmethod(_MfDeriv_func, arg))
        setattr(cls, f"Me{arg}Deriv", functools.partialmethod(_MeDeriv_func, arg))
        setattr(cls, f"Mcc{arg}IDeriv", functools.partialmethod(_MccIDeriv_func, arg))
        setattr(cls, f"Mn{arg}IDeriv", functools.partialmethod(_MnIDeriv_func, arg))
        setattr(cls, f"Mf{arg}IDeriv", functools.partialmethod(_MfIDeriv_func, arg))
        setattr(cls, f"Me{arg}IDeriv", functools.partialmethod(_MeIDeriv_func, arg))

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
        if self._solver is None:
            # do not cache this, in case the user wants to
            # change it after the first time it is requested.
            return get_default_solver(warn=True)
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
class BaseElectricalPDESimulation(BasePDESimulation, ElectricalConductivity):

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh=mesh, **kwargs)


@with_property_mass_matrices("mu")
@with_property_mass_matrices("mui")
class BaseMagneticPDESimulation(BasePDESimulation, MagneticPermeability):

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
