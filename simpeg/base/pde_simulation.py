import discretize.base
import numpy as np
import scipy.sparse as sp
from discretize.base import BaseMesh
from discretize.utils import Zero, TensorType

from ..props import PhysicalPropertyMetaclass, PhysicalProperty
from ..simulation import BaseSimulation
from .. import props
from scipy.constants import mu_0

from ..utils import validate_type

AXIS_ALIGNED_MESH_TYPES = (
    discretize.TensorMesh,
    discretize.TreeMesh,
    discretize.CylindricalMesh,
)


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


def _get_mass_matrix_functions(property_name: str, invertible: bool = False):

    mm_funcs = {}

    @property
    def Mcc_prop(self):
        """
        Cell center property inner product matrix.
        """
        stash_name = f"_Mcc_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            M_prop = sp.diags(self.mesh.cell_volumes * prop, format="csr")
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_Mcc_{property_name}"] = Mcc_prop

    @property
    def Mn_prop(self):
        """
        Node property inner product matrix.
        """
        stash_name = f"_Mn_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            vol = self.mesh.cell_volumes
            M_prop = sp.diags(self.mesh.aveN2CC.T * (vol * prop), format="csr")
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_Mn_{property_name}"] = Mn_prop

    @property
    def Mf_prop(self):
        """
        Face property inner product matrix.
        """
        stash_name = f"_Mf_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            M_prop = self.mesh.get_face_inner_product(model=prop)
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_Mf_{property_name}"] = Mf_prop

    @property
    def Me_prop(self):
        """
        Edge property inner product matrix.
        """
        stash_name = f"_Me_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            M_prop = self.mesh.get_edge_inner_product(model=prop)
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_Me_{property_name}"] = Me_prop

    @property
    def inv_Mcc_prop(self):
        """
        Cell center property inner product inverse matrix.
        """
        stash_name = f"_inv_Mcc_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            M_prop = sp.diags(1.0 / (self.mesh.cell_volumes * prop), format="csr")
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_inv_Mcc_{property_name}"] = inv_Mcc_prop

    @property
    def inv_Mn_prop(self):
        """
        Node property inner product inverse matrix.
        """
        stash_name = f"_inv_Mn_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            vol = self.mesh.cell_volumes
            M_prop = sp.diags(1.0 / (self.mesh.aveN2CC.T * (vol * prop)), format="csr")
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_inv_Mn_{property_name}"] = inv_Mn_prop

    @property
    def inv_Mf_prop(self):
        """
        Face property inner product inverse matrix.
        """
        stash_name = f"_inv_Mf_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            M_prop = self.mesh.get_face_inner_product(model=prop, invert_matrix=True)
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_inv_Mf_{property_name}"] = inv_Mf_prop

    @property
    def inv_Me_prop(self):
        """
        Edge property inner product inverse matrix.
        """
        stash_name = f"_inv_Me_{property_name}"
        if (M_prop := self._cache[stash_name]) is None:
            prop = getattr(self, property_name)
            M_prop = self.mesh.get_edge_inner_product(model=prop, invert_matrix=True)
            self._cache[stash_name] = M_prop
        return M_prop

    mm_funcs[f"_inv_Me_{property_name}"] = inv_Me_prop

    if invertible:

        def Mcc_prop_deriv(self, u, v=None, adjoint=False):
            f"""
            Derivative of `MccProperty` with respect to the model.
            """
            if getattr(self, f"{property_name}_map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            stash_name = f"_Mcc_{property_name}_deriv"
            if (M_prop_deriv := self._cache[stash_name]) is None:
                prop_deriv = getattr(self, f"{property_name}_deriv")
                M_prop_deriv = sp.diags(self.mesh.cell_volumes) @ prop_deriv
                self._cache[stash_name] = M_prop_deriv
            return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)

        mm_funcs[f"_Mcc_{property_name}_deriv"] = Mcc_prop_deriv

        def Mn_prop_deriv(self, u, v=None, adjoint=False):
            """
            Derivative of `MnProperty` with respect to the model.
            """
            if getattr(self, f"{property_name}_map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mn_{property_name}_deriv"
            if (M_prop_deriv := self._cache[stash_name]) is None:
                prop_deriv = getattr(self, f"{property_name}_deriv")
                M_prop_deriv = (
                    self.mesh.aveN2CC.T @ sp.diags(self.mesh.cell_volumes) @ prop_deriv
                )
                self._cache[stash_name] = M_prop_deriv
            return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)

        mm_funcs[f"_Mn_{property_name}_deriv"] = Mn_prop_deriv

        def Mf_prop_deriv(self, u, v=None, adjoint=False):
            """
            Derivative of `MfProperty` with respect to the model.
            """
            if getattr(self, f"{property_name}_map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Mf_{property_name}_deriv"
            if (M_prop_deriv := self._cache[stash_name]) is None:
                prop = getattr(self, property_name)
                t_type = TensorType(self.mesh, prop)

                M_deriv_func = self.mesh.get_face_inner_product_deriv(model=prop)
                prop_deriv = getattr(self, f"{property_name}_deriv")
                # t_type == 3 for full tensor model, t_type < 3 for scalar, isotropic, or axis-aligned anisotropy.
                if t_type < 3 and isinstance(self.mesh, AXIS_ALIGNED_MESH_TYPES):
                    M_prop_deriv = M_deriv_func(np.ones(self.mesh.n_faces)) @ prop_deriv
                else:
                    M_prop_deriv = (M_deriv_func, prop_deriv)
                self._cache[stash_name] = M_prop_deriv

            return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)

        mm_funcs[f"_Mf_{property_name}_deriv"] = Mf_prop_deriv

        def Me_prop_deriv(self, u, v=None, adjoint=False):
            """
            Derivative of `MeProperty` with respect to the model.
            """
            if getattr(self, f"{property_name}_map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()
            stash_name = f"_Me_{property_name}_deriv"
            if (M_prop_deriv := self._cache[stash_name]) is None:
                prop = getattr(self, property_name)
                t_type = TensorType(self.mesh, prop)

                M_deriv_func = self.mesh.get_edge_inner_product_deriv(model=prop)
                prop_deriv = getattr(self, f"{property_name}_deriv")
                # t_type == 3 for full tensor model, t_type < 3 for scalar, isotropic, or axis-aligned anisotropy.
                if t_type < 3 and isinstance(self.mesh, AXIS_ALIGNED_MESH_TYPES):
                    M_prop_deriv = M_deriv_func(np.ones(self.mesh.n_edges)) @ prop_deriv
                else:
                    M_prop_deriv = (M_deriv_func, prop_deriv)
                self._cache[stash_name] = M_prop_deriv
            return __inner_mat_mul_op(M_prop_deriv, u, v=v, adjoint=adjoint)

        mm_funcs[f"_Me_{property_name}_deriv"] = Me_prop_deriv

        def inv_Mcc_prop_deriv(self, u, v=None, adjoint=False):
            """
            Derivative of `MccPropertyI` with respect to the model.
            """
            if getattr(self, f"{property_name}Map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"_inv_Mcc_{property_name}")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"_Mcc_{property_name}_deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        mm_funcs[f"_inv_Mcc_{property_name}_deriv"] = inv_Mcc_prop_deriv

        def inv_Mn_prop_deriv(self, u, v=None, adjoint=False):
            """
            Derivative of `MnPropertyI` with respect to the model.
            """
            if getattr(self, f"{property_name}_map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"_inv_Mn_{property_name}")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"_Mn_{property_name}_deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        mm_funcs[f"_inv_Mn_{property_name}_deriv"] = inv_Mn_prop_deriv

        def inv_Mf_prop_deriv(self, u, v=None, adjoint=False):
            """I
            Derivative of `MfPropertyI` with respect to the model.
            """
            if getattr(self, f"{property_name}_map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"_inv_Mf_{property_name}")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"_Mf_{property_name}_deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        mm_funcs[f"_inv_Mf_{property_name}_deriv"] = inv_Mf_prop_deriv

        def inv_Me_prop_deriv(self, u, v=None, adjoint=False):
            """
            Derivative of `MePropertyI` with respect to the model.
            """
            if getattr(self, f"{property_name}_map") is None:
                return Zero()
            if isinstance(u, Zero) or isinstance(v, Zero):
                return Zero()

            MI_prop = getattr(self, f"_inv_Me_{property_name}")
            u = MI_prop @ (MI_prop @ -u)
            M_prop_deriv = getattr(self, f"_Me_{property_name}_deriv")
            return M_prop_deriv(u, v, adjoint=adjoint)

        mm_funcs[f"_inv_Me_{property_name}_deriv"] = inv_Me_prop_deriv

    cached_items = {
        f"_Mcc_{property_name}",
        f"_Mn_{property_name}",
        f"_Mf_{property_name}",
        f"_Me_{property_name}",
        f"_inv_Mc_{property_name}",
        f"_inv_Mn_{property_name}",
        f"_inv_Mf_{property_name}",
        f"_inv_Me_{property_name}",
    }
    if invertible:
        cached_items |= {
            f"_Mcc_{property_name}_deriv",
            f"_Mn_{property_name}_deriv",
            f"_Mf_{property_name}_deriv",
            f"_Me_{property_name}_deriv",
        }

    return mm_funcs, cached_items


class MassMatrixMeta(PhysicalPropertyMetaclass):

    def __new__(cls, name, bases, attrs: dict):

        cls = super().__new__(cls, name, bases, attrs)
        phys_props = {}
        for base in reversed(cls.__mro__[1:]):
            metaclass = type(base)
            if issubclass(metaclass, PhysicalPropertyMetaclass):
                for prop_name, prop in base._physical_properties.items():
                    # loop through properties that have not already been
                    # given a mass matrix function.
                    if getattr(cls, f"_Mcc_{prop_name}", None) is None:
                        phys_props[prop_name] = prop
        # always update with anything specifically defined on this class.
        phys_props.update(cls._physical_properties)

        mm_funcs = {}
        invertible_props = set()
        for prop_name, prop in phys_props.items():
            invertible = prop.mapping is not None
            mm_func, prop_cache = _get_mass_matrix_functions(
                prop.name, invertible=invertible
            )
            mm_funcs.update(mm_func)
            prop.cached_items = prop_cache
            if invertible:
                invertible_props.add(prop)

        for name, func in mm_funcs.items():
            setattr(cls, name, func)

        model_change_delete_prop = attrs.get("_delete_on_model_change", None)

        @property
        def delete_on_model_change(self):
            if model_change_delete_prop is not None:
                items = model_change_delete_prop.fget(self)
            else:
                items = super(bases[0], self)._delete_on_model_change
            for prop in invertible_props:
                if getattr(self, prop.mapping.name, None) is not None:
                    items.extend(prop.cached_items)
            return items

        cls._delete_on_model_change = delete_on_model_change
        return cls


class BasePDESimulation(BaseSimulation, metaclass=MassMatrixMeta):
    """
    Parameters
    ----------
    mesh : discretize.base.BaseMesh, optional
        Mesh on which the forward problem is discretized.
    solver : None or pymatsolver.base.Base, optional
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
        super().__init__(**kwargs)
        self.mesh = mesh
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
        self._mesh = validate_type("mesh", value, BaseMesh, cast=False)

    @property
    def _Mcc(self):
        """
        Cell center inner product matrix.
        """
        if (Mcc := self._cache["_Mcc"]) is None:
            Mcc = sp.diags(self.mesh.cell_volumes, format="csr")
            self._cache["_Mcc"] = Mcc
        return Mcc

    @property
    def _Mn(self):
        """
        Node inner product matrix.
        """
        if (Mn := self._cache["_Mn"]) is None:
            Mn = sp.diags(self.mesh.aveN2CC.T * self.mesh.cell_volumes, format="csr")
            self._cache["_Mn"] = Mn
        return Mn

    @property
    def _Mf(self):
        """
        Face inner product matrix.
        """
        if (Mf := self._cache["_Mf"]) is None:
            Mf = self.mesh.get_face_inner_product()
            self._cache["_Mf"] = Mf
        return Mf

    @property
    def _Me(self):
        """
        Edge inner product matrix.
        """
        if (Me := self._cache["_Me"]) is None:
            Me = self.mesh.get_face_inner_product()
            self._cache["_Me"] = Me
        return Me

    @property
    def _MccI(self):
        if (MccI := self._cache["_MccI"]) is None:
            MccI = sp.diags(1.0 / self.mesh.cell_volumes, format="csr")
            self._cache["_MccI"] = MccI
        return MccI

    @property
    def _MnI(self):
        """
        Node inner product inverse matrix.
        """
        if (MnI := self._cache["_MnI"]) is None:
            MnI = sp.diags(
                1.0 / (self.mesh.aveN2CC.T * self.mesh.cell_volumes), format="csr"
            )
            self._cache["_MnI"] = MnI
        return MnI

    @property
    def _MfI(self):
        """
        Face inner product inverse matrix.
        """
        if (MfI := self._cache["_MfI"]) is None:
            if isinstance(self.mesh, AXIS_ALIGNED_MESH_TYPES):
                MfI = self.mesh.get_face_inner_product(invert_matrix=True)
            else:
                MfI = self.solver(self._Mf, symmetric=True, positive_definite=True)
            self._cache["_MfI"] = MfI
        return MfI

    @property
    def _MeI(self):
        """
        Edge inner product inverse matrix.
        """
        if (MeI := self._cache["_MeI"]) is None:
            if isinstance(self.mesh, AXIS_ALIGNED_MESH_TYPES):
                MeI = self.mesh.get_edge_inner_product(invert_matrix=True)
            else:
                MeI = self.solver(self._Me, symmetric=True, positive_definite=True)
            self._cache["_MeI"] = MeI
        return self._MeI


class BaseElectricalSimulation(BaseSimulation):
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )
    resistivity, resistivity_map, _res_deriv = props.Invertible(
        "Electrical resistivity (Ohm m)"
    )
    props.Reciprocal(conductivity, resistivity)

    def __init__(
        self,
        conductivity=None,
        conductivity_map=None,
        resistivity=None,
        resistivity_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.resistivity = resistivity
        self.conductivity_map = conductivity_map
        self.resistivity_map = resistivity_map


class BaseMagneticSimulation(BaseSimulation):
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )
    resistivity, resistivity_map, _res_deriv = props.Invertible(
        "Electrical resistivity (Ohm m)"
    )
    props.Reciprocal(conductivity, resistivity)

    def __init__(
        self,
        conductivity=None,
        conductivity_map=None,
        resistivity=None,
        resistivity_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.resistivity = resistivity
        self.conductivity_map = conductivity_map
        self.resistivity_map = resistivity_map


class BaseElectricalPDESimulation(BasePDESimulation, BaseElectricalSimulation):
    pass


class BaseMagneticPDESimulation(BasePDESimulation, BaseElectricalSimulation):
    pass
