import numpy as np

from . import (
    BaseSaveGeoH5,
    InversionDirective,
    SaveModelGeoH5,
    SphericalUnitsWeights,
    Update_IRLS,
    UpdateIRLS,
    UpdateSensitivityWeights,
)
from ..maps import SphericalSystem, Wires
from ..meta.simulation import MetaSimulation
from ..objective_function import ComboObjectiveFunction
from ..regularization import CrossGradient
from ..utils.mat_utils import cartesian2amplitude_dip_azimuth
from ..utils import set_kwargs, spherical2cartesian, cartesian2spherical


class ProjectSphericalBounds(InversionDirective):
    r"""
    Trick for spherical coordinate system.
    Project \theta and \phi angles back to [-\pi,\pi] using
    back and forth conversion.
    spherical->cartesian->spherical
    """

    def __init__(self, mapping: Wires, **kwargs):
        if not isinstance(mapping, Wires):
            raise TypeError("mapping must be a Wires object")

        if len(mapping.maps) != 3:
            raise ValueError("mapping must have 3 maps, one per vector component.")

        self.indices = mapping.deriv(None).indices
        super().__init__(**kwargs)

    def initialize(self):
        self.update()

    def endIter(self):
        self.update()

    def update(self):
        """
        Update the model and the simulation
        """
        x = self.invProb.model
        m = self._reproject(x)
        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        self.invProb.phi_m_last = phi_m_last
        self.invProb.model = m
        self.opt.xc = self.invProb.model

        for misfit in self.dmisfit.objfcts:
            misfit.simulation.model = m

    def _reproject(self, m):
        """
        Round trip conversion to reproject the model.
        """
        vec = m[self.indices]
        xyz = spherical2cartesian(vec.reshape((-1, 3), order="F"))
        vec = cartesian2spherical(xyz.reshape((-1, 3), order="F"))

        m[self.indices] = vec
        return m


class VectorInversion(InversionDirective):
    """
    Control a vector inversion from Cartesian to spherical coordinates.
    """

    chifact_target = 1.0
    reference_model = None
    mode = "cartesian"
    inversion_type = "mvis"
    norms = []
    alphas = []
    cartesian_model = None
    mappings = []
    regularization = []

    def __init__(
        self, simulations: list, regularizations: ComboObjectiveFunction, **kwargs
    ):
        self.reference_angles = (False, False, False)
        self.simulations = simulations
        self.regularizations = regularizations

        set_kwargs(self, **kwargs)

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD

            self._target = nD * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def initialize(self):
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model

        self.reference_model = reg.reference_model

        for dmisfit in self.dmisfit.objfcts:
            if getattr(dmisfit.simulation, "coordinate_system", None) is not None:
                dmisfit.simulation.coordinate_system = self.mode

    def endIter(self):

        model = self.invProb.model.copy()

        # Project bounds
        if self.mode == "cartesian" and np.any(~np.isinf(self.opt.upper)):
            vec_model = []
            indices = []
            upper_bound = []
            for reg in self.regularizations.objfcts:
                vec_model.append(reg.mapping * model)
                upper_bound.append(reg.mapping * self.opt.upper)
                mapping = reg.mapping.deriv(np.zeros(reg.mapping.shape[1]))
                indices.append(mapping.indices)

            amplitude = np.linalg.norm(np.vstack(vec_model), axis=0)
            upper_bound = np.hstack(upper_bound)
            upper_bound = np.max(upper_bound[~np.isinf(upper_bound)])
            out_bound = amplitude > upper_bound

            if np.any(out_bound):
                scale = upper_bound / amplitude
                for ind in indices:
                    model[ind] *= scale

                self.invProb.model = model
                self.opt.xc = model

        if (
            self.invProb.phi_d < self.target
        ) and self.mode == "cartesian":  # and self.inversion_type == 'mvis':
            print("Switching MVI to spherical coordinates")
            self.mode = "spherical"
            self.cartesian_model = model

            vec_model = []
            vec_ref = []
            indices = []
            mappings = []
            for reg in self.regularizations.objfcts:
                mappings.append(reg.mapping)
                vec_model.append(reg.mapping * model)
                vec_ref.append(reg.mapping * reg.reference_model)
                mapping = reg.mapping.deriv(np.zeros(reg.mapping.shape[1]))
                indices.append(mapping.indices)

            indices = np.hstack(indices)
            nC = mapping.shape[0]
            vec_model = cartesian2spherical(np.vstack(vec_model).T)
            vec_ref = cartesian2spherical(np.vstack(vec_ref).T).flatten()
            model[indices] = vec_model.flatten()

            angle_map = []
            for ind, (reg_fun, ref_angle) in enumerate(
                zip(self.regularizations.objfcts, self.reference_angles)
            ):
                reg_fun.model = model
                reg_fun.reference_model[indices] = vec_ref

                if ind > 0:
                    if not ref_angle:
                        reg_fun.alpha_s = 0

                    reg_fun.eps_q = np.pi
                    reg_fun.units = "radian"
                    angle_map.append(reg_fun.mapping)
                else:
                    reg_fun.units = "amplitude"

            # Change units of cross-gradient on angles
            multipliers = []
            for mult, reg in self.reg:
                if isinstance(reg, CrossGradient):
                    units = []
                    for _, wire in reg.wire_map.maps:
                        if wire in angle_map:
                            units.append("radian")
                            mult = 0  # TODO Make this optional
                        else:
                            units.append("metric")

                    reg.units = units

                multipliers.append(mult)

            self.reg.multipliers = multipliers
            self.invProb.beta *= 2
            self.invProb.model = model
            self.opt.xc = model
            self.opt.lower[indices] = np.kron(
                np.asarray([0, -np.inf, -np.inf]), np.ones(nC)
            )
            self.opt.upper[indices[nC:]] = np.inf

            updates = {}
            for simulation in self.simulations:
                if isinstance(simulation, MetaSimulation):

                    if hasattr(self.dmisfit, "client"):
                        updates[simulation] = (
                            "chiMap",
                            SphericalSystem() * simulation.simulations[0].chiMap,
                        )
                    else:
                        simulation.simulations[0].chiMap = (
                            SphericalSystem() * simulation.simulations[0].chiMap
                        )
                else:
                    simulation.chiMap = SphericalSystem() * simulation.chiMap

            if hasattr(self.dmisfit, "client"):
                self.dmisfit.broadcast_updates(updates)

            # Add and update directives
            for directive in self.inversion.directiveList.dList:
                if (
                    isinstance(directive, SaveModelGeoH5)
                    and cartesian2amplitude_dip_azimuth in directive.transforms
                ):
                    transforms = []

                    for fun in directive.transforms:
                        if fun is cartesian2amplitude_dip_azimuth:
                            transforms += [spherical2cartesian]
                        transforms += [fun]

                    directive.transforms = transforms

                elif isinstance(directive, Update_IRLS | UpdateIRLS):
                    directive.sphericalDomain = True
                    directive.model = model
                    directive.coolingFactor = 1.5

                elif isinstance(directive, UpdateSensitivityWeights):
                    directive.every_iteration = True

            spherical_units = SphericalUnitsWeights(
                amplitude=self.regularizations.objfcts[0].mapping,
                angles=self.regularizations.objfcts[1:],
            )
            projections = [(comp, mapping) for comp, mapping in zip("xyz", mappings)]
            directiveList = [
                ProjectSphericalBounds(Wires(*projections)),
                spherical_units,
            ] + self.inversion.directiveList.dList
            self.inversion.directiveList = directiveList

            for directive in directiveList:
                if not isinstance(directive, BaseSaveGeoH5):
                    directive.endIter()
