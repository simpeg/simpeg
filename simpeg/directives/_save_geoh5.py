import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from .directives import InversionDirective
from simpeg.maps import IdentityMap

from geoh5py.data import NumericData
from geoh5py.groups.property_group import GroupTypeEnum
from geoh5py.groups import UIJsonGroup
from geoh5py.objects import ObjectBase
from geoh5py.ui_json.utils import fetch_active_workspace


def compute_JtJdiags(data_misfit, m):
    if hasattr(data_misfit, "getJtJdiag"):
        return data_misfit.getJtJdiag(m)
    else:
        jtj_diags = []
        for dmisfit in data_misfit.objfcts:
            jtj_diags.append(dmisfit.getJtJdiag(m))

        jtj_diag = np.zeros_like(jtj_diags[0])
        for multiplier, diag in zip(data_misfit.multipliers, jtj_diags):
            jtj_diag += multiplier * diag

    return np.asarray(jtj_diag)


class BaseSaveGeoH5(InversionDirective, ABC):
    """
    Base class for saving inversion results to a geoh5 file
    """

    def __init__(
        self,
        h5_object,
        dmisfit=None,
        label: str | None = None,
        channels: list[str] = ("",),
        components: list[str] = ("",),
        association: str | None = None,
        **kwargs,
    ):
        self.label = label
        self.channels = channels
        self.components = components
        self.h5_object = h5_object

        if association is not None:
            self.association = association

        super().__init__(
            inversion=None, dmisfit=dmisfit, reg=None, verbose=False, **kwargs
        )

    def initialize(self):
        self.write(0)

    def endIter(self):
        self.write(self.opt.iter)

    def get_names(
        self, component: str, channel: str, iteration: int
    ) -> tuple[str, str]:
        """
        Format the data and property_group name.
        """
        base_name = f"Iteration_{iteration}"
        if len(component) > 0:
            base_name += f"_{component}"

        channel_name = base_name
        if channel:
            channel_name += f"_{channel}"

        if self.label is not None:
            channel_name += f"_{self.label}"
            base_name += f"_{self.label}"

        return channel_name, base_name

    @abstractmethod
    def write(self, iteration: int, values: list[np.ndarray] = None):  # flake8: noqa
        """
        Save the components of the inversion.
        """

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value: str | None):
        if not isinstance(value, str | type(None)):
            raise TypeError("'label' must be a string or None")

        self._label = value

    @property
    def h5_object(self):
        return self._h5_object

    @h5_object.setter
    def h5_object(self, entity: ObjectBase):
        if not isinstance(entity, ObjectBase | UIJsonGroup):
            raise TypeError(
                f"Input entity should be of type {ObjectBase}. {type(entity)} provided"
            )

        self._h5_object = entity.uid
        self._geoh5 = entity.workspace

        if getattr(entity, "n_cells", None) is not None:
            self.association = "CELL"
        else:
            self.association = "VERTEX"

    @property
    def association(self):
        return self._association

    @association.setter
    def association(self, value):
        if not value.upper() in ["CELL", "VERTEX"]:
            raise ValueError(
                f"'association must be one of 'CELL', 'VERTEX'. {value} provided"
            )

        self._association = value.upper()


class SaveArrayGeoH5(BaseSaveGeoH5, ABC):
    """
    Saves array-based inversion results (model, data) to a geoh5 file.

    Parameters
    ----------

    transforms: List of transformations applied to the values before save.
    sorting: Special re-indexing of the vector values.
    reshape: Re-ordering applied to the data before slicing.
    """

    _attribute_type = None

    def __init__(
        self,
        h5_object,
        transforms: list | tuple = (),
        reshape=None,
        sorting=None,
        **kwargs,
    ):
        self.data_type = {}
        self.transforms = transforms
        self.sorting = sorting
        self.reshape = reshape

        super().__init__(h5_object, **kwargs)

    @property
    def reshape(self):
        """
        Reshape function
        """
        if getattr(self, "_reshape", None) is None:
            self._reshape = lambda x: x.reshape(
                (len(self.channels), len(self.components), -1), order="F"
            )

        return self._reshape

    @reshape.setter
    def reshape(self, fun):
        self._reshape = fun

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, funcs: list | tuple):
        if not isinstance(funcs, list | tuple):
            funcs = [funcs]

        for fun in funcs:
            if not any(
                [
                    isinstance(
                        fun, (IdentityMap, np.ndarray, csr_matrix, csc_matrix, float)
                    ),
                    callable(fun),
                ]
            ):
                raise TypeError(
                    "Input transformation must be of type"
                    + "SimPEG.maps, numpy.ndarray or callable function"
                )

        self._transforms = funcs

    def stack_channels(self, dpred: list):
        """
        Regroup channel values along rows.
        """
        if isinstance(dpred, np.ndarray):
            return self.reshape(dpred)

        return self.reshape(np.hstack(dpred))

    def apply_transformations(self, prop: np.ndarray) -> np.ndarray:
        """
        Re-order the values and apply transformations.
        """
        prop = prop.flatten()
        for fun in self.transforms:
            if isinstance(
                fun, (IdentityMap, np.ndarray, csr_matrix, csc_matrix, float)
            ):
                prop = fun * prop
            else:
                prop = fun(prop)

        if prop.ndim == 2:
            prop = prop.T.flatten()

        prop = prop.reshape((len(self.channels), len(self.components), -1))

        return prop

    @abstractmethod
    def get_values(self, values: list[np.ndarray] | None):
        """
        Get values for the inversion depending on the output type.
        """

    def write(self, iteration: int, values: list[np.ndarray] = None):  # flake8: noqa
        """
        Sort, transform and store data per components and channels.
        """
        prop = self.get_values(values)

        # Apply transformations
        prop = self.apply_transformations(prop)

        # Save results
        with fetch_active_workspace(self._geoh5, mode="r+") as w_s:
            h5_object = w_s.get_entity(self.h5_object)[0]
            for cc, component in enumerate(self.components):
                if component not in self.data_type:
                    self.data_type[component] = {}

                for ii, channel in enumerate(self.channels):
                    values = prop[ii, cc, :]

                    if self.sorting is not None:
                        values = values[self.sorting]

                    channel_name, base_name = self.get_names(
                        component, channel, iteration
                    )

                    data = h5_object.add_data(
                        {
                            channel_name: {
                                "association": self.association,
                                "values": values,
                            }
                        }
                    )
                    # Re-assign the data type
                    if channel not in self.data_type[component].keys():
                        self.data_type[component][channel] = data.entity_type
                        type_name = f"{self._attribute_type}_{component}"
                        if channel:
                            type_name += f"_{channel}"
                        data.entity_type.name = type_name
                    else:
                        data.entity_type = w_s.find_type(
                            self.data_type[component][channel].uid,
                            type(self.data_type[component][channel]),
                        )


class SaveModelGeoH5(SaveArrayGeoH5):
    """
    Save the model at the current iteration to a geoh5 file.
    """

    _attribute_type = "model"

    def get_values(self, values: list[np.ndarray] | None):
        if values is None:
            values = self.invProb.model

        return values


class SaveSensitivityGeoH5(SaveArrayGeoH5):
    """
    Save the model at the current iteration to a geoh5 file.
    """

    _attribute_type = "sensitivities"

    def __init__(self, h5_object, dmisfit=None, **kwargs):
        if dmisfit is None:
            raise ValueError(
                "To save sensitivities, the data misfit object must be provided."
            )
        super().__init__(h5_object, dmisfit=dmisfit, **kwargs)

    def get_values(self, values: list[np.ndarray] | None):
        if values is None:
            values = compute_JtJdiags(self.dmisfit, self.invProb.model)

        return values


class SaveDataGeoH5(SaveArrayGeoH5):
    """
    Save the model at the current iteration to a geoh5 file.
    """

    _attribute_type = "predicted"

    def __init__(self, h5_object, joint_index: list[int] | None = None, **kwargs):
        self.joint_index = joint_index

        super().__init__(h5_object, **kwargs)

    def get_values(self, values: list[np.ndarray] | None):

        if values is not None:
            prop = self.stack_channels(values)

        else:
            dpred = getattr(self.invProb, "dpred", None)
            if dpred is None:
                dpred = self.invProb.get_dpred(self.invProb.model)
                self.invProb.dpred = dpred

            if self.joint_index is not None:
                dpred = [dpred[ind] for ind in self.joint_index]

            prop = self.stack_channels(dpred)

        return prop

    @property
    def joint_index(self):
        """
        Index for joint inversions defining the element in the list of predicted data.
        """
        return self._joint_index

    @joint_index.setter
    def joint_index(self, value: list[int] | None):
        if not isinstance(value, list | type(None)):
            raise TypeError("Input 'joint_index' should be a list of int")

        self._joint_index = value


class SaveLogFilesGeoH5(BaseSaveGeoH5):

    def write(self, iteration: int, **_):
        dirpath = Path(self._geoh5.h5file).parent
        filepath = dirpath / "SimPEG.out"

        if iteration == 0:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("iteration beta phi_d phi_m time\n")
        log = []
        with open(dirpath / "SimPEG.log", "r", encoding="utf-8") as file:
            iteration = 0
            for line in file:
                val = re.findall(r"[+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d+)", line)
                if len(val) == 5:
                    log.append(val[:-2])
                    iteration += 1

        if len(log) > 0:
            with open(filepath, "a", encoding="utf-8") as file:
                date_time = datetime.now().strftime("%b-%d-%Y:%H:%M:%S")
                file.write(f"{iteration-1} " + " ".join(log[-1]) + f" {date_time}\n")

        self.save_log()

    def save_log(self):
        """
        Save iteration metrics to comments.
        """
        dirpath = Path(self._geoh5.h5file).parent

        with fetch_active_workspace(self._geoh5, mode="r+") as w_s:
            h5_object = w_s.get_entity(self.h5_object)[0]

            for file in ["SimPEG.out", "SimPEG.log", "ChiFactors.log"]:
                filepath = dirpath / file

                if not filepath.is_file():
                    continue

                with open(filepath, "rb") as f:
                    raw_file = f.read()

                file_entity = h5_object.get_entity(file)[0]
                if file_entity is None:
                    file_entity = h5_object.add_file(filepath)

                file_entity.file_bytes = raw_file


class SavePropertyGroup(BaseSaveGeoH5):
    """
    Assign the data to a property group in the geoh5 file
    """

    def __init__(
        self,
        h5_object,
        group_type: GroupTypeEnum = GroupTypeEnum.MULTI,
        **kwargs,
    ):
        self.group_type = group_type

        super().__init__(h5_object, **kwargs)

    def write(self, iteration: int, **_):
        """
        Save the model to the geoh5 file
        """
        with fetch_active_workspace(self._geoh5, mode="r+") as w_s:
            h5_object = w_s.get_entity(self.h5_object)[0]

            for component in self.components:
                properties = []
                for channel in self.channels:

                    channel_name, base_name = self.get_names(
                        component, channel, iteration
                    )
                    children = [
                        child
                        for child in h5_object.children
                        if (
                            channel_name in child.name
                            and isinstance(child, NumericData)
                        )
                    ]

                    if children[0] is not None:
                        properties += children

                if len(properties) == 0:
                    return

                prop_group = h5_object.get_property_group(base_name)[0]

                if prop_group is None:
                    prop_group = h5_object.create_property_group(
                        name=base_name,
                        properties=properties,
                        property_group_type=self.group_type,
                    )
                else:
                    prop_group.add_properties(properties)


class SaveLPModelGroup(SavePropertyGroup):
    """
    Save the model as a property group in the geoh5 file
    """

    def __init__(
        self,
        h5_object,
        irls_directive,
        group_type: GroupTypeEnum = GroupTypeEnum.MULTI,
        **kwargs,
    ):
        self.group_type = group_type
        self.irls_directive = irls_directive

        super().__init__(h5_object, **kwargs)

    def get_names(
        self, component: str, channel: str, iteration: int
    ) -> tuple[str, str]:
        """
        Format the data and property_group name.
        """
        channel_name, base_name = super().get_names(component, channel, iteration)

        if self.irls_directive.metrics.irls_iteration_count == 0:
            base_name = "L2 models"
        else:
            base_name = "LP models"

        return channel_name, base_name
