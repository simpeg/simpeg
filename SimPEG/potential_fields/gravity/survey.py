from ...survey import BaseSurvey
from ...utils.code_utils import deprecate_class


class Survey(BaseSurvey):
    """Base Gravity Survey"""

    receiver_locations = None  #: receiver locations
    rxType = None  #: receiver type
    components = ["gz"]

    def __init__(self, source_field, **kwargs):
        self.source_field = source_field
        BaseSurvey.__init__(self, **kwargs)

    def eval(self, fields):
        return fields

    @property
    def nRx(self):
        return self.source_field.receiver_list[0].locations.shape[0]

    @property
    def receiver_locations(self):
        return self.source_field.receiver_list[0].locations

    @property
    def nD(self):
        return len(self.receiver_locations) * len(self.components)

    @property
    def components(self):
        return self.source_field.receiver_list[0].components

    @property
    def Qfx(self):
        if getattr(self, "_Qfx", None) is None:
            self._Qfx = self.prob.mesh.getInterpolationMat(
                self.receiver_locations, "Fx"
            )
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, "_Qfy", None) is None:
            self._Qfy = self.prob.mesh.getInterpolationMat(
                self.receiver_locations, "Fy"
            )
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, "_Qfz", None) is None:
            self._Qfz = self.prob.mesh.getInterpolationMat(
                self.receiver_locations, "Fz"
            )
        return self._Qfz

    def projectFields(self, u):
        """
            This function projects the fields onto the data space.

            First we project our B on to data location

            .. math::

                \mathbf{B}_{rec} = \mathbf{P} \mathbf{B}

            then we take the dot product between B and b_0

            .. math ::

                \\text{TMI} = \\vec{B}_s \cdot \hat{B}_0

        """
        # TODO: There can be some different tyes of data like |B| or B

        gfx = self.Qfx * u["G"]
        gfy = self.Qfy * u["G"]
        gfz = self.Qfz * u["G"]

        fields = {"gx": gfx, "gy": gfy, "gz": gfz}
        return fields


@deprecate_class(removal_version="0.15.0")
class LinearSurvey(Survey):
    pass
