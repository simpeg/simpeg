import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints
from ....utils.code_utils import deprecate_property


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """

    current = properties.Float("amplitude of the source current", default=1.0)

    _q = None

    def __init__(self, receiver_list, **kwargs):
        super(BaseSrc, self).__init__(receiver_list, **kwargs)

    def eval(self, prob):
        raise NotImplementedError

    def evalDeriv(self, prob):
        return Zero()


class Dipole(BaseSrc):
    """
    Dipole source
    """

    location = properties.List(
        "location of the source electrodes",
        survey.SourceLocationArray("location of electrode"),
    )
    loc = deprecate_property(
        location, "loc", new_name="location", removal_version="0.15.0"
    )

    def __init__(
        self,
        receiver_list=[],
        a_location=None,
        b_location=None,
        location=None,
        **kwargs,
    ):
        # Check for old keywords
        if "locationA" in kwargs.keys():
            a_location = kwargs.pop("locationA")
            warnings.warn(
                "The locationA property has been deprecated. Please set the "
                "a_location property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )

        if "locationB" in kwargs.keys():
            b_location = kwargs.pop("locationB")
            warnings.warn(
                "The locationB property has been deprecated. Please set the "
                "b_location property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )

        # if a_location set, then use a_location, b_location
        if a_location is not None:
            if b_location is None:
                raise ValueError(
                    "For a dipole source both a_location and b_location " "must be set"
                )

            if location is not None:
                raise ValueError(
                    "Cannot set both location and a_location, b_location. "
                    "Please provide either location=(a_location, b_location) "
                    "or both a_location=a_location, b_location=b_location"
                )

            location = [a_location, b_location]

        elif location is not None:
            if len(location) != 2:
                raise ValueError(
                    "location must be a list or tuple of length 2: "
                    "[location_a, location_b]. The input location has "
                    f"length {len(location)}"
                )

        if location[0].shape != location[1].shape:
            raise ValueError(
                f"m_location (shape: {location[0].shape}) and "
                f"n_location (shape: {location[1].shape}) need to be "
                f"the same size"
            )

        # instantiate
        super(Dipole, self).__init__(receiver_list, **kwargs)
        self.location = location

    @property
    def a_location(self):
        """Location of the A-electrode"""
        return self.location[0]

    @property
    def b_location(self):
        """Location of the B-electrode"""
        return self.location[1]

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == "HJ":
                inds = closestPoints(prob.mesh, self.location, gridLoc="CC")
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1.0, -1.0]
            elif prob._formulation == "EB":
                qa = prob.mesh.getInterpolationMat(
                    self.location[0], locType="N"
                ).toarray()
                qb = -prob.mesh.getInterpolationMat(
                    self.location[1], locType="N"
                ).toarray()
                self._q = self.current * (qa + qb)
            return self._q


class Pole(BaseSrc):
    def __init__(self, receiver_list=[], location=None, **kwargs):
        super(Pole, self).__init__(receiver_list, location=location, **kwargs)

    def eval(self, prob):
        if self._q is not None:
            return self._q
        else:
            if prob._formulation == "HJ":
                inds = closestPoints(prob.mesh, self.location)
                self._q = np.zeros(prob.mesh.nC)
                self._q[inds] = self.current * np.r_[1.0]
            elif prob._formulation == "EB":
                q = prob.mesh.getInterpolationMat(self.location, locType="N")
                self._q = self.current * q.toarray()
            return self._q
