import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints
from ....utils.code_utils import deprecate_property

import warnings


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
        location_a=None,
        location_b=None,
        location=None,
        **kwargs,
    ):
        # Check for old keywords
        if "locationA" in kwargs.keys():
            location_a = kwargs.pop("locationA")
            warnings.warn(
                "The locationA property has been deprecated. Please set the "
                "location_a property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )

        if "locationB" in kwargs.keys():
            location_b = kwargs.pop("locationB")
            warnings.warn(
                "The locationB property has been deprecated. Please set the "
                "location_b property instead. This will be removed in version"
                " 0.15.0 of SimPEG",
                DeprecationWarning,
            )

        # if location_a set, then use location_a, location_b
        if location_a is not None:
            if location_b is None:
                raise ValueError(
                    "For a dipole source both location_a and location_b " "must be set"
                )

            if location is not None:
                raise ValueError(
                    "Cannot set both location and location_a, location_b. "
                    "Please provide either location=(location_a, location_b) "
                    "or both location_a=location_a, location_b=location_b"
                )

            location = [location_a, location_b]

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
    def location_a(self):
        """Location of the A-electrode"""
        return self.location[0]

    @property
    def location_b(self):
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
