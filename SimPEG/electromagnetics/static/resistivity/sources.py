import numpy as np
import properties

from .... import survey
from ....utils import Zero, closestPoints
from ....utils.code_utils import deprecate_property
from discretize import TensorMesh, TreeMesh

import warnings


class BaseSrc(survey.BaseSrc):
    """
    Base DC source
    """

    current = properties.Float("amplitude of the source current", default=1.0)

    _q = None

    def __init__(self, receiver_list, **kwargs):
        super(BaseSrc, self).__init__(receiver_list, **kwargs)

    def eval(self, sim):
        raise NotImplementedError

    def evalDeriv(self, sim):
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

    def eval(self, sim):
        if self._q is not None:
            return self._q
        else:
            if sim._formulation == "HJ":
                inds = closestPoints(sim.mesh, self.location, gridLoc="CC")
                self._q = np.zeros(sim.mesh.nC)
                self._q[inds] = self.current * np.r_[1.0, -1.0]
            elif sim._formulation == "EB":
                qa = sim.mesh.getInterpolationMat(
                    self.location[0], locType="N"
                ).toarray()
                qb = -sim.mesh.getInterpolationMat(
                    self.location[1], locType="N"
                ).toarray()
                self._q = self.current * (qa + qb)
            return self._q

    def compute_phi_primary(self, loc_grid, zf, rho0, dh):

        R1a = np.sqrt(
            (loc_grid[:, 0] - self.location[0][0])**2 +
            (loc_grid[:, 1] - self.location[0][1])**2 +
            (loc_grid[:, 2] - self.location[0][2])**2
        ) + dh/100.

        R1b = np.sqrt(
            (loc_grid[:, 0] - self.location[1][0])**2 +
            (loc_grid[:, 1] - self.location[1][1])**2 +
            (loc_grid[:, 2] - self.location[1][2])**2
        ) + dh/100.

        # Contribution from image source
        z2a = 2*zf[0] - self.location[0][2]
        R2a = np.sqrt(
            (loc_grid[:, 0] - self.location[0][0])**2 +
            (loc_grid[:, 1] - self.location[0][1])**2 +
            (loc_grid[:, 2] - z2a)**2
        ) + dh/100.

        z2b = 2*zf[1] - self.location[1][2]
        R2b = np.sqrt(
            (loc_grid[:, 0] - self.location[1][0])**2 +
            (loc_grid[:, 1] - self.location[1][1])**2 +
            (loc_grid[:, 2] - z2b)**2
        ) + dh/100.

        return (self.current*rho0/(4*np.pi)) * (R1a**-1 + R2a**-1 - R1b**-1 - R2b**-1)


class Pole(BaseSrc):
    def __init__(self, receiver_list=[], location=None, **kwargs):
        super(Pole, self).__init__(receiver_list, location=location, **kwargs)

    def eval(self, sim):
        if self._q is not None:
            return self._q
        else:
            if sim._formulation == "HJ":
                inds = closestPoints(sim.mesh, self.location)
                self._q = np.zeros(sim.mesh.nC)
                self._q[inds] = self.current * np.r_[1.0]
            elif sim._formulation == "EB":
                q = sim.mesh.getInterpolationMat(self.location, locType="N")
                self._q = self.current * q.toarray()
            return self._q

    def compute_phi_primary(self, loc_grid, zf, rho0, dh):

        # Distance from source to locations
        R1 = np.sqrt(
            (loc_grid[:, 0] - self.location[0])**2 +
            (loc_grid[:, 1] - self.location[1])**2 +
            (loc_grid[:, 2] - self.location[2])**2
        ) + dh/100.

        # Distance from image source to locations
        z2 = 2*zf - self.location[2]
        R2 = np.sqrt(
            (loc_grid[:, 0] - self.location[0])**2 +
            (loc_grid[:, 1] - self.location[1])**2 +
            (loc_grid[:, 2] - z2)**2
        ) + dh/100.

        return (self.current*rho0/(4*np.pi)) * (R1**-1 + R2**-1)


    def eval_interpolation(self, sim):

        mesh = sim.mesh        
        q = np.zeros(mesh.nN)
        xyz = self.location
        k_center = closestPoints(sim.mesh, xyz)[0]

        if isinstance(mesh, TreeMesh):
            k_nodes = mesh[k_center].nodes
            hx, hy, hz = mesh[k_center].h
            
        else:
            z_ind = k_center // (mesh.nCx * mesh.nCy)
            z_remainder = k_center % (mesh.nCx * mesh.nCy)
            y_ind = z_remainder // mesh.nCx
            x_ind = z_remainder % mesh.nCx

            k_nodes = np.ones(8, dtype=int) * z_ind*(mesh.nNx * mesh.nNy) + y_ind*mesh.nNx + x_ind
            k_nodes[4:] += mesh.nNx*mesh.nNy
            k_nodes[[2, 3, 6, 7]] += mesh.nNx
            k_nodes[1::2] += 1

            hx, hy, hz = mesh.hx[x_ind], mesh.hy[y_ind], mesh.hz[z_ind]

        v = hx*hy*hz        
        xyz_nodes = mesh.grid_nodes[k_nodes, :]     
        
        n = 2
        eps = 1e-10

        dx = np.abs(xyz[0]-xyz_nodes[:, 0]) + eps*hx
        dy = np.abs(xyz[1]-xyz_nodes[:, 1]) + eps*hy
        dz = np.abs(xyz[2]-xyz_nodes[:, 2]) + eps*hz
        q[k_nodes] = (np.sum((dx*dy*dz/v)**-n) * (dx*dy*dz/v)**n)**-1

        # r = np.sqrt(
        #     (xyz[0]-xyz_nodes[:, 0])**2 +
        #     (xyz[1]-xyz_nodes[:, 1])**2 +
        #     (xyz[2]-xyz_nodes[:, 2])**2
        # ) + eps * hx
        # q[k_nodes] = (np.sum(r**-n) * r**n)**-1

        

        return self.current * q



            



