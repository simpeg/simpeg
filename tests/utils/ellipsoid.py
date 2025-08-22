from simpeg import utils
import numpy as np

"""
The code for forward modelling magnetic field of ellipsoids present in this
file is based on the implementation made by Diego Takahashi Tomazella and
Vanderlei C. Oliveira Jr., which is available in
https://github.com/pinga-lab/magnetic-ellipsoid, and has been released under
the BSD 3-Clause license:

> Copyright (c) Diego Takahashi Tomazella and Vanderlei C. Oliveira Jr.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the names of the copyright holders nor the names of any
>   contributors may be used to endorse or promote products derived from this
>   software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.
"""


class ProlateEllipsoid:
    r"""Class for magnetostatic solution for a permeable and remanently
    magnetized prolate ellipsoid in a uniform magnetostatic field
    based on: https://github.com/pinga-lab/magnetic-ellipsoid

    The ``ProlateEllipse`` class is used to analytically compute the external and internal
    secondary magnetic flux density

    Parameters
    ----------
    center : (3) array_like, optional
        center of ellipsoid (m).
    axis : (2) array_like, optional
        major and both minor axes of ellipsoid (m).
    strike_dip_rake : (3) array_like, optional
        strike, dip, and rake of ellipsoid, defined in paper (degrees)
        Sets property V (rotation matrix)
    susceptibility : float
        susceptibility of ellipsoid (SI).
    Mr : (3) array_like, optional
        Intrinsic remanent magnetic polarization (\mu_0 M) of ellipsoid.
        If susceptibility = 0,equivalent to total resultant magnetization. (nT)
    inducing_field : (3) array_like, optional
        Ambient Geomagnetic Field.  (strength(nT),inclination (degrees), declination (degrees)
    """

    def __init__(
        self,
        center=(0, 0, 0),
        axes=(100.1, 100),
        strike_dip_rake=(0, 0, 0),
        susceptibility=0.0,
        Mr=(0, 0, 0),
        inducing_field=(50000, 0, 90),
        **kwargs,
    ):
        self.center = self.__redefine_coords(center)
        self.axes = axes
        self.susceptibility = susceptibility
        self.V = strike_dip_rake
        Mr = np.array(Mr)
        self.Mr = Mr.T
        self.B_0 = inducing_field

    @property
    def center(self):
        """Center of the sphere

        Returns
        -------
        (3) numpy.ndarray of float
            Center of the sphere. Default = np.r_[0,0,0]
        """
        return self._center

    @center.setter
    def center(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except (TypeError, AttributeError, ValueError):
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"location must be array_like with shape (3,), got {len(vec)}"
            )

        self._center = vec

    @property
    def axes(self):
        """The major axis and shared minor axes of the prolate ellipsoid

        Returns
        -------
        (2) numpy.ndarray of float
            Center of the sphere. Default = np.r_[100.1,100]
        """
        return self._axes

    @axes.setter
    def axes(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except (TypeError, AttributeError, ValueError):
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 2:
            raise ValueError(
                "location must be array_like with shape (2,), got {len(vec)}"
            )

        if vec[0] <= vec[1]:
            raise ValueError(
                "The major axis of the ellipsoid must be greater then the minor axes"
            )

        if np.any(np.less(vec, 0)):
            raise ValueError("The axes must be positive")
        axes = np.zeros(3)
        axes[:2] = vec
        axes[2] = vec[1]
        self._axes = axes

    @property
    def V(self):
        """Rotation Matrix of Ellipsoid

        Returns
        -------
        (3,3) numpy.ndarray of float
            Rotation Matrix of Ellipsoid
        """
        return self._V

    @V.setter
    def V(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except (TypeError, AttributeError, ValueError):
            raise TypeError(f"strike_dip_rake must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"strike_dip_rake must be array_like with shape (3,), got {len(vec)}"
            )

        self._V = self.__rotation_matrix(np.radians(vec))

    @property
    def susceptibility(self):
        """Magnetic susceptibility (SI)

        Returns
        -------
        float
            Magnetic Susceptibility (SI)
        """
        return self._susceptibility

    @susceptibility.setter
    def susceptibility(self, item):
        item = float(item)
        if item < 0.0:
            raise ValueError("Susceptibility must be positive")
        self._susceptibility = item

    @property
    def Mr(self):
        r"""The remanent polarization (\mu0 M), (nT)

        Returns
        -------
        (3) numpy.ndarray of float
            Remanent Polarization (nT)
        """
        return self._Mr

    @Mr.setter
    def Mr(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except (TypeError, AttributeError, ValueError):
            raise TypeError(f"location must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"location must be array_like with shape (3,), got {len(vec)}"
            )
        self._Mr = self.__redefine_coords(vec)

    @property
    def B_0(self):
        """Amplitude of the inducing field (nT).

        Returns
        -------
        (3) numpy.ndarray of float
            Amplitude of the primary current density.  Default = np.r_[1,0,0]
        """
        return self._B_0

    @B_0.setter
    def B_0(self, vec):

        try:
            vec = np.atleast_1d(vec).astype(float)
        except (TypeError, AttributeError, ValueError):
            raise TypeError(f"primary_field must be array_like, got {type(vec)}")

        if len(vec) != 3:
            raise ValueError(
                f"primary_field must be array_like with shape (3,), got {len(vec)}"
            )

        mag = utils.mat_utils.dip_azimuth2cartesian(
            vec[1],
            vec[2],
        )

        B_0 = np.array([mag[:, 0] * vec[0], mag[:, 1] * vec[0], mag[:, 2] * vec[0]])[
            :, 0
        ]

        B_0 = self.__redefine_coords(B_0)

        self._B_0 = B_0

    def get_indices(self, xyz):
        """Returns Boolean of provided points internal to ellipse

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        ind: Boolean array, True if internal to ellipse

        """

        V = self.V
        a = self.axes[0]
        b = self.axes[1]
        c = self.axes[1]
        A = np.identity(3)
        A[0, 0] = a**-2
        A[1, 1] = b**-2
        A[2, 2] = c**-2
        A = V @ A @ V.T
        center = self.center

        t1 = xyz[:, 1] - center[0]
        t2 = xyz[:, 0] - center[1]
        t3 = -xyz[:, 2] - center[2]

        r_m_rc = np.array([t1, t2, t3])
        b = A @ r_m_rc

        values = np.sum(r_m_rc * b, axis=0)

        ind = values < 1

        return ind

    def Magnetization(self):
        """Returns the resultant magnetization of the ellipsoid as a function
        of susceptibility and remanent magnetization

        Parameters
        ----------

        Returns
        -------
        M: (3) numpy.ndarray of float

        """

        V = self.V

        K = self.susceptibility * np.identity(3)  # /(4*np.pi)

        N1 = self.__depolarization_prolate()

        I = np.identity(3)

        inv = np.linalg.inv(I + K @ N1)

        M = V @ inv @ V.T @ (K @ self.B_0.T + self.Mr.T)

        M = self.__redefine_coords(M.T)

        return M

    def anomalous_bfield(self, xyz):
        """Returns the internal and external secondary magnetic field B_s

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        B_s : (..., 3) np.ndarray
            Units of nT

        """
        a = self.axes[0]
        b = self.axes[1]
        axes_array = np.array([a, b, b])

        internal_indices = self.get_indices(xyz)
        xyz = self.__redefine_coords(xyz)
        xyz_m_center = xyz - self.center

        body_axis_coords = (self.V.T @ xyz_m_center.T).T

        x1 = body_axis_coords[:, 0]
        x2 = body_axis_coords[:, 1]
        x3 = body_axis_coords[:, 2]

        xyz = [x1, x2, x3]

        M = self.__redefine_coords(self.Magnetization())

        lam = self.__get_lam(x1, x2, x3)

        dlam = self.__d_lam(x1, x2, x3, lam)

        R = np.sqrt((a**2 + lam) * (b**2 + lam) * (b**2 + lam))

        h = []
        for i in range(len(axes_array)):
            h.append(-1 / ((axes_array[i] ** 2 + lam) * R))

        g = self.__g(lam)

        N2 = self.__N2(h, g, dlam, xyz)

        B_s = self.V @ N2 @ self.V.T @ M

        N1 = self.__depolarization_prolate()

        M_norotate = self.Magnetization()

        B_s = self.__redefine_coords(B_s)

        B_s[internal_indices, :] = M_norotate - N1 @ M_norotate

        return B_s

    def TMI(self, xyz):
        """Returns the internal and external exact TMI data

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        TMI : (...,) np.ndarray
            Units of nT

        """

        B_0 = self.__redefine_coords(self.B_0)

        B = self.anomalous_bfield(xyz)

        TMI = np.linalg.norm(B_0 + B, axis=1) - np.linalg.norm(self.B_0)

        return TMI

    def TMI_approx(self, xyz):
        """Returns the internal and external approximate TMI data

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Locations to evaluate at in units m.

        Returns
        -------
        TMI_approx : (...,) np.ndarray
            Units of nT

        """

        B = self.anomalous_bfield(xyz)
        B0 = self.__redefine_coords(self.B_0)

        TMI_approx = (B @ B0.T) / np.linalg.norm(B0)

        return TMI_approx

    def __redefine_coords(self, coords):
        coords_copy = np.copy(coords)
        if len(np.shape(coords)) == 1:

            temp = np.copy(coords[0])
            coords_copy[0] = coords[1]
            coords_copy[1] = temp
            coords_copy[2] *= -1
        else:
            temp = np.copy(coords[:, 0])
            coords_copy[:, 0] = coords[:, 1]
            coords_copy[:, 1] = temp
            coords_copy[:, 2] *= -1

        return coords_copy

    def __rotation_matrix(self, strike_dip_rake):
        strike = strike_dip_rake[0]
        dip = strike_dip_rake[1]
        rake = strike_dip_rake[2]

        def R1(theta):
            return np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), np.sin(theta)],
                    [0, -np.sin(theta), np.cos(theta)],
                ]
            )

        def R2(theta):
            return np.array(
                [
                    [np.cos(theta), 0, -np.sin(theta)],
                    [0, 1, 0],
                    [np.sin(theta), 0, np.cos(theta)],
                ]
            )

        def R3(theta):
            return np.array(
                [
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )

        V = R1(np.pi / 2) @ R2(strike) @ R1(np.pi / 2 - dip) @ R3(rake)

        return V

    def __depolarization_prolate(self):
        a = self.axes[0]
        b = self.axes[1]

        m = a / b

        t11 = 1 / (m**2 - 1)
        t22 = m / (m**2 - 1) ** 0.5
        t33 = np.log(m + (m**2 - 1) ** 0.5)

        n11 = t11 * (t22 * t33 - 1)
        n22 = 0.5 * (1 - n11)
        n33 = n22

        N1 = np.zeros((3, 3))
        N1[0, 0] = n11
        N1[1, 1] = n22
        N1[2, 2] = n33

        return N1

    def __N2(self, h, g, dlam, xyz):
        size = np.shape(g[0])[0]
        N2 = np.zeros((size, 3, 3))
        abc_2 = self.axes[0] * self.axes[1] * self.axes[2] / 2
        for i in range(3):
            for j in range(3):
                if i == j:
                    N2[:, i, j] = -abc_2 * (dlam[i] * h[i] * xyz[i] + g[i])
                else:
                    N2[:, i, j] = -abc_2 * (dlam[i] * h[j] * xyz[j])

        return N2

    def __get_lam(self, x1, x2, x3):
        a = self.axes[0]
        b = self.axes[1]
        p1 = a**2 + b**2 - x1**2 - x2**2 - x3**2
        p0 = a**2 * b**2 - b**2 * x1**2 - a**2 * (x2**2 + x3**2)
        lam = (-p1 + np.sqrt(p1**2 - 4 * p0)) / 2

        return lam

    def __d_lam(self, x1, x2, x3, lam):

        dlam = []
        xyz = [x1, x2, x3]

        den = (
            (x1 / (self.axes[0] ** 2 + lam)) ** 2
            + (x2 / (self.axes[1] ** 2 + lam)) ** 2
            + (x3 / (self.axes[1] ** 2 + lam)) ** 2
        )

        for i in range(3):
            num = (2 * xyz[i]) / (self.axes[i] ** 2 + lam)
            dlam.append(num / den)

        return dlam

    def __g(self, lam):
        a = self.axes[0]
        b = self.axes[1]
        a2lam = a**2 + lam
        b2lam = b**2 + lam
        a2mb2 = a**2 - b**2

        gmul = 1 / (a2mb2**1.5)
        g1t1 = np.log((a2mb2**0.5 + a2lam**0.5) / b2lam**0.5)
        g1t2 = (a2mb2 / a2lam) ** 0.5

        g2t2 = (a2mb2 * a2lam) ** 0.5 / b2lam

        g1 = 2 * gmul * (g1t1 - g1t2)
        g2 = gmul * (g2t2 - g1t1)
        g3 = g2

        g = [g1, g2, g3]

        return g
