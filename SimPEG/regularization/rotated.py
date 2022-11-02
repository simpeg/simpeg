from .base import BaseRegularization
import numpy as np
import scipy.sparse as sp


class SmoothnessFullGradient(BaseRegularization):
    def __init__(self, mesh, anisotropic_regularization=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)

        if anisotropic_regularization is None:
            anisotropic_regularization = np.ones((len(mesh), mesh.dim))
        self._anis_reg = anisotropic_regularization

    # overwrite the call, deriv, and deriv2...
    def __call__(self, m):
        r = self.D @ (self.mapping * (self._delta_m(m)))
        return 0.5 * r @ self.W @ r

    def deriv(self, m):
        mD = self.mapping.deriv(self._delta_m(m))
        r = self.D @ (self.mapping * (self._delta_m(m)))
        return mD.T * (self.D.T @ (self.W * r))

    def deriv2(self, m, v=None):
        mDv = self.mapping.deriv(self._delta_m(m), v)
        if v is None:
            return mDv.T * (self.D.T @ self.W @ self.D) * mDv

        return mDv.T * (self.D.T @ (self.W @ (self.D * mDv)))

    @property
    def D(self):
        if getattr(self, "_D", None) is None:
            mesh = self.regularization_mesh.mesh
            # Turn off cell_gradient at boundary faces
            bf = mesh.project_face_to_boundary_face.indices
            v = np.ones(mesh.n_faces)
            v[bf] = 0.0
            P = sp.diags(v)
            try:
                cell_gradient = mesh.cell_gradient
            except AttributeError:
                a = mesh.face_areas
                v = mesh.average_cell_to_face @ mesh.cell_volumes
                cell_gradient = sp.diags(a / v) @ mesh.stencil_cell_gradient
            self._D = P @ cell_gradient
        return self._D

    @property
    def W(self):
        if getattr(self, "_W", None) is None:
            mesh = self.regularization_mesh.mesh
            cell_weights = np.ones(len(mesh))
            for values in self._weights.values():
                cell_weights *= values
            reg_model = self._anis_reg * cell_weights[:, None]

            self._W = mesh.get_face_inner_product(reg_model)
        return self._W
