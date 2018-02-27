import numpy as np
import SimPEG as simpeg


def homo1DModelSource(mesh, freq, sigma_1d):
    """
        Function that calculates and return background fields

        :param Simpeg mesh object mesh: Holds information on the discretization
        :param float freq: The frequency to solve at
        :param np.array sigma_1d: Background model of conductivity to base the calculations on, 1d model.
        :rtype: numpy.ndarray (mesh.nE, 2)
        :return: eBG_bp, E fields for the background model at both polarizations.

    """
    from . import get1DEfields
    # Get a 1d solution for a halfspace background
    if mesh.dim == 1:
        mesh1d = mesh
    elif mesh.dim == 2:
        mesh1d = simpeg.Mesh.TensorMesh([mesh.hy], np.array([mesh.x0[1]]))
    elif mesh.dim == 3:
        mesh1d = simpeg.Mesh.TensorMesh([mesh.hz], np.array([mesh.x0[2]]))

    # # Note: Everything is using e^iwt
    e0_1d = get1DEfields(mesh1d, sigma_1d, freq)
    if mesh.dim == 1:
        eBG_px = simpeg.mkvc(e0_1d, 2)
        eBG_py = -simpeg.mkvc(e0_1d, 2) # added a minus to make the results in the correct quadrents.
    elif mesh.dim == 2:
        ex_px = np.zeros(mesh.vnEx, dtype=complex)
        ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
        for i in np.arange(mesh.vnEx[0]):
            ex_px[i, :] = -e0_1d
        eBG_px = np.vstack((simpeg.Utils.mkvc(ex_px, 2), ey_px))
        # Setup y (north) polarization (_py)
        ex_py = np.zeros((mesh.nEx, 1), dtype='complex128')
        ey_py = np.zeros(mesh.vnEy, dtype='complex128')
        # Assign the source to ey_py
        for i in np.arange(mesh.vnEy[0]):
            ey_py[i, :] = e0_1d
        # ey_py[1:-1, 1:-1, 1:-1] = 0
        eBG_py = np.vstack((ex_py, simpeg.Utils.mkvc(ey_py, 2), ez_py))
    elif mesh.dim == 3:
        # Setup x (east) polarization (_x)
        ex_px = np.zeros(mesh.vnEx, dtype=complex)
        ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
        ez_px = np.zeros((mesh.nEz, 1), dtype=complex)
        # Assign the source to ex_x
        for i in np.arange(mesh.vnEx[0]):
            for j in np.arange(mesh.vnEx[1]):
                ex_px[i, j, :] = -e0_1d
        eBG_px = np.vstack((simpeg.Utils.mkvc(ex_px, 2), ey_px, ez_px))
        # Setup y (north) polarization (_py)
        ex_py = np.zeros((mesh.nEx, 1), dtype='complex128')
        ey_py = np.zeros(mesh.vnEy, dtype='complex128')
        ez_py = np.zeros((mesh.nEz, 1), dtype='complex128')
        # Assign the source to ey_py
        for i in np.arange(mesh.vnEy[0]):
            for j in np.arange(mesh.vnEy[1]):
                ey_py[i, j, :] = e0_1d
        # ey_py[1:-1, 1:-1, 1:-1] = 0
        eBG_py = np.vstack((ex_py, simpeg.Utils.mkvc(ey_py, 2), ez_py))

    # Return the electric fields
    eBG_bp = np.hstack((eBG_px, eBG_py))
    return eBG_bp


def analytic1DModelSource(mesh, freq, sigma_1d):
    """
        Function that calculates and return background fields

        :param Simpeg mesh object mesh: Holds information on the discretization
        :param float freq: The frequency to solve at
        :param np.array sigma_1d: Background model of conductivity to base the calculations on, 1d model.
        :rtype: numpy.ndarray (mesh.nE, 2)
        :return: eBG_bp, E fields for the background model at both polarizations.

    """
    from SimPEG.NSEM.Utils import getEHfields
    # Get a 1d solution for a halfspace background
    if mesh.dim == 1:
        mesh1d = mesh
    elif mesh.dim == 2:
        mesh1d = simpeg.Mesh.TensorMesh([mesh.hy], np.array([mesh.x0[1]]))
    elif mesh.dim == 3:
        mesh1d = simpeg.Mesh.TensorMesh([mesh.hz], np.array([mesh.x0[2]]))

    # # Note: Everything is using e^iwt
    Eu, Ed, _, _ = getEHfields(mesh1d, sigma_1d, freq, mesh.vectorNz)
    # Make the fields into a dictionary of location and the fields
    e0_1d = Eu+Ed
    E1dFieldDict = dict(zip(mesh.vectorNz, e0_1d))
    if mesh.dim == 1:
        eBG_px = simpeg.mkvc(e0_1d, 2)
        eBG_py = -simpeg.mkvc(e0_1d, 2) # added a minus to make the results in the correct quadrents.
    elif mesh.dim == 2:
        ex_px = np.zeros(mesh.vnEx, dtype=complex)
        ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
        for i in np.arange(mesh.vnEx[0]):
            ex_px[i, :] = -e0_1d
        eBG_px = np.vstack((simpeg.Utils.mkvc(ex_px, 2), ey_px))
        # Setup y (north) polarization (_py)
        ex_py = np.zeros((mesh.nEx, 1), dtype='complex128')
        ey_py = np.zeros(mesh.vnEy, dtype='complex128')
        # Assign the source to ey_py
        for i in np.arange(mesh.vnEy[0]):
            ey_py[i, :] = e0_1d
        # ey_py[1:-1, 1:-1, 1:-1] = 0
        eBG_py = np.vstack((ex_py, simpeg.Utils.mkvc(ey_py, 2), ez_py))
    elif mesh.dim == 3:
        # Setup x (east) polarization (_x)
        ex_px = -np.array([E1dFieldDict[i] for i in mesh.gridEx[:, 2]]).reshape(-1, 1)
        ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
        ez_px = np.zeros((mesh.nEz, 1), dtype=complex)
        # Construct the full fields
        eBG_px = np.vstack((ex_px, ey_px, ez_px))
        # Setup y (north) polarization (_py)
        ex_py = np.zeros((mesh.nEx, 1), dtype='complex128')
        ey_py = np.array([E1dFieldDict[i] for i in mesh.gridEy[:, 2]]).reshape(-1, 1)
        ez_py = np.zeros((mesh.nEz, 1), dtype='complex128')
        # Construct the full fields
        eBG_py = np.vstack((ex_py, simpeg.Utils.mkvc(ey_py, 2), ez_py))

    # Return the electric fields
    eBG_bp = np.hstack((eBG_px, eBG_py))
    return eBG_bp

# def homo3DModelSource(mesh, model, freq):
#     """
#         Function that estimates 1D analytic background fields from a 3D model.

#         :param Simpeg mesh object mesh: Holds information on the discretization
#         :param float freq: The frequency to solve at
#         :param np.array sigma_1d: Background model of conductivity to base the calculations on, 1d model.
#         :rtype: numpy.ndarray (mesh.nE, 2)
#         :return: eBG_bp, E fields for the background model at both polarizations.

#     """

#     if mesh.dim < 3:
#         raise IOError('Input mesh has to have 3 dimensions.')


#     # Get the locations
#     a = mesh.gridCC[:, 0:2].copy()
#     unixy = np.unique(a.view(a.dtype.descr * a.shape[1])).view(float).reshape(-1, 2)
#     uniz = np.unique(mesh.gridCC[:, 2])
#     # # Note: Everything is using e^iwt
#     # Need to loop thourgh the xy locations, assess the model and calculate the fields at the phusdo cell centers.
#     # Then interpolate the cc fields to the edges.

#     e0_1d = get1DEfields(mesh1d, sigma_1d, freq)

#     elif mesh.dim == 3:
#         # Setup x (east) polarization (_x)
#         ex_px = np.zeros(mesh.vnEx, dtype=complex)
#         ey_px = np.zeros((mesh.nEy, 1), dtype=complex)
#         ez_px = np.zeros((mesh.nEz, 1), dtype=complex)
#         # Assign the source to ex_x
#         for i in np.arange(mesh.vnEx[0]):
#             for j in np.arange(mesh.vnEx[1]):
#                 ex_px[i, j, :] = -e0_1d
#         eBG_px = np.vstack((simpeg.Utils.mkvc(ex_px, 2), ey_px, ez_px))
#         # Setup y (north) polarization (_py)
#         ex_py = np.zeros((mesh.nEx, 1), dtype='complex128')
#         ey_py = np.zeros(mesh.vnEy, dtype='complex128')
#         ez_py = np.zeros((mesh.nEz, 1), dtype='complex128')
#         # Assign the source to ey_py
#         for i in np.arange(mesh.vnEy[0]):
#             for j in np.arange(mesh.vnEy[1]):
#                 ey_py[i, j, :] = e0_1d
#         # ey_py[1:-1, 1:-1, 1:-1] = 0
#         eBG_py = np.vstack((ex_py, simpeg.Utils.mkvc(ey_py, 2), ez_py))

#     # Return the electric fields
#     eBG_bp = np.hstack((eBG_px, eBG_py))
#     return eBG_bp
