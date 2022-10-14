from __future__ import print_function
import numpy as np
from discretize.utils import mkvc
from ...utils.code_utils import deprecate_method


def read_mag3d_ubc(obs_file):
    """Read UBC-GIF MAG3D formatted survey or data files.

    This method can load survey locations, predicted data or observations
    files formatted for the UBC-GIF MAG3D coding package. For more, see
    the `UBC-GIF MAG3D v6.0 online manual <https://mag3d.readthedocs.io/en/v6/>`__.

    Parameters
    ----------
    obs_file : str
        Path to a UBC-GIF MAG3D formatted data file

    Returns
    -------
    SimPEG.data.Data
        Instance of a SimPEG data class. The `survey` attribute associated with
        the data object is an instance of :class`SimPEG.potential_fields.magnetics.survey.Survey`.
    """

    # Prevent circular import
    from ...potential_fields import magnetics
    from ...data import Data

    fid = open(obs_file, "r")

    # First line has the inclination,declination and amplitude of B0
    line = fid.readline()
    B = np.array(line.split()[:3], dtype=float)

    # Second line has the magnetization orientation and a flag
    line = fid.readline()
    M = np.array(line.split()[:3], dtype=float)

    # Third line has the number of rows
    line = fid.readline()
    ndat = int(line.split()[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()
    temp = np.array(line.split(), dtype=float)

    d = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    ii = 0
    while ii < ndat:

        temp = np.array(line.split(), dtype=float)
        if len(temp) > 0:
            locXYZ[ii, :] = temp[:3]

            if len(temp) > 3:
                d[ii] = temp[3]

                if len(temp) == 5:
                    wd[ii] = temp[4]
            ii += 1
        line = fid.readline()
    fid.close()

    if np.all(wd == 0.0):
        wd = None

    if np.all(d == 0.0):
        d = None

    rxLoc = magnetics.receivers.Point(locXYZ)
    srcField = magnetics.sources.SourceField([rxLoc], parameters=(B[2], B[0], B[1]))
    survey = magnetics.survey.Survey(srcField)
    data_object = Data(survey, dobs=d, standard_deviation=wd)

    return data_object


def write_mag3d_ubc(filename, data_object):
    """Write UBC-GIF MAG3D formatted survey or data files.

    This function can write survey locations, predicted data or observations
    files formatted for the UBC-GIF MAG3D coding package. For more, see
    the `UBC-GIF MAG3D v6.0 online manual <https://mag3d.readthedocs.io/en/v6/>`__.

    Parameters
    ----------
    filename : str
        File path for the output file
    data_object : SimPEG.data.Data
        An instance of SimPEG data class. The `survey` attribute associate with the
        data object must be an instance of :class:`SimPEG.potential_fields.magnetics.survey.Survey`
    """
    survey = data_object.survey

    B = survey.source_field.parameters

    data = survey.source_field.receiver_list[0].locations

    if data_object.dobs is not None:
        data = np.c_[data, data_object.dobs]

    if data_object.standard_deviation is not None:
        data = np.c_[data, data_object.standard_deviation]

    head = (
        "%6.2f %6.2f %6.2f\n" % (B[1], B[2], B[0])
        + "%6.2f %6.2f %6.2f\n" % (B[1], B[2], 1)
        + "%i\n" % survey.nD
    )
    np.savetxt(
        filename, data, fmt="%e", delimiter=" ", newline="\n", header=head, comments=""
    )

    print("Observation file saved to: " + filename)


def read_grav3d_ubc(obs_file):
    """Read UBC-GIF GRAV3D formatted survey or data files.

    This method can load survey locations, predicted data or observations
    files formatted for the UBC-GIF GRAV3D coding package. For more, see
    the `UBC-GIF GRAV3D v6.0 online manual <https://grav3d.readthedocs.io/en/v6.0/>`__.

    Parameters
    ----------
    obs_file : str
        Path to a UBC-GIF GRAV3D formatted file

    Returns
    -------
    SimPEG.data.Data
        Instance of a SimPEG data class. The `survey` attribute associated with
        the data object is an instance of :class`SimPEG.potential_fields.gravity.survey.Survey`.
    """

    # Prevent circular import
    from ...potential_fields import gravity
    from ...data import Data

    fid = open(obs_file, "r")

    # First line has the number of rows
    line = fid.readline()
    ndat = int(line.split()[0])

    # Pre-allocate space for obsx, obsy, obsz, data, uncert
    line = fid.readline()

    d = np.zeros(ndat, dtype=float)
    wd = np.zeros(ndat, dtype=float)
    locXYZ = np.zeros((ndat, 3), dtype=float)

    ii = 0
    while ii < ndat:

        temp = np.array(line.split(), dtype=float)
        if len(temp) > 0:
            locXYZ[ii, :] = temp[:3]
            if len(temp) > 3:
                d[ii] = temp[3]

                if len(temp) == 5:
                    wd[ii] = temp[4]

            ii += 1
        line = fid.readline()
    fid.close()

    if np.all(wd == 0.0):
        wd = None

    # UBC and SimPEG used opposite sign convention for
    # gravity data so must multiply by -1.
    if np.all(d == 0.0):
        d = None
    else:
        d *= -1.0

    rxLoc = gravity.receivers.Point(locXYZ)
    srcField = gravity.sources.SourceField([rxLoc])
    survey = gravity.survey.Survey(srcField)
    data_object = Data(survey, dobs=d, standard_deviation=wd)
    return data_object


def write_grav3d_ubc(filename, data_object):
    """Write UBC-GIF GRAV3D formatted survey or data files.

    This function can write survey locations, predicted data or observations
    files formatted for the UBC-GIF GRAV3D coding package. For more, see
    the `UBC-GIF GRAV3D v6.0 online manual <https://grav3d.readthedocs.io/en/v6.0/>`__.

    Parameters
    ----------
    filename : str
        File path for the output file
    data_object : SimPEG.data.Data
        An instance of SimPEG data class. The `survey` attribute associate with the
        data object must be an instance of :class:`SimPEG.potential_fields.gravity.survey.Survey`
    """
    survey = data_object.survey

    data = survey.source_field.receiver_list[0].locations

    # UBC and SimPEG use opposite sign for gravity data so
    # data are multiplied by -1.
    if data_object.dobs is not None:
        data = np.c_[data, -data_object.dobs]

    if data_object.standard_deviation is not None:
        data = np.c_[data, data_object.standard_deviation]

    head = "%i\n" % survey.nD
    np.savetxt(
        filename, data, fmt="%e", delimiter=" ", newline="\n", header=head, comments=""
    )

    print("Observation file saved to: " + filename)


def read_gg3d_ubc(obs_file):
    """Read UBC-GIF GG3D formatted survey or data files.

    This method can load survey locations, predicted data or observations
    files formatted for the UBC-GIF GG3D coding package. For more, see
    the `UBC-GIF GRAV3D v6.0 online manual <https://gg3d.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    obs_file : str
        Path to a UBC-GIF GG3D formatted file

    Returns
    -------
    SimPEG.data.Data
        Instance of a SimPEG data class. The `survey` attribute associated with
        the data object is an instance of :class`SimPEG.potential_fields.gravity.survey.Survey`.
    """

    # Prevent circular import
    from ...potential_fields import gravity
    from ...data import Data

    with open(obs_file, "r") as fid:

        # First line has components. Extract components
        line = fid.readline()
        line = line.split("=")[1].split("!")[0].split("\n")[0]
        line = line.replace(",", " ").split(" ")  # UBC uses ',' or ' ' as deliminator
        components = [s for s in line if len(s) > 0]  # Remove empty string
        n_comp = len(components)
        factor = np.zeros(n_comp)

        # Convert component types from UBC to SimPEG
        ubc_types = ["xx", "xy", "xz", "yy", "yz", "zz", "uv"]
        simpeg_types = ["gyy", "gxy", "gyz", "gxx", "gxz", "gzz", "guv"]
        factor_list = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]

        for ii in range(n_comp):
            k = ubc_types.index(components[ii])
            factor[ii] = factor_list[k]
            components[ii] = simpeg_types[k]

        # Second Line has number of locations
        line = fid.readline()
        ndat = int(line.split()[0])

        locXYZ = np.zeros((ndat, 3), dtype=float)
        d = []
        wd = []

        ii = 0
        while ii < ndat:
            try:
                line = fid.readline()
                temp = np.array(line.split(), dtype=float)
                locXYZ[ii, :] = temp[:3]
                if len(temp) == 3 + n_comp:
                    d.append(factor * temp[3:])
                elif len(temp) == 3 + n_comp * 2:
                    d.append(factor * temp[3 : 3 + n_comp])
                    wd.append(temp[3 + n_comp :])
                ii += 1
            except:
                raise IOError(f"Unable to read data line {ii}: {line}")

    # Turn into vector. For multiple components, SimPEG orders by rows
    if len(d) > 0:
        d = mkvc(np.stack(d).T)
    else:
        d = None
    if len(wd) > 0:
        wd = mkvc(np.stack(wd).T)
    else:
        wd = None

    rxLoc = gravity.receivers.Point(locXYZ, components=components)
    srcField = gravity.sources.SourceField([rxLoc])
    survey = gravity.survey.Survey(srcField)
    data_object = Data(survey, dobs=d, standard_deviation=wd)
    return data_object


def write_gg3d_ubc(filename, data_object):
    """Write UBC-GIF GG3D formatted survey or data files.

    This function can write survey locations, predicted data or observations
    files formatted for the UBC-GIF GG3D coding package. For more, see
    the `UBC-GIF GG3D v6.0 online manual <https://gg3d.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    filename : str
        File path for the output file
    data_object : SimPEG.data.Data
        An instance of SimPEG data class. The `survey` attribute associate with the
        data object must be an instance of :class:`SimPEG.potential_fields.gravity.survey.Survey`
    """
    survey = data_object.survey

    # Convert component types from UBC to SimPEG
    components = list(survey.components.keys())
    n_comp = len(components)
    factor = np.ones(n_comp)

    ubc_types = ["xx", "xy", "xz", "yy", "yz", "zz", "uv"]
    simpeg_types = ["gyy", "gxy", "gyz", "gxx", "gxz", "gzz", "guv"]
    factor_list = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
    for ii in range(0, len(components)):
        k = simpeg_types.index(components[ii])
        factor[ii] = factor_list[k]
        components[ii] = ubc_types[k]

    components = ",".join(components)

    output = survey.source_field.receiver_list[0].locations
    n_loc = np.shape(output)[0]

    if np.any(data_object.dobs != 0):
        dobs = data_object.dobs.reshape((n_loc, n_comp)) * factor
        output = np.c_[output, dobs]

    if np.any(data_object.standard_deviation != 0):
        std = data_object.standard_deviation.reshape((n_loc, n_comp))
        output = np.c_[output, std]

    head = ("datacomp=%s\n" % components) + ("%i" % n_loc)

    np.savetxt(
        filename,
        output,
        fmt="%e",
        delimiter=" ",
        newline="\n",
        header=head,
        comments="",
    )

    print("Observation file saved to: " + filename)


# ======================================================
# 				Depricated Methods
# ======================================================


readUBCmagneticsObservations = deprecate_method(
    read_mag3d_ubc, "readUBCmagneticsObservations", removal_version="0.14.4"
)
writeUBCmagneticsObservations = deprecate_method(
    write_mag3d_ubc, "writeUBCmagneticsObservations", removal_version="0.14.4"
)
readUBCgravityObservations = deprecate_method(
    read_grav3d_ubc, "readUBCgravityObservations", removal_version="0.14.4"
)
writeUBCgravityObservations = deprecate_method(
    write_grav3d_ubc, "writeUBCgravityObservations", removal_version="0.14.4"
)
