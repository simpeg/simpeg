from __future__ import print_function
import numpy as np
from discretize.utils import mkvc


########################################################################################
#                  DIRECT CURRENT RESISTIVITY AND INDUCED POLARIZATION
########################################################################################


def read_dcip3d_ubc(file_name, data_type):
    """
    Read UBC DCIP3D formatted survey, predicted or observed data files.

    This function loads 3D DC or IP data formatted for the UBC DCIP3D
    coding package and outputs a SimPEG data object. The function will
    recognize if the file is a survey, predicted data or observed data file.

    Parameters
    ----------
    file_name : str
        The file path to the data file
    data_type: str {'volt', 'apparent_chargeability', 'secondary_potential'}
        Defining the input data type:

        - 'volt': DC resistivity data as voltages
        - 'apparent_chargeability': IP data as apparent chargeabilities
        - 'secondary_potential': IP data as secondary potentials

    Returns
    -------
    data
        A SimPEG.data.Data object containing:

        - The survey
        - Observed/predicted data (if present in the data file)
        - Uncertainties (if present in the data file). Note that predicted DC data
        files contain the apparent resistivities, which are loaded into SimPEG and
        defined as uncertainties.

    """
    assert data_type.lower() in [
        "volt",
        "apparent_chargeability",
        "secondary_potential",
    ], "Parameter 'data_type' must be one of {'volt', 'apparent_chargeability', 'secondary_potential'}"

    return _read_dcip_3d_or_octree_ubc(file_name, data_type, "dcip3d")


def read_dcipoctree_ubc(file_name, data_type):
    """
    Read UBC DCIPoctree formatted survey, predicted or observed data files.

    This function loads 3D DC or IP data formatted for the UBC DCIPoctree
    coding package and outputs a SimPEG data object. The function requires
    the user to define whether the data are DC resistivity or IP.

    Parameters
    ----------
    file_name : str
        The file path to the data file
    data_type: str {'volt', 'apparent_chargeability'}
        Defining the input data type:

        - 'volt': DC resistivity data as voltages
        - 'apparent_chargeability': IP data as apparent chargeabilities

    Returns
    -------
    data_object
        A SimPEG.data.Data object containing:

        - The survey
        - Observed/predicted data (if present in the data file)
        - Uncertainties (if present in the data file). Note that predicted DC data
        files contain the apparent resistivities, which are loaded into SimPEG and
        defined as uncertainties.

    """

    # Unused for now but it will be when we manage IP types better.
    assert data_type.lower() in [
        "volt",
        "apparent_chargeability",
    ], "Parameter 'data_type' must be one of {'volt', 'apparent_chargeability'}"

    return _read_dcip_3d_or_octree_ubc(file_name, data_type, "dcipoctree")


def _read_dcip_3d_or_octree_ubc(file_name, data_type, code_type):
    """
    Read 3D DC/IP survey, predicted and observation files in UBC-GIF format.

    Parameters
    ----------

    file_name : str
        The file path to the data file
    code_type : str {'dcip3d', 'dcipoctree'}
        Code type. Choose from {'dcip3d', 'dcipoctree'}

    Returns
    -------
    data_object
        A SimPEG.data.Data object containing:

        - The survey
        - Observed/predicted data (if present in the data file)
        - Uncertainties (if present in the data file). Note that predicted DC data
        files contain the apparent resistivities, which are loaded into SimPEG and
        defined as uncertainties.

    """
    assert data_type.lower() in [
        "volt",
        "apparent_chargeability",
        "secondary_potential",
    ], "Parameter 'data_type' must be one of {'volt', 'apparent_chargeability', 'secondary_potential'}"

    assert code_type.lower() in [
        "dcip3d",
        "dcipoctree",
    ], "Parameter 'code_type' must be one of {'dcip3d', 'dcipoctree'}"

    # Prevent circular import
    from ...electromagnetics.static import resistivity as dc
    from ...data import Data

    # Load file
    obsfile = np.genfromtxt(file_name, delimiter=" \n", dtype=np.str, comments="!")

    # Pre-allocate
    source_list = []
    receiver_list = []
    d = []
    wd = []

    # Flag for z value provided
    is_surface = False
    is_pole_tx = False
    is_pole_rx = False

    # IP data for dcip3d has a line with a flag we can remove.
    if (code_type == "dcip3d") & (data_type != "volt"):
        obsfile = obsfile[1:]

    # Since SimPEG defines secondary potential from IP as voltage,
    # we must use this type when defining the receivers.
    if data_type.lower() == "secondary_potential":
        data_type = "volt"

    # Countdown for number of obs/tx
    count = 0
    for ii in range(obsfile.shape[0]):

        if not obsfile[ii]:
            continue

        # Extract transmitter ii and the number of receivers
        if count == 0:
            rx = []
            temp = np.fromstring(obsfile[ii], dtype=float, sep=" ").T
            count = int(temp[-1])

            # Check if z value is provided, if False -> nan
            if len(temp) == 5:
                # check if pole|dipole
                if np.allclose(temp[0:2], temp[2:4]):
                    tx = np.r_[temp[0:2], np.nan]
                    is_pole_tx = True
                else:
                    tx = np.r_[temp[0:2], np.nan, temp[2:4], np.nan]
                is_surface = True

            else:
                # check if pole|dipole
                if np.allclose(temp[0:3], temp[3:6]):
                    tx = np.r_[temp[0:3]]
                    is_pole_tx = True
                else:
                    tx = temp[:-1]

            continue

        # Extract receivers
        temp = np.fromstring(obsfile[ii], dtype=float, sep=" ")

        if is_surface:
            data_column_index = 4  # Since dpred for dc has app_res

            # Check if Pole Receiver
            if np.allclose(temp[0:2], temp[2:4]):
                is_pole_rx = True
                rx.append(temp[:2])
            else:
                rx.append(np.r_[temp[0:2], np.nan, temp[2:4], np.nan])

        else:
            data_column_index = 6  # Since dpred for dc has app_res

            # Check if Pole Receiver
            if np.allclose(temp[0:3], temp[3:6]):
                is_pole_rx = True
                rx.append(temp[:3])
            else:
                rx.append(temp[:6])

        # Predicted/observed data
        if len(temp) == data_column_index + 1:
            d.append(temp[data_column_index])

        # Observed data or predicted DC data (since app res column)
        elif len(temp) == data_column_index + 2:
            d.append(temp[data_column_index])
            wd.append(temp[data_column_index + 1])

        count = count - 1

        # Reach the end of transmitter block
        if count == 0:
            rx = np.asarray(rx)
            if is_pole_rx:
                Rx = dc.receivers.Pole(rx[:, :3], data_type=data_type)
            else:
                Rx = dc.receivers.Dipole(rx[:, :3], rx[:, 3:], data_type=data_type)
            if is_pole_tx:
                source_list.append(dc.sources.Pole([Rx], tx[:3]))
            else:
                source_list.append(dc.sources.Dipole([Rx], tx[:3], tx[3:]))

    # Define survey type
    if is_pole_tx:
        str1 = "pole-"
    else:
        str1 = "dipole-"

    if is_pole_rx:
        str2 = "pole"
    else:
        str2 = "dipole"

    electrode_configuration = str1 + str2
    survey = dc.survey.Survey(source_list, survey_type=electrode_configuration)
    data_out = Data(survey=survey)

    if len(d) > 0:
        data_out.dobs = d

    if len(wd) > 0:
        data_out.standard_deviation = wd

    return data_out


def write_dcip3d_ubc(
    file_name,
    data_object,
    data_type,
    file_type,
    format_type="general",
    electrode_configuration=None,
    comment_lines="",
):
    """
    Write UBC DCIP3D formatted survey, predicted or observation files.

    Parameters
    ----------
    file_name:
    data_object:
    file_type: 'survey', 'dpred', 'dobs'
    format_type: 'general', 'surface', 'simple'
    data_type: 'volt', 'apparent_chargeability', 'secondary_potential'
    electrode_configuration: 'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'
    comment_lines:)
    """

    assert data_type.lower() in [
        "volt",
        "apparent_chargeability",
        "secondary_potential",
    ], "Parameter 'data_type' must be one of {'volt', 'apparent_chargeability', 'secondary_potential'}"

    assert file_type.lower() in [
        "survey",
        "dpred",
        "dobs",
    ], "Parameter 'file_type' must be one of {'survey', 'dpred', 'dobs'}"

    _write_dcip_3d_or_octree_ubc(
        file_name,
        data_object,
        data_type,
        file_type,
        format_type=format_type,
        electrode_configuration=electrode_configuration,
        code_type="dcip3d",
        comment_lines=comment_lines,
    )


def write_dcipoctree_ubc(
    file_name,
    data_object,
    data_type,
    file_type,
    format_type="general",
    electrode_configuration=None,
    comment_lines="",
):
    """
    Write UBC DCIPoctree formatted survey, predicted or observation files.

    Parameters
    ----------
    file_name:
    data_object:
    file_type: 'survey', 'dpred', 'dobs'
    format_type: 'general', 'surface', 'simple'
    data_type: 0 (DC), 1 (IP), 2 (another IP)
    electrode_configuration: 'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'
    comment_lines:)
    """

    assert data_type.lower() in [
        "volt",
        "apparent_chargeability",
    ], "Parameter 'data_type' must be one of {'volt', 'apparent_chargeability'}"

    assert file_type.lower() in [
        "survey",
        "dpred",
        "dobs",
    ], "Parameter 'file_type' must be one of {'survey', 'dpred', 'dobs'}"

    _write_dcip_3d_or_octree_ubc(
        file_name,
        data_object,
        data_type,
        file_type,
        format_type="general",
        electrode_configuration=electrode_configuration,
        code_type="dcipoctree",
        comment_lines=comment_lines,
    )


def _write_dcip_3d_or_octree_ubc(
    file_name,
    data_object,
    data_type,
    file_type,
    format_type="general",
    electrode_configuration=None,
    code_type="dcip3d",
    comment_lines="",
):
    """
    Write UBC DCIP3D formatted survey, predicted or observation files.

    Parameters
    ----------
    file_name:
    data_object:
    file_type: 'survey', 'dpred', 'dobs'
    format_type: 'general', 'surface', 'simple'
    data_type: {'volt', 'apparent_chargeability', 'secondary_potential'}
    electrode_configuration: 'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'
    code_type: 'dcip3d', 'dcipoctree'
    comment_lines:
    """

    # Prevent circular import
    from ...electromagnetics.static import resistivity as dc
    from ...electromagnetics.static.utils.static_utils import apparent_resistivity
    from ...data import Data

    # Validate inputs
    if not isinstance(data_object, Data):
        raise Exception(
            "A Data instance ({datacls}: <{datapref}.{datacls}>) must be "
            "provided as the second input. The provided input is a "
            "{providedcls} <{providedpref}.{providedcls}>".format(
                datacls=Data.__name__,
                datapref=Data.__module__,
                providedcls=data_object.__class__.__name__,
                providedpref=data_object.__module__,
            )
        )

    assert data_type.lower() in [
        "volt",
        "apparent_chargeability",
        "secondary_potential",
    ], "Parameter 'data_type' must be one of {'volt', 'apparent_chargeability', 'secondary_potential'}"

    assert file_type.lower() in [
        "survey",
        "dpred",
        "dobs",
    ], "Parameter 'file_type' must be one of {'survey', 'dpred', 'dobs'}"

    assert code_type.lower() in [
        "dcip3d",
        "dcipoctree",
    ], "Parameter 'code_type' must be one of {'dcip3d', 'dcipoctree'}"

    format_type = format_type.lower()
    if format_type not in ["surface", "general", "simple"]:
        raise Exception(
            "format_type must be 'surface' | 'general' | 'simple' "
            " not {}".format(format_type)
        )

    if electrode_configuration is None:
        electrode_configuration = data_object.survey.survey_type

    # Predicted DC data will automatically contain apparent resistivity column.
    # Here we compute the apparent resistivities and treat it like an uncertainties column.
    if (file_type.lower() == "dpred") & (data_type == "volt"):
        data_object.standard_deviation = apparent_resistivity(data_object)
        file_type = "dobs"

    # Write comments and IP type (if applicable)
    fid = open(file_name, "w")
    fid.write("! " + format_type + " FORMAT\n")

    if comment_lines:
        fid.write(comment_lines)

    # DCIP3D will allow user to choose definition of IP data. DC data has no flag.
    # DCIPoctree IP data is always apparent chargeability.
    if (code_type.lower() == "dcip3d") & (data_type == "apparent_chargeability"):
        fid.write("IPTYPE=%i\n" % 1)
    elif (code_type.lower() == "dcip3d") & (data_type == "secondary_potential"):
        fid.write("IPTYPE=%i\n" % 2)

    fid.close()

    # Index deciding if z locations are written
    if format_type.lower() == "surface":
        end_index = 2
    elif format_type.lower() == "general":
        end_index = 3

    # Loop over all sources
    count = 0
    for src in data_object.survey.source_list:

        # Write Source
        nD = src.nD

        if electrode_configuration.lower() in ["pole-dipole", "pole-pole"]:
            tx = np.r_[src.location]
            tx = np.repeat(np.r_[[tx]], 2, axis=0)
        elif electrode_configuration.lower() in ["dipole-dipole", "dipole-pole"]:
            tx = np.c_[src.location]

        fid = open(file_name, "a")
        fid.writelines("%e " % ii for ii in mkvc(tx[:, 0:end_index].T))
        fid.write("%i\n" % nD)
        fid.close()

        # Write receivers
        for rx in src.receiver_list:

            if electrode_configuration.lower() in ["pole-dipole", "dipole-dipole"]:
                M = rx.locations[0][0:end_index]
                N = rx.locations[1][0:end_index]
            elif electrode_configuration.lower() in ["pole-pole", "dipole-pole"]:
                M = rx.locations[0:end_index]
                N = rx.locations[0:end_index]

            if file_type.lower() != "survey":
                N = np.c_[N, data_object.dobs[count : count + rx.nD]]

            if file_type.lower() == "dobs":
                N = np.c_[N, data_object.standard_deviation[count : count + rx.nD]]

            # Write receivers and locations
            fid = open(file_name, "ab")
            if isinstance(N, np.ndarray):
                np.savetxt(
                    fid,
                    np.c_[M, N],
                    fmt=str("%e"),
                    delimiter=str(" "),
                    newline=str("\n"),
                )
            else:
                raise Exception(
                    """Uncertainities SurveyObject.std should be set.
                    Either float or nunmpy.ndarray is expected, """
                    "not {}".format(type(data_object.relative_error))
                )

            fid.close()
            fid = open(file_name, "a")
            fid.write("\n")
            fid.close()

            count += rx.nD

    fid.close()
