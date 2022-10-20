from __future__ import print_function
import numpy as np
from discretize.utils import mkvc
import warnings
from ..code_utils import validate_string, validate_type


########################################################################################
#                  DIRECT CURRENT RESISTIVITY AND INDUCED POLARIZATION
########################################################################################


def read_dcip_xyz(
    file_name,
    data_type,
    a_headers=["XA", "YA", "ZA"],
    b_headers=["XB", "YB", "ZB"],
    m_headers=["XM", "YM", "ZM"],
    n_headers=["XN", "YN", "ZN"],
    data_header=None,
    uncertainties_header=None,
    dict_headers=None,
    is_surface_data=False,
):
    """Read 2D or 3D DC/IP data from XYZ-formatted file.

    This function loads DC and/or IP data from XYZ-formatted data files into the
    SimPEG framework; i.e. each row defines the data for the unique electrode
    locations provided. This function is versatile enough to load 2D or 3D data.
    The data file may include elevations for the electrodes or be surface formatted.
    Columns containing data which are not defined as part of a
    :class:`SimPEG.data.Data` object may be loaded and output to a dictionary.

    Parameters
    ----------
    file_name : str
        Path to the data file
    data_type : {"volt", "apparent_resistivity", "apparent_chargeability"}
        Type of data being loaded.
    a_headers : list or tuple of str
        A list or tuple of strings providing the headers of the A-electrode location columns;
        i.e. the X (, Y and Z) columns.
    b_headers : list or tuple of str
        A list or tuple of strings providing the headers of the B-electrode location columns;
        i.e. the X (, Y and Z) columns.
    m_headers : list or tuple of str
        A list or tuple of strings providing the headers of the M-electrode location columns;
        i.e. the X (, Y and Z) columns.
    n_headers : list or tuple of str
        A list or tuple of strings providing the headers of the N-electrode location columns;
        i.e. the X (, Y and Z) columns.
    data_header : str
        The header for the data column
    uncertainties_header : str
        The header for the column containing the data uncertainties
    dict_headers : list or tuple of str
        Provide the hearders for additional data columns that you would like to load and output
        to a dictionary.
    is_surface_data : bool
        If ``True``, we assume electrode elevations are not supplied. That is, the header lists
        for ``a``, ``b``, ``m`` and ``n`` electrode locations do not have headers for
        elevation columns.

    Returns
    -------
    SimPEG.data.Data
        DC or IP data. The survey attribute associated with the data object will be an
        instance of :class:`SimPEG.electromagnetics.static.resistivity.survey.Survey`
        or :class:`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
    dict
        If additional columns are loaded and output to a dictionary using the keyward argument
        `dict_headers`, the output of this function has the form `(out_data, out_dict)`.
    """
    data_type = validate_string(
        "data_type",
        data_type,
        [
            "volt",
            "apparent_resistivity",
            "apparent_chargeability",
        ],
    )

    # Prevent circular import
    from ...electromagnetics.static.utils import generate_survey_from_abmn_locations
    from ...data import Data

    # Load file headers
    FID = open(file_name, "r")
    file_headers = FID.readline()
    FID.close()
    file_headers = file_headers.split()

    # Find indices of columns being loaded
    out_headers = a_headers + b_headers + m_headers + n_headers

    has_data = False
    has_uncert = False
    has_dict = False
    non_location_columns = 0
    if data_header is not None:
        out_headers.append(data_header)
        has_data = True
        non_location_columns += 1
    if uncertainties_header is not None:
        out_headers.append(uncertainties_header)
        has_uncert = True
        non_location_columns += 1
    if dict_headers is not None:
        out_headers += dict_headers
        has_dict = True
        non_location_columns += len(dict_headers)

    col_indices = []
    for h in out_headers:
        col_indices.append(file_headers.index(h))

    # Load specified columns of data
    data_array = np.loadtxt(file_name, comments="!", skiprows=1, usecols=col_indices)
    n_rows = np.shape(data_array)[0]
    num_location_columns = np.shape(data_array)[1] - non_location_columns

    # Extract columns for electrode locations
    if num_location_columns == 12:
        a_cols = [0, 1, 2]
        b_cols = [3, 4, 5]
        m_cols = [6, 7, 8]
        n_cols = [9, 10, 11]
    # 2D survey data or surface 3D data
    elif num_location_columns == 8:
        a_cols = [0, 1]
        b_cols = [2, 3]
        m_cols = [4, 5]
        n_cols = [6, 7]
    # 2D surface data
    else:
        a_cols = [1]
        b_cols = [2]
        m_cols = [3]
        n_cols = [4]

    # Extract electrode locations
    if is_surface_data:
        dummy_elevation = 9999  # Taller than mount Everest
        locations_a = np.c_[data_array[:, a_cols], dummy_elevation * np.ones(n_rows)]
        locations_b = np.c_[data_array[:, b_cols], dummy_elevation * np.ones(n_rows)]
        locations_m = np.c_[data_array[:, m_cols], dummy_elevation * np.ones(n_rows)]
        locations_n = np.c_[data_array[:, n_cols], dummy_elevation * np.ones(n_rows)]
        warnings.warn(
            "Loaded data are in surface format. Elevations automatically set to 9999 m. "
            "Use the project_to_discretized_topography method of the survey to project "
            "electrode locations to the discretized surface."
        )
    else:
        locations_a = data_array[:, a_cols]
        locations_b = data_array[:, b_cols]
        locations_m = data_array[:, m_cols]
        locations_n = data_array[:, n_cols]

    survey, out_indices = generate_survey_from_abmn_locations(
        locations_a=locations_a,
        locations_b=locations_b,
        locations_m=locations_m,
        locations_n=locations_n,
        data_type=data_type,
        output_sorting=True,
    )

    data_object = Data(survey)

    # Sort and organize all data columns
    if has_data:
        data_object.dobs = data_array[out_indices, file_headers.index(data_header)]

    # Sort and organize all data columns
    if has_uncert:
        data_object.standard_deviation = data_array[
            out_indices, file_headers.index(uncertainties_header)
        ]

    # Sort and organize all data columns
    if has_dict:
        out_dict = {}
        for h in dict_headers:
            out_dict[h] = data_array[out_indices, file_headers.index(h)]
        return data_object, out_dict

    else:
        return data_object


def read_dcip2d_ubc(file_name, data_type, format_type):
    """Read UBC-GIF DCIP2D formatted survey or data files.

    This method can load survey locations, predicted data or observations
    files formatted for the UBC-GIF DCIP2D coding package. For more, see
    the `UBC-GIF DCIP2D online manual <https://dcip2d.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    file_name : str
        The file path to the data file
    data_type : {'volt', 'apparent_chargeability', 'secondary_potential'}
        The type of data.
    format_type : {'general', 'surface', 'simple'}
        The format of the formated file.

    Returns
    -------
    SimPEG.data.Data
        A SimPEG data object. The data from the input file is loaded and parsed into
        three attributes of the data object:

        - `survey`: the survey geometry as defined by an instance of :class`SimPEG.electromagnetics.static.resistivity.survey.Survey` or :class`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
        - `dobs`: observed/predicted data if present in the data file
        - `standard_deviations`: uncertainties (if observed data file) or apparent resistivities (if predicted data file)

    """

    data_type = validate_string(
        "data_type",
        data_type,
        [
            "volt",
            "apparent_chargeability",
            "secondary_potential",
        ],
    )

    format_type = validate_string(
        "format_type",
        format_type,
        [
            "general",
            "surface",
            "simple",
        ],
    )

    # Prevent circular import
    from ...electromagnetics.static import resistivity as dc
    from ...electromagnetics.static.utils import generate_survey_from_abmn_locations
    from ...data import Data

    # Load file
    obsfile = np.genfromtxt(file_name, delimiter="\n", dtype=str, comments="!")

    # Find starting data
    start_index = 0
    if obsfile[0] == "COMMON_CURRENT":
        start_index = 2
    if data_type != "volt":
        start_index = start_index + 1

    obsfile = obsfile[start_index:]

    # Since SimPEG defines secondary potential from IP as voltage,
    # we must use this type when defining the receivers.
    if data_type == "secondary_potential":
        data_type = "volt"

    # Pre-allocate
    source_list = []
    d = []
    wd = []

    # Flag for z value provided
    is_surface = False
    is_pole_tx = False
    is_pole_rx = False

    if format_type == "simple":

        # Load numeric data into an array
        if data_type == "volt":
            data_array = np.loadtxt(file_name, comments="!", skiprows=start_index)
        else:
            data_array = np.loadtxt(
                file_name, comments={"!", "IPTYPE"}, skiprows=start_index
            )

        n_rows = np.shape(data_array)[0]

        if np.shape(data_array)[1] > 5:
            wd = data_array[:, 5]

        if np.shape(data_array)[1] > 4:
            d = data_array[:, 4]

        # Get ABMN electrode locations
        dummy_elevation = 9999

        locations_a = np.c_[data_array[:, 0], dummy_elevation * np.ones(n_rows)]
        locations_b = np.c_[data_array[:, 1], dummy_elevation * np.ones(n_rows)]
        locations_m = np.c_[data_array[:, 2], dummy_elevation * np.ones(n_rows)]
        locations_n = np.c_[data_array[:, 3], dummy_elevation * np.ones(n_rows)]

        survey, out_indices = generate_survey_from_abmn_locations(
            locations_a=locations_a,
            locations_b=locations_b,
            locations_m=locations_m,
            locations_n=locations_n,
            data_type=data_type,
            output_sorting=True,
        )

        data_out = Data(survey)

        # Sort and organize all data columns
        if len(d) > 0:
            data_out.dobs = d[out_indices]

        # Sort and organize all data columns
        if len(wd) > 0:
            data_out.standard_deviation = wd[out_indices]

        warnings.warn(
            "Loaded data did not have elevations. Elevations automatically set to 9999 m. "
            "Use the project_to_discretized_topography method of the survey to project "
            "electrode locations to the discretized surface."
        )

    else:

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

                # Check if z value is provided, if False -> 9999
                if len(temp) == 3:
                    # check if pole|dipole
                    if np.allclose(temp[0], temp[1]):
                        tx = np.r_[temp[0], 9999]
                        is_pole_tx = True
                    else:
                        tx = np.r_[temp[0], 9999, temp[1], 9999]
                    is_surface = True

                else:
                    # check if pole|dipole
                    if np.allclose(temp[0:2], temp[2:4]):
                        tx = np.r_[temp[0:2]]
                        is_pole_tx = True
                    else:
                        tx = temp[:-1]

                continue

            # Extract receivers
            temp = np.fromstring(obsfile[ii], dtype=float, sep=" ")

            if is_surface:
                data_column_index = 2

                # Check if Pole Receiver
                if np.allclose(temp[0], temp[1]):
                    is_pole_rx = True
                    rx.append(np.r_[temp[0], 9999])
                else:
                    rx.append(np.r_[temp[0], 9999, temp[1], 9999])

            else:
                data_column_index = 4  # Since dpred for dc has app_res

                # Check if Pole Receiver
                if np.allclose(temp[0:2], temp[2:4]):
                    is_pole_rx = True
                    rx.append(temp[:2])
                else:
                    rx.append(temp[:4])

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
                    Rx = dc.receivers.Pole(rx[:, :2], data_type=data_type)
                else:
                    Rx = dc.receivers.Dipole(rx[:, :2], rx[:, 2:], data_type=data_type)
                if is_pole_tx:
                    source_list.append(dc.sources.Pole([Rx], tx[:2]))
                else:
                    source_list.append(dc.sources.Dipole([Rx], tx[:2], tx[2:]))

        survey = dc.survey.Survey(source_list)
        data_out = Data(survey=survey)

        if len(d) > 0:
            data_out.dobs = d

        if len(wd) > 0:
            data_out.standard_deviation = wd

        if is_surface:
            warnings.warn(
                "Loaded data were in surface format. Elevations automatically set to 9999 m. "
                "Use the project_to_discretized_topography method of the survey to project "
                "electrode locations to the discretized surface."
            )

    return data_out


def read_dcip3d_ubc(file_name, data_type):
    """Read UBC-GIF DCIP3D formatted survey or data files.

    This method can load survey locations, predicted data or observations
    files formatted for the UBC-GIF DCIP3D coding package. For more, see
    the `UBC-GIF DCIP3D online manual <https://dcip3d.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    file_name : str
        The file path to the data file
    data_type : {'volt', 'apparent_chargeability', secondary_potential'}
        Data type.

    Returns
    -------
    SimPEG.data.Data
        A SimPEG data object. The data from the input file is loaded and parsed into
        three attributes of the data object:

        - `survey`: the survey geometry as defined by an instance of :class`SimPEG.electromagnetics.static.resitivity.survey.Survey` or :class`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
        - `dobs`: observed/predicted data if present in the data file
        - `standard_deviations`: uncertainties (if observed data file) or apparent resistivities (if predicted data file)

    """

    data_type = validate_string(
        "data_type",
        data_type,
        [
            "volt",
            "apparent_chargeability",
            "secondary_potential",
        ],
    )

    # Prevent circular import
    from ...electromagnetics.static import resistivity as dc
    from ...data import Data

    # Load file
    obsfile = np.genfromtxt(file_name, delimiter="\n", dtype=str, comments="!")

    # Pre-allocate
    source_list = []
    d = []
    wd = []

    # Flag for z value provided
    is_surface = False
    is_pole_tx = False
    is_pole_rx = False

    # IP data for dcip3d has a line with a flag we can remove.
    if obsfile[0][0:6] == "IPTYPE":
        obsfile = obsfile[1:]

    # Since SimPEG defines secondary potential from IP as voltage,
    # we must use this type when defining the receivers.
    if data_type == "secondary_potential":
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

            # Check if z value is provided, if False -> 9999
            if len(temp) == 5:
                # check if pole|dipole
                if np.allclose(temp[0:2], temp[2:4]):
                    tx = np.r_[temp[0:2], 9999]
                    is_pole_tx = True
                else:
                    tx = np.r_[temp[0:2], 9999, temp[2:4], 9999]
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
                rx.append(np.r_[temp[0:2], 9999, temp[2:4], 9999])

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

    survey = dc.survey.Survey(source_list)
    data_out = Data(survey=survey)

    if len(d) > 0:
        data_out.dobs = d

    if len(wd) > 0:
        data_out.standard_deviation = wd

    if is_surface:
        warnings.warn(
            "Loaded data were in surface format. Elevations automatically set to 9999 m. "
            "Use the project_to_discretized_topography method of the survey to project "
            "electrode locations to the discretized surface."
        )

    return data_out


def read_dcipoctree_ubc(file_name, data_type):
    """Read UBC-GIF DCIP OcTree formatted survey or data files.

    This method can load survey locations, predicted data or observations
    files formatted for the UBC-GIF DCIP OcTree coding package. For more, see
    the `UBC-GIF DCIP OcTree online manual <https://dcipoctree.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    file_name : str
        The file path to the data file
    data_type : {'volt', 'apparent_chargeability', secondary_potential'}
        Data type.

    Returns
    -------
    SimPEG.data.Data
        A SimPEG data object. The data from the input file is loaded and parsed into
        three attributes of the data object:

        - `survey`: the survey geometry as defined by an instance of :class`SimPEG.electromagnetics.static.resistivity.survey.Survey` or :class`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
        - `dobs`: observed/predicted data if present in the data file
        - `standard_deviations`: uncertainties (if observed data file) or apparent resistivities (if predicted data file)

    """

    return read_dcip3d_ubc(file_name, data_type)


def write_dcip2d_ubc(
    file_name,
    data_object,
    data_type,
    file_type,
    format_type="general",
    comment_lines=None,
):
    """Write UBC-GIF DCIP2D formatted survey or data files.

    This function can write survey locations, predicted data or observations
    files formatted for the UBC-GIF DCIP2D coding package. For more, see
    the `UBC-GIF DCIP2D online manual <https://dcip2d.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    file_name : str
        file path for output file
    data_object :
        SimPEG.data.Data object. The `survey` attribute of this data object must be
        an instance of :class`SimPEG.electromagnetics.static.resistivity.survey.Survey` or
        :class`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
    data_type : {'volt', 'apparent_chargeability', 'secondary_potential'}
        The type of data.
    file_type : {'survey', 'dpred', 'dobs'}
        Whether to include predicted data/observed data in the file.
    format_type : {'general', 'surface', 'simple'}
        The format of the file to write.
    comment_lines :
        Comment lines printed to beginning of the file
    """

    # Prevent circular import
    from ...electromagnetics.static.resistivity.sources import (
        Pole as PoleSrc,
        Dipole as DipoleSrc,
    )
    from ...electromagnetics.static.resistivity.receivers import (
        Pole as PoleRx,
        Dipole as DipoleRx,
    )

    from ...data import Data

    # Validate inputs
    data_object = validate_type("data_object", data_object, Data, cast=False)

    data_type = validate_string(
        "data_type",
        data_type,
        [
            "volt",
            "apparent_chargeability",
            "secondary_potential",
        ],
    )

    file_type = validate_string(
        "file_type",
        file_type,
        [
            "survey",
            "dpred",
            "dobs",
        ],
    )

    format_type = validate_string(
        "format_type",
        format_type,
        [
            "general",
            "surface",
            "simple",
        ],
    )

    # Write comments and IP type (if applicable)
    with open(file_name, "w") as fid:

        if format_type != "simple":
            fid.write("COMMON_CURRENT\n")

        fid.write(f"! {format_type} FORMAT\n")

        if comment_lines is not None and len(comment_lines) > 0:
            # ensure comment_lines ends with a new line character
            if comment_lines[0] != "!":
                comment_lines = "! " + comment_lines
            if comment_lines[-1] != "\n":
                comment_lines += "\n"
            fid.write(comment_lines)

        if format_type.lower() != "simple":
            fid.write("{:g}\n".format(len(data_object.survey.source_list)))

        # DCIP3D will allow user to choose definition of IP data. DC data has no flag.
        # DCIPoctree IP data is always apparent chargeability.
        if data_type == "apparent_chargeability":
            fid.write("IPTYPE=1\n")
        elif data_type == "secondary_potential":
            fid.write("IPTYPE=2\n")

        if format_type.lower() == "simple":

            out_array = np.c_[
                data_object.survey.locations_a[:, 0],
                data_object.survey.locations_b[:, 0],
                data_object.survey.locations_m[:, 0],
                data_object.survey.locations_n[:, 0],
            ]

            if file_type != "survey":
                out_array = np.c_[out_array, data_object.dobs]

            if file_type == "dobs":
                out_array = np.c_[out_array, data_object.standard_deviation]

            np.savetxt(fid, out_array, fmt="%e", delimiter="    ")

        else:

            # Index deciding if z locations are written
            if format_type == "surface":
                end_index = 1
            elif format_type == "general":
                end_index = 2

            # Loop over all sources
            count = 0
            for src in data_object.survey.source_list:

                # Write Source
                nD = src.nD

                if isinstance(src, PoleSrc):
                    tx = src.location[0]
                    tx = np.repeat(np.r_[[tx]], 2, axis=0)
                elif isinstance(src, DipoleSrc):
                    tx = np.c_[src.location]

                fid.writelines("%e " % ii for ii in mkvc(tx[:, 0:end_index].T))
                fid.write(f"{nD}\n")

                # Write receivers
                for rx in src.receiver_list:

                    if isinstance(rx, DipoleRx):
                        M = rx.locations[0][:, 0:end_index]
                        N = rx.locations[1][:, 0:end_index]
                    elif isinstance(rx, PoleRx):
                        M = rx.locations[:, 0:end_index]
                        N = rx.locations[:, 0:end_index]

                    if file_type.lower() != "survey":
                        N = np.c_[N, data_object.dobs[count : count + rx.nD]]

                    if file_type.lower() == "dobs":
                        N = np.c_[
                            N, data_object.standard_deviation[count : count + rx.nD]
                        ]

                    # Write receivers and locations
                    if isinstance(N, np.ndarray):
                        np.savetxt(
                            fid,
                            np.c_[M, N],
                            fmt="%e",
                        )
                    else:
                        raise Exception(
                            """Uncertainities SurveyObject.std should be set.
                            Either float or nunmpy.ndarray is expected, """
                            "not {}".format(type(data_object.relative_error))
                        )

                    fid.write("\n")

                    count += rx.nD


def write_dcip3d_ubc(
    file_name,
    data_object,
    data_type,
    file_type,
    format_type="general",
    comment_lines=None,
):
    """Write UBC-GIF DCIP3D formatted survey or data files.

    This function can write survey locations, predicted data or observations
    files formatted for the UBC-GIF DCIP3D coding package. For more, see
    the `UBC-GIF DCIP3D online manual <https://dcip3d.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    file_name : str
        file path for output file
    data_object :
        SimPEG.data.Data object. The `survey` attribute of this data object must be
        an instance of :class`SimPEG.electromagnetics.static.resistivity.survey.Survey` or
        :class`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
    data_type : {'volt', 'apparent_chargeability', 'secondary_potential'}
    file_type : {'survey', 'dpred', 'dobs'}
    format_type : {'general', 'surface'}
    comment_lines : str
        Comments added to beginning of output file
    """

    # Prevent circular import
    from ...electromagnetics.static.utils.static_utils import (
        apparent_resistivity_from_voltage,
    )
    from ...electromagnetics.static.resistivity.sources import (
        Pole as PoleSrc,
        Dipole as DipoleSrc,
    )
    from ...electromagnetics.static.resistivity.receivers import (
        Pole as PoleRx,
        Dipole as DipoleRx,
    )
    from ...data import Data

    # Validate inputs
    data_object = validate_type("data_object", data_object, Data, cast=False)

    data_type = validate_string(
        "data_type",
        data_type,
        [
            "volt",
            "apparent_chargeability",
            "secondary_potential",
        ],
    )

    file_type = validate_string(
        "file_type",
        file_type,
        [
            "survey",
            "dpred",
            "dobs",
        ],
    )

    format_type = validate_string(
        "format_type",
        format_type,
        [
            "general",
            "surface",
        ],
    )

    # Predicted DC data will automatically contain apparent resistivity column.
    # Here we compute the apparent resistivities and treat it like an uncertainties column.
    if (file_type == "dpred") & (data_type == "volt"):
        data_object.standard_deviation = apparent_resistivity_from_voltage(
            data_object.survey, data_object.dobs
        )
        file_type = "dobs"

    # Write comments and IP type (if applicable)
    with open(file_name, "w") as fid:
        fid.write(f"! {format_type} FORMAT\n")

        if comment_lines is not None and len(comment_lines) > 0:
            if comment_lines[0] != "!":
                comment_lines = "! " + comment_lines
            # ensure comment_lines ends with a new line character
            if comment_lines[-1] != "\n":
                comment_lines += "\n"
            fid.write(comment_lines)

        if data_type == "apparent_chargeability":
            fid.write("IPTYPE=1\n")
        elif data_type == "secondary_potential":
            fid.write("IPTYPE=2\n")

        # Index deciding if z locations are written
        if format_type == "surface":
            end_index = 2
        elif format_type == "general":
            end_index = 3

        # Loop over all sources
        count = 0
        for src in data_object.survey.source_list:

            # Write Source
            nD = src.nD

            if isinstance(src, PoleSrc):
                tx = src.location[0]
                tx = np.repeat(np.r_[[tx]], 2, axis=0)
            elif isinstance(src, DipoleSrc):
                tx = np.c_[src.location]

            fid.writelines("%e " % ii for ii in mkvc(tx[:, 0:end_index].T))
            fid.write(f"{nD}\n")

            # Write receivers
            for rx in src.receiver_list:

                if isinstance(rx, DipoleRx):
                    M = rx.locations[0][:, 0:end_index]
                    N = rx.locations[1][:, 0:end_index]
                elif isinstance(rx, PoleRx):
                    M = rx.locations[:, 0:end_index]
                    N = rx.locations[:, 0:end_index]

                if file_type != "survey":
                    N = np.c_[N, data_object.dobs[count : count + rx.nD]]

                if file_type == "dobs":
                    N = np.c_[N, data_object.standard_deviation[count : count + rx.nD]]

                # Write receivers and locations
                if isinstance(N, np.ndarray):
                    np.savetxt(
                        fid,
                        np.c_[M, N],
                        fmt="%e",
                    )
                else:
                    raise Exception(
                        """Uncertainities SurveyObject.std should be set.
                        Either float or nunmpy.ndarray is expected, """
                        "not {}".format(type(data_object.relative_error))
                    )

                fid.write("\n")

                count += rx.nD


def write_dcipoctree_ubc(
    file_name,
    data_object,
    data_type,
    file_type,
    format_type="general",
    comment_lines="",
):
    """Write UBC-GIF DCIP OcTree formatted survey or data files.

    This function can write survey locations, predicted data or observations
    files formatted for the UBC-GIF DCIP OcTree coding package. For more, see
    the `UBC-GIF DCIP OcTree online manual <https://dcipoctree.readthedocs.io/en/latest/>`__.

    Parameters
    ----------
    file_name : str
        file path for output file
    data_object :
        SimPEG.data.Data object. The `survey` attribute of this data object must be
        an instance of :class`SimPEG.electromagnetics.static.resistivity.survey.Survey` or
        :class`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
    data_type : {'volt', 'apparent_chargeability', 'secondary_potential'}
    file_type : {'survey', 'dpred', 'dobs'}
    format_type : {'general', 'surface'}
    comment_lines : str
        Comments added to beginning of output file
    """

    write_dcip3d_ubc(
        file_name,
        data_object,
        data_type,
        file_type,
        format_type=format_type,
        comment_lines=comment_lines,
    )


def write_dcip_xyz(
    file_name, data_object, data_header=None, uncertainties_header=None, out_dict=None
):
    """Write 2D or 3D DC/IP data to XYZ-formatted file.

    This function writes DC and/or IP data from the SimPEG framework to an XYZ-formatted
    data file; i.e. each row in the file defines the data for the unique electrode
    locations provided. This function is versatile enough to write 2D or 3D data.
    The data may include elevations for the electrodes or be surface formatted.

    Parameters
    ----------
    file_name : str
        Path to the file
    data_object : SimPEG.data.Data
        SimPEG.data.Data object. The `survey` attribute of this data object must be
        an instance of :class`SimPEG.electromagnetics.static.resistivity.survey.Survey` or
        :class`SimPEG.electromagnetics.static.induced_polarization.survey.Survey`
    data_header: str
        Header for the data column; i.e. the header for the data defined in the `dobs`
        attibute of the data object. If ``None``, these data are not written to file
    uncertainties_header : str
        Header for the uncertainties column; i.e. the header for the uncertainties defined in
        the `standard_deviations` attibute of the data object. If ``None``, these data are not
        written to file
    out_dict : dict
        A python dictionary containing the names and associated vectors for any additional data
        columns you wish to write to the file; out_dict = {header1: vec1, header2: vec2, ...}.
    """

    out_columns = np.c_[
        data_object.survey.locations_a,
        data_object.survey.locations_b,
        data_object.survey.locations_m,
        data_object.survey.locations_n,
    ]

    # Determine if 2D or 3D survey
    if np.shape(out_columns)[1] == 8:
        dim = 2
        out_headers = "XA    ZA    XB    ZB    XM    ZM    XN    ZN"
    else:
        dim = 3
        out_headers = (
            "XA    YA    ZA    XB    YB    ZB    XM    YM    ZM    XN    YN    ZN"
        )

    # Append data and uncertainties headers
    if (data_object.dobs is not None) & (data_header is not None):
        out_headers += "    " + data_header
        out_columns = np.c_[out_columns, data_object.dobs]

    if (data_object.standard_deviation is not None) & (
        uncertainties_header is not None
    ):
        out_headers += "    " + uncertainties_header
        out_columns = np.c_[out_columns, data_object.standard_deviation]

    # Append additional columns from dictionary
    if out_dict != None:
        for k in list(out_dict.keys()):
            out_headers += "    " + k
            out_columns = np.c_[out_columns, out_dict[k]]

    # Write to file
    np.savetxt(
        file_name,
        out_columns,
        fmt="%.8e",
        delimiter="    ",
        newline="\n",
        header=out_headers,
        comments="",
    )

    print("XYZ file saved to: " + file_name)
