from __future__ import print_function
import numpy as np
import time as tm
import re
import warnings
from discretize.utils import mkvc
from ...utils.code_utils import deprecate_method


########################################################################################
#                  DIRECT CURRENT RESISTIVITY AND INDUCED POLARIZATION
########################################################################################

def read_dcip3d_ubc(file_name):

    return _read_dcip_3d_or_octree_ubc(file_name, 'dcip3d')


def read_dcipoctree_ubc(file_name):

    return _read_dcip_3d_or_octree_ubc(file_name, 'dcipoctree')


def _read_dcip_3d_or_octree_ubc(file_name, code_type):
    """
    Read 3D DC/IP survey, predicted and observation files in UBC-GIF format.

    Parameters
    ----------

    file_name : str
        Path to the file.
    file_type : str
        File type. Choose from {'loc','dpred','dobs'}

    Returns
    -------
    SimPEG.electromagnetic.static.survey
        A DC/IP survey object. Locations will be defined. Observed data
        and uncertainties defined in the survey object will depend on
        **file_type**.

    """ 

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
            
        # Predicted IP data
        if len(temp) == data_column_index + 1:
            d.append(temp[-1])
        
        # Observed data or predicted DC data (since app res column)
        elif len(temp) == data_column_index + 2:
            d.append(temp[-2])
            wd.append(temp[-1])

        count = count - 1

        # Reach the end of transmitter block
        if count == 0:
            rx = np.asarray(rx)
            if is_pole_rx:
                Rx = dc.receivers.Pole(rx[:, :3])
            else:
                Rx = dc.receivers.Dipole(rx[:, :3], rx[:, 3:])
            if is_pole_tx:
                source_list.append(dc.sources.Pole([Rx], tx[:3]))
            else:
                source_list.append(dc.sources.Dipole([Rx], tx[:3], tx[3:]))

    if len(d) == 0:
        d = None
    else:
        d = np.asarray(d)
    
    if len(wd) == 0:
        wd = None
    else:
        wd = np.asarray(wd)

    # Define survey type
    if is_pole_tx:
        str1 = 'pole-'
    else:
        str1 = 'dipole-'

    if is_pole_rx:
        str2 = 'pole'
    else:
        str2 = 'dipole'

    electrode_configuration = str1 + str2

    survey = dc.Survey(source_list, survey_type=electrode_configuration)
    data_out = Data(survey=survey, dobs=d, standard_deviation=wd)
    return data_out




def write_dcip3d_ubc(
    file_name,
    data_object,
    file_type='dobs',
    format_type='general',
    data_type=0,
    electrode_configuration=None,
    comment_lines=""
    ):
    """
    Write UBC DCIP3D formatted survey, predicted or observation files.

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

    _write_dcip_3d_or_octree_ubc(
        file_name,
        data_object,
        file_type=file_type,
        format_type=format_type,
        data_type=data_type,
        electrode_configuration=electrode_configuration,
        code_type='dcip3d',
        comment_lines=comment_lines
    )


def write_dcipoctree_ubc(
    file_name,
    data_object,
    file_type='dobs',
    format_type='general',
    data_type=0,
    electrode_configuration=None,
    comment_lines=""
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

    _write_dcip_3d_or_octree_ubc(
        file_name,
        data_object,
        file_type=file_type,
        format_type=format_type,
        data_type=data_type,
        electrode_configuration=electrode_configuration,
        code_type='dcipoctree',
        comment_lines=comment_lines
    )


def _write_dcip_3d_or_octree_ubc(
    file_name,
    data_object,
    file_type='dobs',
    format_type='general',
    data_type=0,
    electrode_configuration=None,
    code_type='dcip3d',
    comment_lines=""
    ):
    """
    Write UBC DCIP3D formatted survey, predicted or observation files.

    Parameters
    ----------
    file_name:
    data_object:
    file_type: 'survey', 'dpred', 'dobs'
    format_type: 'general', 'surface', 'simple'
    data_type: 0 (DC), 1 (IP), 2 (another IP)
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
    if (file_type.lower() == 'dpred') & (data_type == 0):
        data_object.standard_deviation = apparent_resistivity(data_object)
        file_type = 'dobs'

    # Write comments and IP type (if applicable)
    fid = open(file_name, "w")
    fid.write("! " + format_type + " FORMAT\n")

    if comment_lines:
        fid.write(comment_lines)

    # DCIP3D will allow user to choose definition of IP data. DCIPoctree will not.
    if (data_type != 0) & (code_type.lower() == 'dcip3d'):
        fid.write("IPTYPE=%i\n" % data_type)

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
            
            if file_type.lower() != 'survey':
                N = np.c_[N, data_object.dobs[count : count+rx.nD]]

            if file_type.lower()=='dobs':
                N = np.c_[N, data_object.standard_deviation[count : count+rx.nD]]
            
            # Write receivers and locations
            fid = open(file_name, "ab")
            if isinstance(N, np.ndarray):
                np.savetxt(
                    fid, np.c_[M, N], fmt=str("%e"), delimiter=str(" "), newline=str("\n")
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