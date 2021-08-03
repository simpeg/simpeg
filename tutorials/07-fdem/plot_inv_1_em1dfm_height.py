# """
# 1D Inversion of for Conductivity and Flight Height
# ==================================================

# Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to invert
# frequency domain data and recover a 1D electrical conductivity model and the
# flight height. In this tutorial, we focus on the following:

#     - How to define sources and receivers in this case
#     - How to define the survey
#     - Sparse 1D inversion of with iteratively re-weighted least-squares

# For this tutorial, we will invert 1D frequency domain data for a single sounding.
# The end product is layered Earth model which explains the data and the estimated
# flight height of the system. The survey consisted of a vertical magnetic dipole
# source at an unknown height. The receiver measured the vertical
# component of the secondary field at a 10 m offset from the source in ppm.

# """


# #########################################################################
# # Import modules
# # --------------
# #

# import os, tarfile
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# from discretize import TensorMesh

# import SimPEG.electromagnetics.frequency_domain_1d as em1d
# from SimPEG.electromagnetics.utils.em1d_utils import get_vertical_discretization_frequency, plot_layer
# from SimPEG.utils import mkvc
# from SimPEG import (
#     maps, data, data_misfit, inverse_problem, regularization, optimization,
#     directives, inversion, utils
#     )

# plt.rcParams.update({'font.size': 16, 'lines.linewidth': 2, 'lines.markersize':8})

# # sphinx_gallery_thumbnail_number = 3

# #############################################
# # Download Test Data File
# # -----------------------
# #
# # Here we provide the file path to the data we plan on inverting.
# # The path to the data file is stored as a
# # tar-file on our google cloud bucket:
# # "https://storage.googleapis.com/simpeg/doc-assets/em1dfm.tar.gz"
# #

# # storage bucket where we have the data
# data_source = "https://storage.googleapis.com/simpeg/doc-assets/em1dfm.tar.gz"

# # download the data
# downloaded_data = utils.download(data_source, overwrite=True)

# # unzip the tarfile
# tar = tarfile.open(downloaded_data, "r")
# tar.extractall()
# tar.close()

# # path to the directory containing our data
# dir_path = downloaded_data.split(".")[0] + os.path.sep

# # files to work with
# data_filename = dir_path + "em1dfm_data.txt"


# #############################################
# # Load Data and Plot
# # ------------------
# #
# # Here we load and plot the 1D sounding data. In this case, we have the 
# # secondary field response in ppm for a set of frequencies.
# #

# # Load field data
# #dobs = np.loadtxt(str(data_filename))
# dobs = np.loadtxt(str(data_filename), skiprows=1)

# # Define receiver locations and observed data
# frequencies = dobs[:, 0]
# dobs = mkvc(dobs[:, 1:])

# fig, ax = plt.subplots(1,1, figsize = (7, 7))
# ax.loglog(frequencies, np.abs(dobs[0:len(frequencies)]), 'k-o', lw=3)
# ax.loglog(frequencies, np.abs(dobs[len(frequencies):]), 'k:o', lw=3)
# ax.set_xlabel("Frequency (Hz)")
# ax.set_ylabel("|Hs/Hp| (ppm)")
# ax.set_title("Observed Data")
# ax.legend(["Real", "Imaginary"])


# #############################################
# # Defining the Survey
# # -------------------
# # 
# # Here we demonstrate a general way to define the receivers, sources, waveforms and survey.
# # The receiver measured the vertical component of the secondary field
# # at a 10 m offset from the source in ppm. However, we do not know the true
# # height of the source and receiver. We only know their offset.
# #

# # Source geometry. In this case, the vertical location is a dummy value because the
# # flight height is unknown.
# source_location = np.array([0., 0., 30.])
# moment_amplitude = 1.

# source_receiver_offset = np.array([10., 0., 0.])
# receiver_orientation = "z"
# field_type = "ppm"

# # Receiver list. Because we are inverting for the flight height, we MUST set
# # the receiver location as the offset between the source and receiver.
# receiver_list = []
# receiver_list.append(
#     em1d.receivers.PointReceiver(
#         source_receiver_offset, frequencies, orientation=receiver_orientation,
#         field_type=field_type, component="real", use_source_receiver_offset=True
#     )
# )
# receiver_list.append(
#     em1d.receivers.PointReceiver(
#         source_receiver_offset, frequencies, orientation=receiver_orientation,
#         field_type=field_type, component="imag", use_source_receiver_offset=True
#     )
# )
    
# # Source list
# source_list = [
#     em1d.sources.MagneticDipoleSource(
#         receiver_list=receiver_list, location=source_location, orientation="z",
#         moment_amplitude=moment_amplitude
#     )
# ]

# # Survey
# survey = em1d.survey.EM1DSurveyFD(source_list)


# ###############################################################
# # Assign Uncertainties and Define the Data Object
# # -----------------------------------------------
# #
# # Here is where we define the data that are inverted. The data are defined by
# # the survey, the observation values and the uncertainties.
# #

# # 5% of the absolute value
# uncertainties = 0.05*np.abs(dobs)*np.ones(np.shape(dobs))

# # Define the data object
# data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)


# ###############################################################
# # Defining a 1D Layered Earth (1D Tensor Mesh)
# # --------------------------------------------
# #
# # Here, we define the layer thicknesses for our 1D simulation. To do this, we use
# # the TensorMesh class.
# #

# # Layer thicknesses
# inv_thicknesses = np.logspace(0,1.5,25)

# # Define a mesh for plotting and regularization.
# mesh = TensorMesh([(np.r_[inv_thicknesses, inv_thicknesses[-1]])], '0')


# #################################################################
# # Define a Starting and/or Reference Model and the Mapping
# # --------------------------------------------------------
# #
# # Here, we create starting and/or reference models for the inversion as
# # well as the mapping from the model space. Starting and
# # reference models can be a constant background value or contain a-priori
# # structures.
# # 
# # Since we are inverting for the layer condutivity and flight height, both quantities
# # must be accounted for in the starting model, reference model and mapping
# # Here, the starting conductivity is log(0.1) S/m and our initial estimate
# # of the flight height is 25 m.


# # Define starting model.
# starting_conductivity = np.log(0.1*np.ones(mesh.nC))
# starting_height = 35.
# starting_model = np.r_[starting_conductivity, starting_height]

# # Use the *Wires* mapping to define which model parameters are conductivities
# # and which is the height.
# wires = maps.Wires(('sigma', mesh.nC),('h', 1))

# # Define mapping from model to conductivities
# sigma_map = maps.ExpMap() * wires.sigma

# # Define mapping from model to flight height
# h_map = wires.h


# #######################################################################
# # Define the Physics
# # ------------------
# #

# simulation = em1d.simulation.EM1DFMSimulation(
#     survey=survey, thicknesses=inv_thicknesses, sigmaMap=sigma_map, hMap=h_map
# )


# #######################################################################
# # Define Inverse Problem
# # ----------------------
# #
# # The inverse problem is defined by 3 things:
# #
# #     1) Data Misfit: a measure of how well our recovered model explains the field data
# #     2) Regularization: constraints placed on the recovered model and a priori information
# #     3) Optimization: the numerical approach used to solve the inverse problem
# #
# #

# # Define the data misfit. Here the data misfit is the L2 norm of the weighted
# # residual between the observed data and the data predicted for a given model.
# # The weighting is defined by the reciprocal of the uncertainties.
# dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
# dmis.W = 1./uncertainties

# # Define the regularization for the conductivity
# reg_sigma = regularization.Sparse(
#     mesh, mapping=wires.sigma, alpha_s=0.1, alpha_x=1.
# )

# p = 1.
# q = 0.
# reg_sigma.norms = np.c_[p, q]

# # Define the regularization for the flight height
# reg_height = regularization.Sparse(
#     TensorMesh([1]), mapping=wires.h,
# )

# p = 1.
# reg_sigma.p = p

# # Combine the regularization objects
# reg = reg_sigma + reg_height
# reg.mref = starting_model

# # Define how the optimization problem is solved. Here we will use an inexact
# # Gauss-Newton approach that employs the conjugate gradient solver.
# opt = optimization.ProjectedGNCG(maxIter=50, maxIterLS=20, maxIterCG=20, tolCG=1e-3)

# # Define the inverse problem
# inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)


# #######################################################################
# # Define Inversion Directives
# # ---------------------------
# #
# # Here we define any directiveas that are carried out during the inversion. This
# # includes the cooling schedule for the trade-off parameter (beta), stopping
# # criteria for the inversion and saving inversion results at each iteration.
# #

# # Defining a starting value for the trade-off parameter (beta) between the data
# # misfit and the regularization.
# starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# # Update the preconditionner
# update_Jacobi = directives.UpdatePreconditioner()

# # Options for outputting recovered models and predicted data for each beta.
# save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# # Directives for IRLS
# update_IRLS = directives.Update_IRLS(
#     max_irls_iterations=30, minGNiter=1,
#     coolEpsFact=1.5, update_beta=True
# )

# # Updating the preconditionner if it is model dependent.
# update_jacobi = directives.UpdatePreconditioner()

# # Add sensitivity weights
# sensitivity_weights = directives.UpdateSensitivityWeights()

# # The directives are defined as a list.
# directives_list = [
#     sensitivity_weights,
#     starting_beta,
#     save_iteration,
#     update_IRLS,
#     update_jacobi,
# ]

# #####################################################################
# # Running the Inversion
# # ---------------------
# #
# # To define the inversion object, we need to define the inversion problem and
# # the set of directives. We can then run the inversion.
# #

# # Here we combine the inverse problem and the set of directives
# inv = inversion.BaseInversion(inv_prob, directives_list)

# # Run the inversion
# recovered_model = inv.run(starting_model)


# #####################################################################
# # Plotting Results
# # ---------------------


# # Load the true model and layer thicknesses
# true_model = np.array([0.1, 1., 0.1])
# hz = np.r_[20., 40., 160.]
# true_layers = TensorMesh([hz])

# # Extract Least-Squares model
# l2_model = inv_prob.l2model

# # Plot true model and recovered model
# fig = plt.figure(figsize=(8, 9))
# x_min = np.min(np.r_[sigma_map * recovered_model, sigma_map * l2_model, true_model])
# x_max = np.max(np.r_[sigma_map * recovered_model, sigma_map * l2_model, true_model])

# ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
# plot_layer(true_model, true_layers, ax=ax1, showlayers=False, color="k")
# plot_layer(sigma_map * l2_model, mesh, ax=ax1, showlayers=False, color="b")
# plot_layer(sigma_map * recovered_model, mesh, ax=ax1, showlayers=False, color="r")
# ax1.set_xlim(0.01, 10)
# ax1.set_title("True and Recovered Models")
# ax1.legend(["True Model", "L2-Model", "Sparse Model"])
# ax1.text(0.15, 150, "True height = {} m".format(30))
# ax1.text(0.15, 165, "Recovered height = {0:.2f} m".format((h_map * recovered_model)[0]))
# plt.gca().invert_yaxis()

# # Plot predicted and observed data
# dpred_l2 = simulation.dpred(l2_model)
# dpred_final = simulation.dpred(recovered_model)

# fig = plt.figure(figsize=(11, 6))
# ax1 = fig.add_axes([0.2, 0.1, 0.6, 0.8])
# ax1.loglog(frequencies, np.abs(dobs[0:len(frequencies)]), "k-o")
# ax1.loglog(frequencies, np.abs(dobs[len(frequencies):]), "k:o")
# ax1.loglog(frequencies, np.abs(dpred_l2[0:len(frequencies)]), "b-o")
# ax1.loglog(frequencies, np.abs(dpred_l2[len(frequencies):]), "b:o")
# ax1.loglog(frequencies, np.abs(dpred_final[0:len(frequencies)]), "r-o")
# ax1.loglog(frequencies, np.abs(dpred_final[len(frequencies):]), "r:o")
# ax1.set_xlabel("Frequencies (Hz)")
# ax1.set_ylabel("|Hs/Hp| (ppm)")
# ax1.set_title("Predicted and Observed Data")
# ax1.legend([
#     "Observed (real)", "Observed (imag)",
#     "L2-Model (real)", "L2-Model (imag)",
#     "Sparse (real)", "Sparse (imag)"],
#     loc="upper left"
# )
# plt.show()

