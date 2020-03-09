Basic Inversions
****************

Inversion is used to recover a model from geophysical data. Recovered models are meant to approximate the distribution of a physical property within the survey region; i.e. density, magnetic susceptibility, electrical resistivity, etc. To carry out a geophysical inversion with SimPEG, the user
must:

    - have a set of field measurements, or data, to invert
    - assign uncertainties to the data
    - define the survey for the geophysical method that was used
    - define the mesh upon which the forward simulation is solved
    - define the mapping from the model to the mesh
    - define the inverse problem (data misfit, regularization, optimization)
    - specify directives for the inversion


For various types of geophysical data, we will demonstrate standard inversion approaches. More advanced inversion algorithms (joint-inversion, petrophysically constrained, etc.) are covered in the next set of tutorials.
