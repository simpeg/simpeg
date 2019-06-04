Models and Mapping
******************

In SimPEG, the model represents the set of parameters that define our geology.
For many problems, the model is a vector containing one or more physical
property values for each cell in the mesh. However, we will see that there are
other ways to parameterize models.

Simulations and inverse problems in SimPEG require the user to define a
mapping which goes from the model space to the vector space containing the
mesh. For different types of models, we will demonstrate how this mapping is
generated and applied.