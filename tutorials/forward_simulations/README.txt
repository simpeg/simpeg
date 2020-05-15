Forward Simulations
===================

In SimPEG, the forward simulation represents the computation of predicted data
for a given model and geophysical survey. To carry out a forward simulation, the user
must define:

	- the numerical formulation for the geophysical problem being solved
	- the survey for the geophysical method being applied
	- the mesh upon which our numerical solution is computed
	- the model and a mapping from the model to the mesh

For various geophysical methods, we will demonstrate how to define the aforementionned
items and carry out the forward simulation. Data predicted by the simulation are then
plotted.
