try:
    import simpeg.dask.simulation
    import simpeg.dask.electromagnetics.frequency_domain.simulation
    import simpeg.dask.electromagnetics.static.resistivity.simulation
    import simpeg.dask.electromagnetics.static.resistivity.simulation_2d
    import simpeg.dask.electromagnetics.static.induced_polarization.simulation
    import simpeg.dask.electromagnetics.static.induced_polarization.simulation_2d
    import simpeg.dask.electromagnetics.time_domain.simulation
    import simpeg.dask.potential_fields.base
    import simpeg.dask.potential_fields.gravity.simulation
    import simpeg.dask.potential_fields.magnetics.simulation
    import simpeg.dask.simulation
    import simpeg.dask.data_misfit
    import simpeg.dask.inverse_problem
    import simpeg.dask.objective_function

except ImportError as err:
    print("unable to load dask operations")
    print(err)
