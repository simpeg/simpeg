try:
    import simpeg.dask.simulation
    import simpeg.dask.electromagnetics.static.resistivity.simulation
    import simpeg.dask.electromagnetics.static.induced_polarization.simulation
    import simpeg.dask.potential_fields.base
    import simpeg.dask.potential_fields.gravity.simulation
    import simpeg.dask.potential_fields.magnetics.simulation
except ImportError as err:
    print("unable to load dask operations")
    print(err)
