try:
    import SimPEG.dask.simulation
    import SimPEG.dask.electromagnetics.static.resistivity.simulation
    import SimPEG.dask.electromagnetics.static.induced_polarization.simulation
except ImportError as err:
    print('unable to load dask operations')
    print(err)
