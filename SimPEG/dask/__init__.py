try:
    import SimPEG.dask.electromagnetics.static.resistivity.simulation
except ImportError as err:
    print('unable to load dask operations')
    print(err)
