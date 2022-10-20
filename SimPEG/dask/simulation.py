from ..simulation import BaseSimulation as Sim


@property
def chunk_format(self):
    "Apply memory chunks along rows of G, either 'equal', 'row', or 'auto'"
    return self._chunk_format


@chunk_format.setter
def chunk_format(self, other):
    if other not in ["equal", "row", "auto"]:
        raise ValueError("Chunk format must be 'equal', 'row', or 'auto'")
    self._chunk_format = other


Sim.chunk_format = chunk_format


@property
def max_ram(self):
    "Maximum ram in (Gb)"
    return self._max_ram


@max_ram.setter
def max_ram(self, other):
    if other <= 0:
        raise ValueError("max_ram must be greater than 0")
    self._max_ram = other


Sim.max_ram = max_ram


@property
def max_chunk_size(self):
    "Largest chunk size (Mb) used by Dask"
    return self._max_chunk_size


@max_chunk_size.setter
def max_chunk_size(self, other):
    if other <= 0:
        raise ValueError("max_chunk_size must be greater than 0")
    self._max_chunk_size = other


Sim.max_chunk_size = max_chunk_size

# add dask options to BaseSimulation.__init__
_old_init = Sim.__init__


def __init__(
    self,
    mesh=None,
    survey=None,
    solver=None,
    solver_opts=None,
    sensitivity_path="./sensitivity/",
    counter=None,
    verbose=False,
    chunk_format="row",
    max_ram=16,
    max_chunk_size=128,
    **kwargs,
):
    _old_init(
        self,
        mesh=mesh,
        survey=survey,
        solver=solver,
        solver_opts=solver_opts,
        sensitivity_path=sensitivity_path,
        counter=counter,
        verbose=verbose,
        **kwargs,
    )
    self.chunk_format = chunk_format
    self.max_ram = max_ram
    self.max_chunk_size = max_chunk_size


Sim.__init__ = __init__
