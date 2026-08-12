"""PyWake

An open source wind farm simulation tool capable of calculating wind farm flow fields,
power production and annual energy production (AEP) of wind farms.
"""
import netCDF4
import numpy

from py_wake.utils.numpy_utils import NumpyWrapper

np = numpy
locals()['np'] = NumpyWrapper()

from py_wake.deficit_models.deficit_model import DeficitModel  # noqa: E402
from py_wake.deficit_models.fuga import Fuga, FugaBlockage  # noqa: E402
from py_wake.deficit_models.gaussian import (  # noqa: E402
    BastankhahGaussian,
    IEA37SimpleBastankhahGaussian,
    NiayifarGaussian,
    ZongGaussian,
)
from py_wake.deficit_models.gcl import GCL, GCLLocal  # noqa: E402
from py_wake.deficit_models.noj import NOJ, NOJLocal  # noqa: E402
from py_wake.flow_map import HorizontalGrid, XYGrid, XZGrid, YZGrid  # noqa: E402
from py_wake.literature.turbopark import Nygaard_2022  # noqa: E402
from py_wake.wind_farm_models.wind_farm_model import WindFarmModel  # noqa: E402

try:  # pragma: no cover
    # version.py created when installing py_wake
    from py_wake import version
    __version__ = version.__version__
    __release__ = version.__version__
except BaseException:  # pragma: no cover
    pass
