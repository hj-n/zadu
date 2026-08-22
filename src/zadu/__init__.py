from importlib.metadata import PackageNotFoundError, version

from .engine import ExecutionConfig
from .spec import MEASURE, MeasureId, Spec, make_spec
from .zadu import ZADU

try:
    __version__ = version("zadu")
except PackageNotFoundError:  # Source checkout without installed metadata.
    __version__ = "0.5.0.dev0"

__all__ = [
    "MEASURE",
    "ZADU",
    "ExecutionConfig",
    "MeasureId",
    "Spec",
    "__version__",
    "make_spec",
]
