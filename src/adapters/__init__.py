import importlib
import logging

logger = logging.getLogger(__name__)

for module_name in (
    "hdf5_adapter",
    "ros_adapter",
    "lerobot_adapter",
    "unitree_adapter",
    "folder_adapter",
    "dasmcap_adapter",
    "singorix_adapter",
):
    try:
        importlib.import_module(f"{__name__}.{module_name}")
    except ImportError as exc:
        logger.warning("跳过 adapter %s，可选依赖缺失: %s", module_name, exc)
