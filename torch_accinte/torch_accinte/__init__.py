import sys

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import torch_accinte._C  # type: ignore[misc]
import torch_accinte.accinte


torch.utils.rename_privateuse1_backend("accinte")
torch._register_device_module("accinte", torch_accinte.accinte)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)


# LITERALINCLUDE START: AUTOLOAD
def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass


# LITERALINCLUDE END: AUTOLOAD
