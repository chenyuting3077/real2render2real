import numpy as np
import torch

try:
    from numpy._core.multiarray import scalar as _scalar
except ImportError:
    from numpy.core.multiarray import scalar as _scalar
torch.serialization.add_safe_globals([
    _scalar,
    (_scalar, "numpy.core.multiarray.scalar"),
    (_scalar, "numpy._core.multiarray.scalar"),
    np.dtype,
    type(np.dtype(np.float64)),
    type(np.dtype(np.float32)),
    type(np.dtype(np.int64)),
])
