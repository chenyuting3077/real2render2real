import numpy as np
import torch

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))])
