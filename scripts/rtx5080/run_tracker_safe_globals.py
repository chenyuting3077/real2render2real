import runpy
import sys

import numpy as np
import torch

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))])
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
