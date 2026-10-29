# 執行（RTX 5080 + conda）

> ⚠️ 警告：以下流程目前只確認「能跑完」，各步驟的結果正確性（切割、追蹤、grasp、`part_deltas` 區段、產生的資料內容）都還沒逐項驗證。

安裝見 [install_rtx5080_conda.md](install_rtx5080_conda.md)。每一步都佔大部分 GPU 記憶體，一次只跑一個。

```bash
conda activate r2r2r
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES
```

`run_tracker_safe_globals.py`：

```python
import runpy
import sys

import numpy as np
import torch

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))])
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
```

`$HOME/safe_globals_site/sitecustomize.py`：上面的 imports 加 `add_safe_globals` 那一行。

## 完整流程（cardboard_box）

```bash
S=data/corl_data/jy_scans/cardboard_box
W=run_tracker_safe_globals.py

ns-train garfield --data $S --max-num-iterations 30000
python $W $CONDA_PREFIX/bin/ns-train dig --data $S --pipeline.garfield-ckpt outputs/cardboard_box/garfield/<ts>/config.yml

TORCH_COMPILE_DISABLE=1 PYTHONPATH=$HOME/viser027 \
  python $W $CONDA_PREFIX/bin/ns-viewer --load-config outputs/cardboard_box/dig/<ts>/config.yml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python $W dependencies/rsrd/scripts/run_tracker.py --is-obj-jointed False \
  --dig-config-path outputs/cardboard_box/dig/<ts>/config.yml \
  --video-path data/corl_data/jy_demonstrations/cardboard_box_lift.MOV \
  --output-dir outputs/cardboard_box/track --save-hand

PYTHONPATH=$HOME/safe_globals_site PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dependencies/SuGaR/full_train_pipeline.py -s $PWD/outputs/cardboard_box/state_rigid_<ts>

TORCH_COMPILE_DISABLE=1 python $W dependencies/rsrd/scripts/dev_grasp.py \
  --dig-config outputs/cardboard_box/dig/<ts>/config.yml

python dependencies/IsaacLab/scripts/tools/convert_mesh.py <state_dir>/<mesh>_fixed_normals_centered.obj \
  <state_dir>/usd/<mesh>_fixed_normals_centered.usd \
  --make-instanceable --collision-approximation none --mass 1.0 --headless

python scripts/run.py
```

- **`ns-viewer`：** 切出物體後存成 `state_rigid_<ts>.pt`。
- **`run_tracker.py`：** 寫出 `keyframes.txt` 後就可以停掉。
- **`dev_grasp`：** 在紙箱兩側各存一筆 grasp，第 0 筆給左手、第 1 筆給右手。
- **`run.py`：** 改用 `YumiCardboardPickupCfg` 和 `CardboardLift`；`yumi_cardboard_lift.py:133` 的 DiG 路徑和 scene cfg 的 `usd_path` 改成自己的。瀏覽器連上 `http://<host>:8080` 才會開始。
