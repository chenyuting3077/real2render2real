# RTX 5080 + conda 環境安裝

在 RTX 5080 上用單一 conda 環境 `r2r2r` 跑通三段流程的安裝紀錄：

- **data generation：** YuMi coffee maker
- **real-to-sim 追蹤：** cardboard_box，使用現成的 DiG
- **完整流程：** cardboard_box，從掃描開始依序跑 GARField → DiG → 切割 → 追蹤 → SuGaR → `dev_grasp` → USD → data generation

## 測試環境

- GPU：RTX 5080 16GB（Blackwell，sm_120），driver 590.48.01
- OS：Ubuntu 22.04.5，系統沒有安裝 `nvcc`
- conda：`~/miniforge3`
- 限制：不能動系統層級的 CUDA 和 NVIDIA driver。CUDA runtime 全部由 pip wheel 提供。

## 安裝

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n r2r2r python=3.10
conda activate r2r2r
conda install -y pip
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES

cd ~/real2render2real
git submodule update --init --recursive
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
cd dependencies/IsaacLab && ./isaaclab.sh --install none
cd ../rsrd/dependencies && pip install -e jaxmp -e jaxls
cd ../.. && pip install -e trajgen viser 'viser[examples]' 'jax[cuda12]==0.5.3'
cd .. && pip install -e . mujoco==3.3.1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

P=$(ls -d $CONDA_PREFIX/lib/python3.10/site-packages/isaacsim/extscache/omni.kit.pip_archive-*/pip_prebundle)
mv $P/websockets $P/websockets.disabled_for_viser
```

## 安裝 real-to-sim 追蹤（接在上面的步驟之後）

```bash
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit
conda env config vars set CPATH=$CONDA_PREFIX/targets/x86_64-linux/include TORCH_CUDA_ARCH_LIST=12.0
conda deactivate && conda activate r2r2r
export PYTHONNOUSERSITE=1 CUDA_HOME=$CONDA_PREFIX TCNN_CUDA_ARCHITECTURES=120 MAX_JOBS=8

pip install "setuptools<81" wheel ninja
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install gsplat==1.4.0
cd ~/real2render2real/dependencies/rsrd && pip install -e .
pip install nerfstudio==1.1.5
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.10.* cuml-cu12==24.10.*
cd dependencies && pip install -e dig -e garfield -e hamer_helper
pip install torchtyping==0.1.5
pip install typeguard==4.4.4
pip install warp-lang
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2
cd hamer && pip install --no-build-isolation -e .[all]
pip install --no-build-isolation -v -e third-party/ViTPose
bash fetch_demo_data.sh

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install 'jax[cuda12]==0.5.3' numpy==1.26.4 "cupy-cuda12x<14" "fast_simplification<0.2" viser==1.1.1
```

還要把 MANO 的 `MANO_RIGHT.pkl` 放到 `dependencies/rsrd/dependencies/hamer/_DATA/data/mano/`。這個檔案在 https://mano.is.tue.mpg.de 註冊後，從「Models & Code」下載 `mano_v1_2.zip`，再從 `mano_v1_2/models/` 取出。

## 安裝 SuGaR（完整流程才需要）

先照下表改好 SuGaR 的三個檔案再裝：

```bash
export PYTHONNOUSERSITE=1 CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 MAX_JOBS=6
cd ~/real2render2real/dependencies/SuGaR/gaussian_splatting/submodules
pip install --no-build-isolation ./simple-knn
pip install --no-build-isolation -e diff-gaussian-rasterization
cd ../.. && git clone https://github.com/NVlabs/nvdiffrast
pip install --no-build-isolation ./nvdiffrast
pip install open3d==0.17.0 pymcubes==0.1.4 plyfile pyquaternion
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install numpy==1.26.4 viser==1.1.1
```

## submodule 的手動修改

| 檔案 | 修改 |
|---|---|
| `rsrd/dependencies/dig/dig/twodgs.py:48` | `TORCH_CUDA_ARCH_LIST` 從 `'8.9'` 改成 `"12.0"` |
| `rsrd/dependencies/hamer_helper/src/hamer_helper/_hamer_helper.py` | 照 README 的 Common Issues 處理 `vitpose_model` |
| `rsrd/rsrd/robot/graspable_obj.py:315` | 改成 `simplify_quadric_decimation(face_count=150)` |
| `rsrd/rsrd/extras/grasp_helper.py` 兩處 `glob` | 排除 `graspable_part_mesh*` |
| `rsrd/data` | 建立軟連結 `ln -s ../../data dependencies/rsrd/data`，`dev_grasp` 要讀夾爪 URDF |
| `SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization/CMakeLists.txt:33` | `CUDA_ARCHITECTURES` 改成 `"120"` |
| `SuGaR/.../diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h` | 加 `#include <cstdint>` |
| `SuGaR/gaussian_splatting/submodules/simple-knn/simple_knn.cu` | 加 `#include <cfloat>` |

## 下載資料

```bash
pip install gdown
gdown 1JA_8RcskQcFQKfjou4Sg9YzN8kDgsoeC -O data.zip
gdown 1r8nYx3WHf_6axkZAGn1RJJS6JmAKe6Jr -O outputs.zip
unzip -q data.zip && unzip -q outputs.zip
```

| 檔案 | zip 大小 | 解壓後 |
|---|---|---|
| `data.zip` | 1.6 GB | `data/` |
| `outputs.zip` | 14 GB | `outputs/`，51 GB |

解壓後就可以把 zip 刪掉。

## 執行

```bash
conda activate r2r2r
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES
python scripts/run.py
```

模擬要等到有瀏覽器連上 viser 的 `http://<host>:8080` 才會開始，網頁要一直開著。

real-to-sim 追蹤：

```bash
python run_tracker_safe_globals.py dependencies/rsrd/scripts/run_tracker.py \
  --is-obj-jointed False \
  --dig-config-path outputs/cardboard_box/dig/2025-04-18_204749/config.yml \
  --video-path data/corl_data/jy_demonstrations/cardboard_box_lift.MOV \
  --output-dir outputs/cardboard_box/track_test --save-hand
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

## 完整流程（cardboard_box）

每一步都會佔用大部分 GPU 記憶體，必須等上一步結束再跑下一步。

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
```

- **切割：** 在 `ns-viewer` 裡完成，存成 `state_rigid_<ts>.pt`。
- **`viser027`：** 只給 `ns-viewer` 用的獨立目錄，用 `pip install --no-deps --target $HOME/viser027 viser==0.2.7 websockets==13.1` 建立。
- **`safe_globals_site/sitecustomize.py`：** 內容就是上面 `run_tracker_safe_globals.py` 的 `add_safe_globals` 那一行。SuGaR 會再開子程序，用 wrapper 包不到子程序，所以改用 `sitecustomize`。
- **data generation：** `yumi_cardboard_lift.py:133` 寫死了作者的 DiG 時間戳，scene cfg 的 `usd_path` 也指向作者的 USD，要改成自己的路徑。

遇到的問題和實測數據記錄在 [issues](https://github.com/chenyuting3077/real2render2real/issues)。
