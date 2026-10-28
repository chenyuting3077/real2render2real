# RTX 5080 + conda 環境安裝

在 RTX 5080 上用單一 conda 環境 `r2r2r` 跑通兩段流程的安裝紀錄：

- **data generation：** YuMi coffee maker
- **real-to-sim 追蹤：** cardboard_box，使用現成的 DiG

原本的 `env_data_gen.sh` 用 uv，torch 版本也不支援 Blackwell，所以改用 conda 並調整 torch 版本。

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

submodule 裡有四處要手動修改：

| 檔案 | 修改 |
|---|---|
| `rsrd/dependencies/dig/dig/twodgs.py:48` | `TORCH_CUDA_ARCH_LIST` 從 `'8.9'` 改成 `"12.0"` |
| `rsrd/dependencies/hamer_helper/src/hamer_helper/_hamer_helper.py` | 照 README 的 Common Issues 處理 `vitpose_model` |
| `rsrd/rsrd/robot/graspable_obj.py:315` | 改成 `simplify_quadric_decimation(face_count=150)` |
| `SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization/CMakeLists.txt:33` | `CUDA_ARCHITECTURES` 改成 `"120"` |

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

追蹤的結果是 `keyframes.txt`。寫出這個檔案之後，程式會接著做抓取取樣，並一直開著 viser，不會自己結束。抓取取樣的結果只會顯示在 viser 上，不會存檔，所以拿到 `keyframes.txt` 之後就可以直接停掉。

## 踩過的坑

- **torch 要用 2.7.0+cu128**
  - torch 2.5.1+cu121 沒有 sm_120 的 kernel，在 5080 上不能用。
  - IsaacLab 的 `setup.py` 鎖定 `torch==2.5.1`，所以 torch 要放在最後一步重裝。
  - 之後 `pip check` 會回報 isaaclab 的 torch 版本衝突。這是預期中的，實際執行沒有問題。
- **Isaac Sim 4.5.0 在 5080 上能跑 headless**
  - 實測可以正常建立場景和 step。
  - log 裡會出現 `[Error] GLInteropContext::init: carb::windowing is not available`，這是 headless 模式沒有視窗系統造成的，不影響執行。
- **conda 環境建好後要先 `conda install pip`**
  - miniforge 建出來的環境不一定有 pip。沒有的話，`pip` 會指到系統的 `/usr/bin/pip`，套件就會裝錯位置。
- **`PYTHONNOUSERSITE=1`**
  - `~/.local/lib/python3.10/site-packages` 裡有其他人裝的套件，不設定的話，會比 conda 環境裡的套件先被 import。
- **`OMNI_KIT_ACCEPT_EULA=YES`**
  - 沒設定的話，Isaac Sim 第一次啟動會停在互動式 EULA 確認畫面。
- **`isaaclab.sh --install none`**
  - data generation 用不到 RL 框架，省略可以縮短安裝時間。
- **websockets 版本衝突**
  - Isaac Sim 自帶的 websockets 12 會先被載入。viser 需要 13.1 以上，否則會出現 `No module named 'websockets.asyncio'`，詳見 [asyncio_error.md](../asyncio_error.md)。
  - 用改名的方式停用，不要直接刪除，之後要還原比較方便。
- **gdown 6.x 拿掉了 `--remaining-ok` 和 `--skip-download`**
  - 要先列出 Drive 資料夾內容的話，改用 `gdown --json <folder_url>`。
- **`outputs.zip` 裡 DiG 和 GARField 的 `config.yml`**
  - data 路徑寫成 `real2render2real/data/corl_data/...`，從 repo 根目錄執行會找不到檔案，要把開頭的 `real2render2real/` 刪掉。
- **setuptools 81 以後沒有 `pkg_resources`**
  - tiny-cuda-nn、chumpy、detectron2 的 `setup.py` 都還在用它，所以要降到 81 以下，並且加上 `--no-build-isolation`。
- **conda 的 cuda-toolkit 把 header 放在 `targets/x86_64-linux/include`**
  - torch 的即時編譯只會去 `$CUDA_HOME/include` 找，所以 gsplat 會報 `cuda_runtime_api.h: No such file`。
  - 解法是用 `conda env config vars` 設定 `CPATH`，只在 `r2r2r` 環境內生效。
- **gsplat 第一次執行要即時編譯，大約 8 分鐘**
  - 這段時間 log 不會動，看起來像卡住。
  - 編譯快取在 `~/.cache/torch_extensions/py310_cu128/gsplat_cuda`，改架構設定後要先刪掉。
- **`no kernel image is available`**
  - 原因是 `dig/twodgs.py` 在 import 時把 `TORCH_CUDA_ARCH_LIST` 覆寫成 8.9，gsplat 就照 sm_89 編譯。
- **torch 2.6 以後 `torch.load` 預設 `weights_only=True`**
  - nerfstudio 1.1.5 的 checkpoint 裡有 numpy scalar，會讀取失敗。
  - 只把需要的 numpy 型別加進白名單，不要全面關掉 `weights_only`。
- **依賴互相蓋版本**
  - **nerfstudio 1.1.5：** 會把 viser 降到 0.2.7。data generation 的 `camera_manager.py` 需要 1.x，否則會報 `CameraFrustumHandle` 沒有 `name`。
  - **fast_simplification 0.2：** 會把 numpy 拉到 2.x。Isaac Sim 和 cupy 13 都需要 1.26。
  - **cupy 14：** 需要 numpy 2。
  - 最後強制裝 viser 1.1.1，nerfstudio Viewer 實測仍然能啟動，只會多出 `add_dropdown` 的棄用警告。
- **`run.py` 不會自己停**
  - 要累積到 1000 條成功軌跡才會結束（`yumi_coffee_maker.py:184`）。

## 實測數據

data generation，條件：YuMi coffee maker，`num_envs=2`。

| 項目 | 數值 |
|---|---|
| IK JIT 編譯（第一次） | 6.9 s |
| 成功軌跡產生速度 | 約 12 條/分鐘 |
| 每條軌跡大小 | 約 160 MB（兩個相機的 RGB + `robot_data.h5`） |
| GPU 記憶體 | 約 12.7 GB |
| 跑滿 1000 條 | 推估約 85 分鐘、160 GB |

跑之前要先確認硬碟空間夠不夠；如果只是要測試，可以把 1000 改小。

real-to-sim 追蹤，條件：cardboard_box、`cardboard_box_lift.MOV`。

| 項目 | 數值 |
|---|---|
| HaMeR 手部偵測 | 57 幀，約 3.8 s/幀 |
| GPU 記憶體 | 約 13.4 GB |
| 和作者 `track/keyframes.txt` 比較 | 46 幀；物體平移平均差 1.9 cm，最大差 8.5 cm |

和作者結果的差距，推測來自套件版本不同和最佳化的隨機性，還沒驗證。驗證方式：用同一個 seed 重跑兩次，比較兩次之間的差異。
