# Data Generation 環境安裝（RTX 5080 + conda）

在 RTX 5080 上跑通 data generation 最小流程（YuMi coffee maker）的安裝紀錄。原本的 `env_data_gen.sh` 用 uv，torch 版本也不支援 Blackwell，所以改用 conda 並調整 torch 版本。

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
- **`run.py` 不會自己停**
  - 要累積到 1000 條成功軌跡才會結束（`yumi_coffee_maker.py:184`）。

## 實測數據

條件：YuMi coffee maker，`num_envs=2`。

| 項目 | 數值 |
|---|---|
| IK JIT 編譯（第一次） | 6.9 s |
| 成功軌跡產生速度 | 約 12 條/分鐘 |
| 每條軌跡大小 | 約 160 MB（兩個相機的 RGB + `robot_data.h5`） |
| GPU 記憶體 | 約 12.7 GB |
| 跑滿 1000 條 | 推估約 85 分鐘、160 GB |

跑之前要先確認硬碟空間夠不夠；如果只是要測試，可以把 1000 改小。
