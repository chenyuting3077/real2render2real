# 安裝（RTX 5080 + conda）

環境：RTX 5080（sm_120）、driver 590.48、Ubuntu 22.04、`~/miniforge3`，不動系統 CUDA。全部裝在 conda 環境 `r2r2r`。

## data generation

```bash
conda create -y -n r2r2r python=3.10 && conda activate r2r2r
conda install -y pip
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES

cd ~/real2render2real && git submodule update --init --recursive
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
cd dependencies/IsaacLab && ./isaaclab.sh --install none
cd ../rsrd/dependencies && pip install -e jaxmp -e jaxls
cd ../.. && pip install -e trajgen viser 'viser[examples]' 'jax[cuda12]==0.5.3'
cd .. && pip install -e . mujoco==3.3.1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

P=$(ls -d $CONDA_PREFIX/lib/python3.10/site-packages/isaacsim/extscache/omni.kit.pip_archive-*/pip_prebundle)
mv $P/websockets $P/websockets.disabled_for_viser
```

## real-to-sim

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
pip install torchtyping==0.1.5 && pip install typeguard==4.4.4 warp-lang
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2
cd hamer && pip install --no-build-isolation -e .[all]
pip install --no-build-isolation -e third-party/ViTPose
bash fetch_demo_data.sh

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install 'jax[cuda12]==0.5.3' numpy==1.26.4 "cupy-cuda12x<14" "fast_simplification<0.2" viser==1.1.1
pip install --no-deps --target $HOME/viser027 viser==0.2.7 websockets==13.1
```

MANO：從 https://mano.is.tue.mpg.de 下載 `mano_v1_2.zip`，把 `models/MANO_RIGHT.pkl` 放到 `dependencies/rsrd/dependencies/hamer/_DATA/data/mano/`。

## SuGaR

```bash
export PYTHONNOUSERSITE=1 CUDA_HOME=$CONDA_PREFIX FORCE_CUDA=1 MAX_JOBS=6
cd ~/real2render2real/dependencies/SuGaR/gaussian_splatting/submodules
pip install --no-build-isolation ./simple-knn -e diff-gaussian-rasterization
cd ../.. && git clone https://github.com/NVlabs/nvdiffrast && pip install --no-build-isolation ./nvdiffrast
pip install open3d==0.17.0 pymcubes==0.1.4 plyfile pyquaternion
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install numpy==1.26.4 viser==1.1.1
```

## submodule 修改

| 檔案 | 修改 |
|---|---|
| `rsrd/dependencies/dig/dig/twodgs.py:48` | `TORCH_CUDA_ARCH_LIST` 改成 `"12.0"` |
| `rsrd/dependencies/hamer_helper/src/hamer_helper/_hamer_helper.py` | 照 README Common Issues 處理 `vitpose_model` |
| `rsrd/rsrd/robot/graspable_obj.py:315` | `simplify_quadric_decimation(face_count=150)` |
| `rsrd/rsrd/extras/grasp_helper.py` 兩處 `glob` | 排除 `graspable_part_mesh*` |
| `rsrd/data` | `ln -s ../../data dependencies/rsrd/data` |
| `SuGaR/.../diff-gaussian-rasterization/CMakeLists.txt:33` | `CUDA_ARCHITECTURES "120"` |
| `SuGaR/.../diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h` | 加 `#include <cstdint>` |
| `SuGaR/.../simple-knn/simple_knn.cu` | 加 `#include <cfloat>` |

## 資料

```bash
pip install gdown
gdown 1JA_8RcskQcFQKfjou4Sg9YzN8kDgsoeC -O data.zip
gdown 1r8nYx3WHf_6axkZAGn1RJJS6JmAKe6Jr -O outputs.zip
unzip -q data.zip && unzip -q outputs.zip && rm data.zip outputs.zip
```

`outputs/` 解壓後約 51 GB。遇到的問題記錄在 [issues](https://github.com/chenyuting3077/real2render2real/issues)。
