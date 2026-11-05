# 執行（RTX 5080 + conda）

> ⚠️ 警告：以下流程目前只確認「能跑完」，各步驟的結果正確性（切割、追蹤、grasp、`part_deltas` 區段、產生的資料內容）都還沒逐項驗證。

安裝見 [install_rtx5080_conda.md](install_rtx5080_conda.md)。每一步都佔大部分 GPU 記憶體，一次只跑一個。

```bash
conda activate r2r2r
export PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES
```

## 完整流程（cardboard_box）

```bash
S=data/corl_data/jy_scans/cardboard_box
W=scripts/rtx5080/run_tracker_safe_globals.py

ns-train garfield --data $S --max-num-iterations 30000
python $W $CONDA_PREFIX/bin/ns-train dig --data $S --pipeline.garfield-ckpt outputs/cardboard_box/garfield/<ts>/config.yml

TORCH_COMPILE_DISABLE=1 PYTHONPATH=$HOME/viser027 \
  python $W $CONDA_PREFIX/bin/ns-viewer --load-config outputs/cardboard_box/dig/<ts>/config.yml

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python $W dependencies/rsrd/scripts/run_tracker.py --is-obj-jointed False \
  --dig-config-path outputs/cardboard_box/dig/<ts>/config.yml \
  --video-path data/corl_data/jy_demonstrations/cardboard_box_lift.MOV \
  --output-dir outputs/cardboard_box/track --save-hand

PYTHONPATH=$PWD/scripts/rtx5080/safe_globals_site PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dependencies/SuGaR/full_train_pipeline.py -s $PWD/outputs/cardboard_box/state_rigid_<ts>

TORCH_COMPILE_DISABLE=1 python $W dependencies/rsrd/scripts/dev_grasp.py \
  --dig-config outputs/cardboard_box/dig/<ts>/config.yml

python dependencies/IsaacLab/scripts/tools/convert_mesh.py <state_dir>/<mesh>_fixed_normals_centered.obj \
  <state_dir>/usd/<mesh>_fixed_normals_centered.usd \
  --make-instanceable --collision-approximation none --mass 1.0 --headless

python scripts/run.py
```

- **`ns-viewer`：** 切出物體後存成 `state_rigid_<ts>.pt`。
- **`PYTHONNOUSERSITE=1`：** 一定要設（可用 `conda env config vars set PYTHONNOUSERSITE=1 -n r2r2r` 永久設定）。沒設的話會用到 `~/.local` 裡的舊套件（例如 cupy 14 需要 numpy 2，`import cuml` 直接失敗）。
- **`state_rigid_*.pt`：** DiG 會把 `outputs/<物體>/` 底下**所有** `state_rigid_*.pt` 都當成獨立剛體載入。同一物體在 viewer 存了兩次，tracker 就會追兩個重疊的物體，各自漂移造成重影。只留要用的那份，其他的移到子資料夾。
- **`run_tracker.py`：** 寫出 `keyframes.txt` 後就可以停掉。改了物體切割後，要先清掉輸出目錄裡的 `track_init.json`，否則會沿用舊的初始位姿。追蹤時不要開 nerfstudio viewer（7007），開著會多佔 GPU 記憶體，轉動視角也會讓渲染執行緒丟出 `IOChangeException`。
- **`dev_grasp`：** 在紙箱兩側各存一筆 grasp，第 0 筆給左手、第 1 筆給右手。
- **`run.py`：** 改用 `YumiCardboardPickupCfg` 和 `CardboardLift`；`yumi_cardboard_lift.py:133` 的 DiG 路徑和 scene cfg 的 `usd_path` 改成自己的。瀏覽器連上 `http://<host>:8080` 才會開始。
