# OpenArm 移植計畫

目標：用自己的 OpenArm 取代 YuMi，跑 R2R2R 的資料生成。
示範：`demonstrations/pick_up_pringle.mp4`（右手拿起、雙手交接、放下），物體 `outputs/pringles`。

**已決定：** 先做**右手**單手拿取；渲染相機用 OpenArm 的**頭部相機**。

## 現況：R2R2R 的模擬怎麼運作

```
scripts/run.py
 ├─ scene cfg：機器人 + 桌子 + 物體 USD + 渲染相機（MultiTiledCameraCfg）
 └─ 模擬器（如 TigerPickR2R2R）
     ├─ 讀 outputs/<物體>/track/keyframes.txt（物體軌跡）與 state_rigid_*/grasps.txt
     ├─ 狀態機：setup → grasp（夾住）→ 跟著軌跡走 → release（放開）
     ├─ IK：JaxMP，讀機器人 URDF（controllers/jaxmp_diff_ik_controller.py）
     └─ 夾爪：直接設定手指關節位置
```

範本用作者的 Franka 移植版 `real2render2real/isaaclab_viser/franka_simulators/franka_coffee_maker.py`：單手、夾爪用 `[-2:]`、使用存檔的 grasp 方向。tiger 版本會忽略存檔方向，固定從正上方夾。

## YuMi 與 OpenArm 的差異

| 項目 | YuMi | OpenArm | 程式位置 |
|---|---|---|---|
| IK 目標 | `left/right_dummy_joint`（link7 前 0.13 m） | `openarm_{left,right}_hand_tcp_joint`（hand 前 0.08 m） | 模擬器 `run_simulator()` |
| 末端位姿 link | `left/right_dummy_point` | `openarm_{left,right}_hand_tcp` | 同上 |
| 關節順序對應 | 寫死 `[1,3,5,…,-2,-4]` | 改成用關節名稱對應 | `run_simulator()`、`_ik_wrapped()` |
| 夾爪 | `[-4:-2]`、`[-2:]`，張開 0.024 | `openarm_*_finger_joint1/2`，0～0.044，張開 0.044 | `_set_gripper_positions()` |
| 手指閉合方向 | 沿 x 軸 | 沿 y 軸 | 影響 grasp 方向（步驟 4） |
| 夾得住的寬度 | 約 50 mm（夾不住罐身） | 約 88 mm（罐身約 72 mm） | — |
| 模型 | `data/yumi_*` | `Stanley_ws/lerobot_openarm/model/openarm_description.urdf` + meshes | — |

## 階段 A：OpenArm 右手單手拿取品客罐

### 步驟 0：環境與資源
- 在 r2r2r conda 環境（Isaac Sim 4.5 + IsaacLab 0.36.6）裡跑。`Stanley_ws/IsaacLab` 是 0.54.4（Isaac Sim 5.x），不要混用。`MultiTiledCameraCfg` 只有 R2R2R 的 IsaacLab 有。
- 把 OpenArm URDF 和 meshes 複製到 `data/openarm_description/`。

### 步驟 1：確認機器人能載入（風險最高，先做）
- 優先用 `UrdfFileCfg` 匯入 URDF。`base.py:113-118` 會自動拿同一份 URDF 給 IK 和 viser 用。
- 備案：`v1_camera_isaac.usd`（新版 Isaac Sim 做的，要測 4.5 能不能載入）。
- 確認關節名稱、手指範圍和連動、TCP 位置；用小腳本測 IK 能不能追到幾個目標點。

### 步驟 2：機器人參數 `configs/articulation_configs/openarm_cfg.py`
- 以 `OPENARM_BI_CFG` 為基礎（速度、力量上限、剛度）。
- 初始姿勢改成夾爪在桌面上方前方。

### 步驟 3：場景 `configs/scene_configs/openarm_*_cfg.py`
- 桌子高度和位置配合實機（OpenArm 肩膀離底座 0.698 m）。
- 品客罐 USD 用 `convert_mesh.py` 從清理過的 mesh 轉，起始位置要在右手可達範圍。
- 渲染相機放在頭部相機的位置（參考 `v1_camera_isaac.usd` 的相機架）。

### 步驟 4：`dev_grasp` 支援 OpenArm
- 新增只有夾爪的 OpenArm URDF（hand + 兩根手指 + TCP），保留 YuMi 選項。
- TCP 偏移 0.08，寬度滑桿上限 0.044。
- 舊的 YuMi grasp 要重存。
- `dev_grasp` 的夾爪 TCP 和模擬的 IK 目標是同一個 frame，grasp 方向就不用另外換算。

### 步驟 5：模擬器 `openarm_simulators/openarm_pringles_pick.py`（從 Franka 版改）
- IK：OpenArm URDF，`num_ees=1`，目標 `openarm_right_hand_tcp_joint`。
- 關節對應、夾爪關節都用名稱查。
- 使用存檔的 grasp 方向。
- `dig_config_path` = `outputs/pringles/dig/2026-10-30_181519`。
- 檢查起始位置隨機化（原本 ±0.23 m）在 OpenArm 可達範圍內。
- 重設夾爪離桌太低的安全門檻（YuMi 是 0.015 m）和成功條件。
- `part_deltas[4:]` 會丟掉前 4 格；10fps 影片的前 4 格罐子還沒動，沒關係。

### 步驟 6：`scripts/run.py`
- 加入 OpenArm 場景、模擬器、URDF 路徑；`num_envs` 先用 2～4（16 GB GPU）。

### 步驟 7：分段驗證
1. 機器人能出現，IK 能追到測試目標。
2. 不開物理，只讓右手沿罐子軌跡走一次，確認整段可達（含手腕轉 60°）。
3. 開物理跑完整 rollout，看成功率。
4. 檢查渲染畫面。

## 階段 B：雙手交接（A 穩定後）
- `num_ees=2`，兩筆 grasp：右手夾的位置、左手接的位置（錯開）。
- 狀態機：右手夾住 → 走軌跡 → 左手接住 → 右手放開退開 → 左手走完軌跡。
- 交接時間點從 `keyframes.txt` 的手部資訊推算（第 0–22 格右手、23–62 格雙手）。

## 進度

### 步驟 0、1：完成（2026-11-05）
- `data/openarm_description/openarm_bimanual.urdf` + `meshes/`：從 `Stanley_ws/lerobot_openarm/model` 複製，只把 mesh 路徑改成 `meshes/<name>.stl`。`data/` 不在 git 裡，換機器要重新複製。
- `scripts/openarm/check_openarm_load.py`：在 Isaac Sim 4.5 用 `UrdfFileCfg` 載入、驅動、對照 URDF FK。結果：
  - 18 個關節（左右各 7 + 每手 2 根手指），左右 TCP link 都在。必須 `merge_fixed_joints=False`，不然 TCP link 會被合併掉。
  - 手指：保留 mimic 時第二根手指不會跟著動，所以改用 `convert_mimic_joints_to_normal_joints=True` 兩根都直接驅動；轉換後 `finger_joint2` 範圍被改成 [-0.009, 0.053]，要用 `write_joint_position_limit_to_sim` 改回 [0, 0.044]。
  - Isaac 的 TCP 位置和 URDF FK 差 0.0 mm，關節都能準確到達目標。
  - 致動器的 regex 要寫 `openarm_(left|right)_joint[1-7]`，`openarm_.*_joint[1-7]` 會連 `finger_joint1` 一起配到。
- JaxMP IK（同一份 URDF）：IK 目標是 `openarm_right_hand_tcp_joint`。連續求解（模擬裡每步都會用上一次的解當起點）時，夾爪朝前的目標收斂到 0 mm / 0°；夾爪朝下又放太低的目標會卡在關節極限。OpenArm 比較適合**從側面夾**。
- 用固定朝前方向畫的可達範圍圖結果很零散，看起來是求解器卡在局部解，不是手臂真的搆不到。步驟 7.2 要用實際軌跡驗證，可能要調 IK 的初始姿勢、迭代次數或方向權重。

### 步驟 3 要用的實機資料（來自 `Stanley_ws/IsaacLab`）
- 頭部相機 `body_cam`：掛在 `openarm_body_link0`，`pos=(0.05621, -0.00305, 0.83062)`，`rot=(0.18720, -0.68244, 0.69543, -0.12490)`（ROS convention），640×480，focal 24，horizontal aperture 25.2956（`stack_ik_abs_visuomotor_env_cfg.py:169`）。
- 桌面（workspace pad）：高 0.28 m，0.48 × 0.57 m，近邊離機器人底座 0.03 m（`stack_joint_pos_env_cfg.py:133-150`）。罐子放在約 (0.55, 0.05)（`openarm_task_modes.py:2278`）。

### 步驟 2–7：完成（2026-11-05）
- **機器人／場景：** `configs/articulation_configs/openarm_cfg.py`、`configs/scene_configs/openarm_base_cfg.py`、`openarm_scene_cfg.py`；頭部相機姿勢存在 `data/camera_poses/openarm_head_cam.h5`。`scripts/openarm/check_openarm_scene.py` 會讓罐子落桌、存一張頭部相機畫面。
- **品客罐 USD：**
  - README 的 `--collision-approximation none` 讓會動的物體沒有碰撞形狀，罐子會穿過桌面，所以改用 `convexHull`。
  - 罐底沒掃到，碰撞外殼的底部是尖錐，一放就倒；所以把罐底 1.8 cm 的毛邊切掉（z < −0.116，從這個高度以上截面才是完整的圓）。
  - 原檔都在 `outputs/pringles/_unused/`。
- **dev_grasp：** `--gripper openarm` 使用 `data/openarm_description/openarm_hand.urdf`，TCP 0.08、手指寬度 0～0.044，抓取候選寬度上限 0.088（YuMi 的 0.045 會讓罐身完全沒有候選）。改動在 `patches/rsrd.patch`。
- **模擬器：** `openarm_simulators/openarm_pringles_pick.py`，入口 `scripts/openarm/run_openarm_pringles.py`（`--auto-start` 可不等瀏覽器直接跑）。和 YuMi tiger pick 的差別：
  - 用存檔的 grasp 方向；
  - 罐子的偏航角讓夾爪從機器人這一側接近（右偏 40°，±0.15 rad），罐子起點 (0.33, −0.10)；
  - 10fps 軌跡插值成 30fps；
  - IK 解先夾到關節範圍內再寫入。
- **grasp 的手腕方向：** 平行夾爪繞接近方向轉 180° 是同一個 grasp。離線 IK 搜尋（90 種組合）顯示右手要 TCP x 軸朝上才搆得到，所以模擬器載入時會自動把 x 軸朝下的 grasp 翻過來。
- **`base.py` 的兩個問題（YuMi 的模擬器都有自己的 override，所以沒碰到）：**
  - 用 URDF 載入機器人時，不會把路徑存回 `self.urdf_path`；
  - `render_wrapped_impl` 不會呼叫 `sim.render()`，相機畫面會一直是灰的。

- **驗證：** `run_openarm_pringles.py --auto-start --num-envs 2 --max-successes 4`，6 個回合成功 5 個，每回合 301 張 640×480 頭部相機畫面，加上 `robot_data.h5`（16 個關節：兩隻手臂 + 每手一根手指；左右 TCP 位姿；`gripper_binary_cmd [left, right]`）。
- **還沒做：** 正式的 grasp 要用 `dev_grasp --gripper openarm` 多存幾筆；渲染畫面顆粒感較重；大量生成前要先看成功率。

## 風險
1. OpenArm 在 Isaac Sim 4.5 能不能正常載入（步驟 1 就知道）。不行的話，要把 R2R2R 的模擬搬到新版 IsaacLab，工作量大很多。
2. 品客罐碰撞：README 用 `--collision-approximation none`。動態物體用三角網格碰撞、清理後的破洞會不會有問題，在步驟 7.3 驗證。
3. 品客罐 mesh 有掃描造成的缺口（罐底沒拍到、反光）。之後應改善拍攝方式重新掃描。
