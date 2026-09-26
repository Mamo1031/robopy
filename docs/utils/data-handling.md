# データ処理

このページでは、Rakuda の記録データ（`RakudaExpHandler.record_save()` が保存するファイル）の形式と読み方を説明します。

## 保存されるファイル

`record_save(max_frames=..., save_path="test_01")` で番号キー（1〜9）を押すと、次のディレクトリに保存されます。

```text
data/test_01/<番号>_<連番>/
├── rakuda_observations.h5      # アーム・カメラ・触覚・音声（HDF5、gzip 圧縮）
├── metadata.json               # タスク情報・設定・データ形状・control ブロック
├── arm_obs.jpg                 # leader / follower の関節角プロット
└── rakuda_obs_animation.gif    # save_gif=True のときのみ
```

## `rakuda_observations.h5`（schema v2）

`N` は記録フレーム数、関節の並び（17 個）は `arm@joint_names` の順です（`RAKUDA_JOINT_NAMES` と同じ）。
`H5Handler` は配列をすべて float32 で保存します。

### `arm/` のデータセット

| データセット | 形状 | 単位・規約 |
|---|---|---|
| `arm/leader`, `arm/follower` | `(N, 17)` | 位置 [count]。v1 から変更なし |
| `arm/leader_velocity`, `arm/follower_velocity` | `(N, 17)` | 速度 [count]（1 count = 0.229 rpm）。関節規約（カウント増加方向が正） |
| `arm/leader_current`, `arm/follower_current` | `(N, 17)` | 電流 [mA] = `PRESENT_CURRENT` × 単位。**モーターの生符号**（関節規約ではない） |
| `arm/leader_time_s`, `arm/follower_time_s` | `(N,)` | 各バス読み取りの完了時刻。記録開始（`t0`）からの秒 |
| `arm/frame_time_s` | `(N,)` | フレームを確定した時刻。記録開始からの秒 |

- 位置と速度はカウント増加方向が正で、関節規約に揃っています。電流だけがモーターの規約のままなので、関節規約の電流が必要なら `arm@leader_current_sign` の符号表を掛けます（下記）。
- リーダーの XC330 の `PRESENT_CURRENT` は電源入力側の電流、XM430/XM540（リーダーの `torso_yaw` とフォロワ）は相電流です。
- 時刻について前提にしてよいのは「`frame_time_s` は非負で単調非減少」「`leader_time_s`・`follower_time_s` ≤ `frame_time_s`」の 2 点だけです。バイラテラル記録では最初のフレームの `leader_time_s` がわずかに負になることがあります（リーダーの読み取りが `t0` より前）。
- `camera/<名前>`, `tactile/<名前>`, `audio/<名前>` は v1 から変更ありません。

!!! note "データセットが無い場合"
    v2 のデータセットは値が取れたときだけ書かれます。`record_with_fixed_leader()` はリーダーを読まないので、`arm/leader_velocity`・`arm/leader_current`・`arm/leader_time_s` がありません（`arm/leader` には与えた指令値が入ります）。

### `arm` グループの属性

| 属性 | 型 | 内容 |
|---|---|---|
| `schema_version` | int | `2`（属性が無ければ v1） |
| `joint_names` | str | 17 関節名のカンマ区切り |
| `leader_models`, `follower_models` | str | モーター型番のカンマ区切り（`joint_names` と同じ順） |
| `position_unit` / `velocity_unit` / `current_unit` / `time_unit` | str | `"count"` / `"count_0.229rpm"` / `"mA"` / `"s_since_record_start"` |
| `control_mode` | str | `"position_teleop"`（従来）または `"leader_current"`（バイラテラル） |
| `record_fps` | int | 記録 fps |
| `control_hz` | int | 制御ループ周波数。バイラテラル記録のみ |
| `current_sign_convention` | str | `"motor_raw"` |
| `leader_current_sign`, `follower_current_sign` | str | 17 個の整数のカンマ区切り。`+1` / `-1` はループが使った電流符号（通常は `robopy-rakuda-gravity sign-check` の結果）、**`0` は未計測** |
| `time_origin` | str | `"monotonic_ns_at_record_start"` |
| `t0_monotonic_ns` | int | 記録開始時の `time.monotonic_ns()`（int64） |
| `t0_unix_s` | float | 記録開始時の `time.time()`（壁時計。単調ではない） |
| `terminated_by` | str | `"max_frame"` / `"keyboard_interrupt"` / `"teleop_stopped"` / `"loop_fault"` |
| `frames_requested` | int | 要求フレーム数（`max_frames`）。実際のフレーム数は `N` |

- `leader_current_sign` はバイラテラルループの `current_sign`（電流制御した関節のみ `±1`）です。従来モードの記録では全て `0` です。
- `follower_current_sign` は現状（Stage 1）常に全て `0` です。フォロワの電流符号は計測していません。
- 関節規約の電流への変換は `i_joint = leader_current * sign`（`sign == 0` の関節は変換できません）。

### 読み方（v1 / v2 両対応）

`arm/leader` と `arm/follower` は v1 と同じなので、既存の読み込みコードはそのまま動きます。
v2 のデータセットは必ず `in` で存在を確認してから使います。
`H5Handler.load_hierarchical()` はデータセットだけを読むので、属性は `h5py` で直接読みます。

```python
import h5py
import numpy as np

path = "data/test_01/1_1/rakuda_observations.h5"

with h5py.File(path, "r") as f:
    arm = f["arm"]
    leader = arm["leader"][()]          # (N, 17) count
    follower = arm["follower"][()]      # (N, 17) count

    version = int(arm.attrs.get("schema_version", 1))
    if "leader_current" in arm:
        current = arm["leader_current"][()]  # (N, 17) mA, motor raw sign
        sign = np.array(
            [int(s) for s in arm.attrs["leader_current_sign"].split(",")], dtype=np.float32
        )
        measured = sign != 0
        joint_current = current[:, measured] * sign[measured]  # joint convention
    if "frame_time_s" in arm:
        t = arm["frame_time_s"][()]      # (N,) s since t0
```

サンプル: `examples/data_process/read_h5.py`, `examples/data_process/h5_format_usage.py`

## `metadata.json`

`task_details`（`MetaDataConfig`）、`data_shape`（`None` のフィールドを除いた各配列の形状）、`robot_config`（`RakudaConfig` 全体。`bilateral` のパラメータと `hold_on_disconnect` を含む）に加えて、`control` に `RakudaRobot.control_report()` の結果が入ります。

### `control` の主なキー

| キー | 内容 |
|---|---|
| `mode` | 実際に動いたモード: `"position_teleop"` / `"leader_current"` |
| `config_mismatch` | `mode` と `robot_config.bilateral` が食い違ったときだけ `true` |
| `control_hz` | ループ周波数（バイラテラルのみ） |
| `state`, `joints`, `current_sign`, `current_sign_convention` | ループ状態、電流制御した関節、その符号表、`"motor_raw"`（バイラテラルのみ。従来モードの `current_sign` は `{}`） |
| `measured` | ループのタイミング: `cycles`, `ok_cycles`, `period_ms` / `read_ms` / `write_ms`（各 `p50`/`p95`/`p99`/`max`/`count`）, `overruns`, `skipped_cycles`, `read_fail_count`, `hold_window_ms` |
| `preflight` | 起動前検査の結果: `registers`, `drive_mode`, `warnings`, `notes`, `problems`, `read_ms`, `range_checked`, `stale_current_mode_recovered` |
| `configure` | `goal_current_rewritten_at_torque_on`, `current_limit_raw` |
| `hold` | 停止時の保持の記録（未停止なら `null`）: `source`, `sag_counts`, `window_ms`, `goal_rewritten_at_torque_on`, `none`, `verified`, `notes` |
| `faults` | フォルトのリスト。各要素は `reason`, `detail`, `t_ns`, `t_s`（記録開始からの秒。記録外は `null`）, `hold_window_ms`, `state_after`。先頭が原因 |
| `follower_io` | フォロワ入出力: `attached`, `divider`, `hz_effective`, `age_ms`, `step_ms`, `failures`, `lost` |
| `feedback_gate` | 力覚ゲート: `drops`, `low_cycles`, `gate`, `engaged` |
| `sampler` | 記録側の計数: `frames`, `queue_empty_waits`, `over_budget_frames`, `duplicate_snapshots`（フォロワ快照が前フレームと同じだった数） |
| `record` | 直前の記録: `t0_monotonic_ns`, `t0_unix_s`, `duration_s`, `frames`, `frames_requested`, `terminated_by`, `teleop_hz_effective`, `worker_error` |
| `teleop_hz_effective` | `record.teleop_hz_effective` と同じ値 |
| `setup` | バイラテラル設定の出所: `source`（識別ファイルのパス等）, `gravity_validated`, `uncompensated`（重力項なしで動かしたとき `true`） |
| `motors` | `names`, `leader_models`, `follower_models` |
| `dynamixel_sdk` | 実際に import された SDK の `version` と `path` |
| `ports` | 解決後のポート（`auto` を指定した場合も実デバイス名） |

従来モード（`bilateral=None`）では `mode`, `current_sign`（`{}`）, `record`, `faults`（`[]`）, `teleop_hz_effective`, `sampler`, `motors`, `dynamixel_sdk`, `ports` だけが入ります。

!!! warning "フォルトで終わった記録"
    `terminated_by` が `"loop_fault"` の記録は、フォルトまでに集めたフレームが保存されています（`N < frames_requested`）。原因は `control.faults[0].reason` を見てください。0 フレームで終わった記録は `RuntimeError` になり保存されません。

## 関連ページ

- [Rakuda](../robots/rakuda.md)
- [実験ハンドラー](../experiments/handlers.md)
- [アニメーション](animation.md)
