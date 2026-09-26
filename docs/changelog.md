# Changelog

## [Unreleased]

### Added
- Dynamixel バス層の追加: 期限付き状態読み取り `read_state_block()`、`write_goal_current_raw()`、読み戻し付き書き込み `write_with_readback()`、`read_diagnostics()`、`read_model_numbers()` / `verify_models()`、タイムアウト例外 `DynamixelTimeoutError`
- シミュレータ `SimulatedDynamixelBus` の追加（`DynamixelBus` の代替。実機なしで Rakuda の制御スタックとテストを実行可能）
- CLI `robopy-rakuda-ports` の追加（`scan` / `show` / `bench` / `set-return-delay` / `write-eeprom` / `release`）。`RakudaConfig` のポートに `"auto"` を指定するとスキャンで leader / follower を判定
- Rakuda バイラテラル制御（Stage 1）の追加: リーダー電流制御ループ `LeaderCurrentLoop`（単一制御スレッド、既定 50 Hz）、制御則 `BilateralLaw`（重力補償 + 可動域バリア + 位置誤差の力覚フィードバック）、停止・フォルト時の保持 `hold_joints()`、`RakudaPairSys.start_bilateral()` / `stop_bilateral()` / `release()`。`RakudaConfig(bilateral=RakudaBilateralParams(...))` を渡したときだけ有効
- リーダー重力同定 CLI `robopy-rakuda-gravity` の追加（`range` / `sign-check` / `identify` / `fit` / `verify` / `show`）。結果は `.robopy/rakuda/leader_gravity.json`
- 記録形式 schema v2: `arm/` に速度・電流・時刻（`leader_velocity`, `follower_velocity`, `leader_current`, `follower_current`, `leader_time_s`, `follower_time_s`, `frame_time_s`）と `arm` 属性（`schema_version`, 単位, 電流符号表, `t0_monotonic_ns`, `terminated_by` など）、`metadata.json` に `control` ブロック。詳細は [データ処理](utils/data-handling.md)
- サンプルの追加・更新: `examples/robot/rakuda_bilateral.py`（`--sim` でシミュレータ実行）、`examples/exp_handler/record.py` に `--bilateral`

### Changed
- Rakuda の `connect()`: モーター型番を照合し、不一致は `ConnectionError`
- Rakuda の `connect()`: 電流モードのまま残った関節を位置モードに戻す（トルクOFFの関節はそのまま復旧、トルクONの関節はその場で保持してから復旧）
- Rakuda の `connect()`: トルクを全関節OFF→ONし直すのをやめ、現在の状態との差分だけ切り替える。グリッパの EEPROM も値が違うときだけ書く
- 従来モードの `connect()`: 前回セッションが保持中の関節（トルクONで、設定上はOFFの関節）があれば脱力させずに `ConnectionError` で拒否（`robopy-rakuda-ports release` を案内）
- リーダーグリッパの接続時の目標位置を 2600 から 2400（テレオペ中と同じ値）に変更
- `.robopy/rakuda/config.yaml` に `leader.port` / `follower.port`（`auto` 可）、`safety.hold_on_disconnect`、`bilateral:` セクションを追加。既知セクション内の未知キーは `ValueError`
- `RakudaRobot.record()` / `record_parallel()` / `record_with_fixed_leader()` は 0 フレームで終わると `RuntimeError`。途中で止まった場合は集めたフレームを返し、理由を `terminated_by` に残す
- `record_with_fixed_leader()` はフォロワだけを読む（`leader_velocity` / `leader_current` / `leader_time_s` は記録されない）
- 記録の時刻基準を記録ごとの `time.monotonic_ns()` に統一

### Fixed
- `RakudaExpHandler._init_config()` が `RakudaConfig` を作り直して `slow_mode`・トルク設定・`bilateral`・`hold_on_disconnect` を落としていた問題を修正（`dataclasses.replace` で既定カメラだけ補う）
- `examples/exp_handler/record.py` が例外・Ctrl-C で `handler.close()` を呼ばずにポートを開いたままにしていた問題を修正（`finally: handler.close()`）
- フォロワグリッパの `CURRENT_LIMIT`（EEPROM）書き込みが確認されず、トルクONのときは無言で失敗し得た問題を修正（値が違うときだけ、トルクOFFにしてから読み戻し付きで書く。不一致は warning）
- 従来の `sync_read` / `sync_write` でシリアル書き込みが止まると呼び出し側が固まる、またはポートが使用中（`COMM_PORT_BUSY`）のまま残る問題を修正（書き込みタイムアウトで `DynamixelTimeoutError` にし、`is_using` を戻す）

## [0.3.2] - 2026-01-30

### Added
- 音声センサー統合の追加


## [0.2.0] - 2025-09-30

### Added
- Robopyの初期リリース
- Rakudaロボット対応
- Kochロボット対応（開発中）
- Intel RealSenseカメラサポート
- DIGITタクタイルセンサーサポート
- H5形式データ保存機能
- 実験ハンドラー機能

### Features
- テレオペレーション機能
- 並列データ記録
- マルチセンサー統合


## [0.1.0] - 2025-09-18

### Initial Release
- 基本的なロボット制御機能
