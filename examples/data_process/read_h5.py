import h5py

from robopy.utils.h5_handler import H5Handler

# rakuda_observations.h5 スキーマ v2 の任意データセット（旧ファイルには無い）。
OPTIONAL_ARM_DATASETS = (
    "leader_velocity",
    "follower_velocity",
    "leader_current",
    "follower_current",
    "leader_time_s",
    "follower_time_s",
    "frame_time_s",
)

if __name__ == "__main__":
    # H5ファイルのパス
    h5_file_path = "path/to/your/hierarchical_data.h5"

    # H5ファイルの情報を取得して表示
    file_info = H5Handler.get_info(h5_file_path)
    print("H5 File Information:")
    for key, value in file_info.items():
        print(f"  {key}: shape={value['shape']}, dtype={value['dtype']}")

    # 階層的なデータを読み込む
    hierarchical_data = H5Handler.load_hierarchical(h5_file_path)
    print("\nLoaded Hierarchical Data:")
    for category, datasets in hierarchical_data.items():
        for name, data in datasets.items():
            print(f"  {category}/{name}: shape={data.shape}, dtype={data.dtype}")

    # スキーマ v2: 速度・電流・時刻は存在するときだけ使う
    arm = hierarchical_data.get("arm", {})
    present = [name for name in OPTIONAL_ARM_DATASETS if name in arm]
    print(f"\nOptional arm datasets present: {present}")

    # 属性（schema_version, joint_names, 単位, 電流符号 ...）は h5py で直接読む
    with h5py.File(h5_file_path, "r") as f:
        if "arm" in f:
            attrs = dict(f["arm"].attrs)
            print(f"arm schema_version: {attrs.get('schema_version', 1)}")
            if "leader_current_sign" in attrs:
                print(f"leader_current_sign: {attrs['leader_current_sign']}")
