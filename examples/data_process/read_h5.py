import h5py

from robopy.utils.h5_handler import H5Handler

# Optional datasets of rakuda_observations.h5 schema v2 (absent in older files).
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
    # Path of the H5 file
    h5_file_path = "path/to/your/hierarchical_data.h5"

    # Get and print the H5 file information
    file_info = H5Handler.get_info(h5_file_path)
    print("H5 File Information:")
    for key, value in file_info.items():
        print(f"  {key}: shape={value['shape']}, dtype={value['dtype']}")

    # Load the hierarchical data
    hierarchical_data = H5Handler.load_hierarchical(h5_file_path)
    print("\nLoaded Hierarchical Data:")
    for category, datasets in hierarchical_data.items():
        for name, data in datasets.items():
            print(f"  {category}/{name}: shape={data.shape}, dtype={data.dtype}")

    # Schema v2: velocity, current and time are used only when present
    arm = hierarchical_data.get("arm", {})
    present = [name for name in OPTIONAL_ARM_DATASETS if name in arm]
    print(f"\nOptional arm datasets present: {present}")

    # Attributes (schema_version, joint_names, units, current sign ...) are read with h5py
    with h5py.File(h5_file_path, "r") as f:
        if "arm" in f:
            attrs = dict(f["arm"].attrs)
            print(f"arm schema_version: {attrs.get('schema_version', 1)}")
            if "leader_current_sign" in attrs:
                print(f"leader_current_sign: {attrs['leader_current_sign']}")
