import sys
import h5py

# Run this in your terminal [ python test_labels_saved.py /full/path/to/your/file.h5 ]
if len(sys.argv) != 2:
    print("Usage: python test_labels_saved.py /full/path/to/your/file.h5")
    sys.exit(1)

path = sys.argv[1]

with h5py.File(path, "r") as f:

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and "checked" in obj.attrs:
            label = obj.attrs.get("quench_labels", None)
            print(f"{name!r:55} label={label!r:20}")

    f.visititems(visitor)
