import h5py

# Run [ python remove_attr.py ]in your terminal
h5_file_path = ""
attr_to_remove = ""
removed = 0

with h5py.File(h5_file_path, "a") as f:
    if attr_to_remove in f.attrs:
        del f.attrs[attr_to_remove]
        removed += 1
        print("Removed")

    def visit(name, obj):
        global removed
        if attr_to_remove in obj.attrs:
            del obj.attrs[attr_to_remove]
            removed += 1

    f.visititems(visit)

# Confirm the attribute is removed
print(f"Removed '{attr_to_remove}' from {removed} places.")
