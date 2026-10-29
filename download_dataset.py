
import kagglehub
import os

print("Downloading Cars Dataset 2025...")

path = kagglehub.dataset_download(
    "abdulmalik1518/cars-datasets-2025"
)

print("\nDownload complete!")
print("Dataset location:", path)

print("\nDataset folder structure:")

for root, dirs, files in os.walk(path):
    level = root.replace(path, "").count(os.sep)

    if level > 2:
        dirs[:] = []
        continue

    print("  " * level + os.path.basename(root) + "/")

    for filename in files[:8]:
        print("  " * (level + 1) + filename)
