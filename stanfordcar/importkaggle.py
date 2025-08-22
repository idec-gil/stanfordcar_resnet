import kagglehub

# Download latest version
path = kagglehub.dataset_download("jutrera/stanford-car-dataset-by-classes-folder")

print("Path to dataset files:", path)
