import os
import requests
import zipfile
from pathlib import Path
import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract variables from config
url = config['download']['url']
output_path_str = config['download']['output_path']

# Use Path for output path, and ensure its parent directories exist
output_path = Path(output_path_str)
output_path.parent.mkdir(parents=True, exist_ok=True)

download_dir = output_path.parent 

# -----------------------------
# Download ZIP file
# -----------------------------
print("Starting download...")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Download complete: {output_path}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
    exit(1)

# -----------------------------
# Extract files (flattened)
# -----------------------------
print("Extracting files...")
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    all_files = zip_ref.namelist()
    root_folder = "Plant_leave_diseases_dataset_with_augmentation/"

    for member in all_files:
        if member.startswith(root_folder) and member != root_folder:
            # Strip root folder
            relative_path = member[len(root_folder):]
            if not relative_path:  
                continue

            target_path = download_dir / relative_path

            if member.endswith('/'):
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
  
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

# -----------------------------
# Delete ZIP file
# -----------------------------
try:
    output_path.unlink()
    print(f"Deleted ZIP file: {output_path}")
except Exception as e:
    print(f"Error deleting ZIP file: {e}")
