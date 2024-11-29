import subprocess
import os

download_dir = "data"

os.makedirs(download_dir, exist_ok=True)

urls = [
    "https://huggingface.co/datasets/lparkourer10/small/resolve/main/train.bin",
    "https://huggingface.co/datasets/lparkourer10/small/resolve/main/val.bin"
]

for url in urls:
    filename = os.path.join(download_dir, url.split('/')[-1])
    try:
        print(f"Downloading {url} to {filename}...")
        subprocess.run(["wget", "-O", filename, url], check=True)
        print(f"Successfully downloaded {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}: {e}")
