
"""
This downloads the tokenized dataset. run this to save compute lol
"""
import subprocess

urls = [
    "https://huggingface.co/datasets/lparkourer10/dataset/resolve/main/train.bin",
    "https://huggingface.co/datasets/lparkourer10/dataset/resolve/main/val.bin"
]

for url in urls:
    try:
        print(f"Downloading {url}...")
        subprocess.run(["wget", url], check=True)
        print(f"Successfully downloaded {url.split('/')[-1]}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}: {e}")
