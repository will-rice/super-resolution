"""Download HuggingFace dataset."""

import argparse
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def main():
    """Download the dataset."""
    parser = argparse.ArgumentParser("Download the dataset.")
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()

    data_root = args.data_root / "laion_hd"
    data_root.mkdir(exist_ok=True, parents=True)

    ds = load_dataset("drhead/laion_hd_21M_deduped")

    # full_ds = datasets.concatenate_datasets([ds["train"], ds["test"]])
    full_ds = ds["train"]

    for idx, row in tqdm(enumerate(full_ds), total=len(full_ds)):
        url = row["URL"]
        if idx % 10000 == 0:
            sub_root = data_root / str(idx).zfill(len(str(len(full_ds))))
            sub_root.mkdir(exist_ok=True, parents=True)

        try:
            response = requests.get(url, stream=True, timeout=5)
            image = Image.open(response.raw).convert("RGBA").convert("RGB")

            if image.size[0] > 1024 and image.size[1] > 1024:
                name = str(idx).zfill(5)
                image.save(sub_root / f"{name}.png")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue


if __name__ == "__main__":
    main()
