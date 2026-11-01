"""Data setup script for V-COCO and COCO images.

Downloads MS-COCO 2014 annotations, extracts instances for V-COCO image IDs,
and optionally downloads images for train/val/test splits.
"""
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMAGE_BASE_URL = "http://images.cocodataset.org"


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    print(f"Downloading {url} to {dest_path}...")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks
        with open(temp_path, "wb") as f, tqdm(
            desc=desc, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    temp_path.replace(dest_path)
    print(f"Saved: {dest_path}")


def prepare_vcoco_coco_annotations(
    data_dir: Path, zip_cache_dir: Path, force: bool = False
) -> Path:
    out_json = data_dir / "instances_vcoco_all_2014.json"
    if out_json.exists() and not force:
        print(f"V-COCO COCO annotations already exist at {out_json}")
        return out_json

    ids_file = data_dir / "splits" / "vcoco_all.ids"
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing {ids_file}. Ensure V-COCO splits exist.")

    with open(ids_file, "r") as f:
        target_ids = {int(line.strip()) for line in f if line.strip()}
    print(f"Loaded {len(target_ids)} target image IDs from {ids_file}")

    zip_path = zip_cache_dir / "annotations_trainval2014.zip"
    if not zip_path.exists():
        download_file(COCO_ANNOTATIONS_URL, zip_path, desc="COCO Annotations Zip")

    print("Extracting and filtering annotations for V-COCO...")
    combined_images = []
    combined_annotations = []
    categories = None
    info = None
    licenses = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        for split in ["train2014", "val2014"]:
            inner_name = f"annotations/instances_{split}.json"
            print(f"Reading {inner_name} from zip...")
            with zf.open(inner_name) as f:
                data = json.load(f)
                if categories is None:
                    categories = data.get("categories", [])
                    info = data.get("info", {})
                    licenses = data.get("licenses", [])
                for img in data.get("images", []):
                    if img["id"] in target_ids:
                        combined_images.append(img)
                for ann in data.get("annotations", []):
                    if ann["image_id"] in target_ids:
                        combined_annotations.append(ann)

    print(f"Filtered {len(combined_images)} images and {len(combined_annotations)} annotations.")
    result = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": combined_images,
        "annotations": combined_annotations,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f)
    print(f"Successfully generated {out_json}")
    return out_json


def download_coco_images(
    data_dir: Path,
    image_dir: Path,
    split: str = "val",
    max_images: int = 50,
) -> None:
    """Download a subset of COCO 2014 images for the specified V-COCO split."""
    ids_file = data_dir / "splits" / f"vcoco_{split}.ids"
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing {ids_file}")

    with open(ids_file, "r") as f:
        image_ids = [int(line.strip()) for line in f if line.strip()]

    if max_images > 0:
        image_ids = image_ids[:max_images]

    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading up to {len(image_ids)} images for split '{split}'...")

    # Load annotations to map image_id to file_name and coco_url if available
    annot_file = data_dir / "instances_vcoco_all_2014.json"
    id_to_file = {}
    if annot_file.exists():
        with open(annot_file, "r") as f:
            annot_data = json.load(f)
            for img in annot_data["images"]:
                id_to_file[img["id"]] = (img.get("file_name"), img.get("coco_url"))

    for img_id in tqdm(image_ids, desc=f"Downloading {split} images"):
        file_name, coco_url = id_to_file.get(img_id, (None, None))
        if not file_name:
            # Fallback guessing: try train2014 or val2014 naming convention
            file_name = f"COCO_train2014_{img_id:012d}.jpg"
        
        dest_file = image_dir / file_name
        if dest_file.exists() and dest_file.stat().st_size > 0:
            continue

        # Try coco_url first, or standard URLs
        urls_to_try = []
        if coco_url:
            urls_to_try.append(coco_url)
        urls_to_try.append(f"{COCO_IMAGE_BASE_URL}/train2014/COCO_train2014_{img_id:012d}.jpg")
        urls_to_try.append(f"{COCO_IMAGE_BASE_URL}/val2014/COCO_val2014_{img_id:012d}.jpg")

        success = False
        for url in urls_to_try:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp, open(dest_file, "wb") as out:
                    out.write(resp.read())
                success = True
                break
            except Exception:
                if dest_file.exists():
                    dest_file.unlink()
                continue
        if not success:
            print(f"Warning: Failed to download image {img_id}")


def main():
    parser = argparse.ArgumentParser(description="V-COCO data setup helper")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--cache-dir", default="cache", type=Path)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-images", default=50, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepare_vcoco_coco_annotations(args.data_dir, args.cache_dir, force=args.force)

    if args.download_images:
        img_dest = Path("datasets/coco/images")
        download_coco_images(args.data_dir, img_dest, split=args.split, max_images=args.max_images)


if __name__ == "__main__":
    main()
