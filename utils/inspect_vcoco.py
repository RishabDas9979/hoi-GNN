"""Print the real structure of the V-COCO and COCO annotation files.
Run from project root:  python -m utils.inspect_vcoco
"""
import json
from pathlib import Path

VCOCO = Path("datasets/vcoco")
COCO_ANN = Path("datasets/coco/annotations")


def brief(v, limit=70):
    if isinstance(v, list):
        return f"list(len={len(v)}) first={str(v[:3])[:limit]}"
    if isinstance(v, dict):
        return f"dict(keys={list(v)[:6]})"
    return f"{type(v).__name__}: {str(v)[:limit]}"


def describe(obj, name):
    print(f"\n=== {name} ===")
    print(f"top-level type: {type(obj).__name__}", end="")
    if isinstance(obj, list):
        print(f", len={len(obj)}")
        for i, item in enumerate(obj[:2]):
            print(f"  item[{i}] type={type(item).__name__}")
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"    {k}: {brief(v)}")
    elif isinstance(obj, dict):
        print(f", keys={list(obj)[:10]}")
        for k, v in list(obj.items())[:6]:
            print(f"  {k}: {brief(v)}")


print("--- files under datasets/vcoco/data ---")
for p in sorted((VCOCO / "data").rglob("*")):
    if p.is_file():
        print(f"{p.relative_to(VCOCO)}  ({p.stat().st_size:,} bytes)")

print("\n--- COCO annotation files ---")
for p in sorted(COCO_ANN.glob("*")):
    print(f"{p.name}  ({p.stat().st_size:,} bytes)")

for p in sorted((VCOCO / "data").rglob("*.json")):
    if p.stat().st_size < 400_000_000:
        with open(p, "r", encoding="utf-8") as f:
            describe(json.load(f), str(p.relative_to(VCOCO)))

for p in sorted((VCOCO / "data").rglob("*.ids"))[:3] + \
         sorted((VCOCO / "data").rglob("*.txt"))[:3]:
    lines = p.read_text(errors="ignore").splitlines()
    print(f"\n=== {p.relative_to(VCOCO)} === {len(lines)} lines, first: {lines[:3]}")

with open(COCO_ANN / "instances_val2014.json", "r", encoding="utf-8") as f:
    d = json.load(f)
print("\n=== COCO instances_val2014.json ===")
print("keys:", list(d))
print("annotation[0] keys:", list(d["annotations"][0]))
print("image[0]:", d["images"][0])
