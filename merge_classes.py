"""
合併相近類別（只改切分檔的標籤，不動任何圖片）。

會把 data/splits_base.json 的類別重新編號，原檔備份成 data/splits_base_18.json。
做完後請執行 `python augment.py` 重新產生 data/splits.json。

用法：
    python merge_classes.py
"""
import json
import shutil

from dataset import PROJECT_ROOT

MERGE = {
    "Full Black": "Black", "Partial Black": "Black",
    "Full Sour": "Sour", "Partial Sour": "Sour",
    "Slight Insect Damage": "Insect Damage", "Severe Insect Damage": "Insect Damage",
}


def main():
    base = PROJECT_ROOT / "data" / "splits_base.json"
    backup = base.with_name("splits_base_18.json")
    if not backup.exists():
        shutil.copy(base, backup)
    with open(backup, "r", encoding="utf-8") as f:
        data = json.load(f)

    old = data["class_names"]
    new = sorted({MERGE.get(n, n) for n in old})
    remap = {i: new.index(MERGE.get(n, n)) for i, n in enumerate(old)}
    data["class_names"] = new
    data["splits"] = {s: [[p, remap[l]] for p, l in v] for s, v in data["splits"].items()}
    with open(base, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"{len(old)} -> {len(new)} classes: {new}")


if __name__ == "__main__":
    main()
