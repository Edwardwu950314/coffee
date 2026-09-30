"""
產生「好豆 / 壞豆」二分類切分檔 data/splits_binary.json。

標籤來自 data/binary_labels.csv（人工逐張判斷）。沿用 data/splits_base.json 的 train/val/test 分配，
USK-Coffee（split=usk_*）沿用原切分；另外若有 data/binary/good/ 的整盤裁切圖（split=extra）依比例隨機分到三份。

用法：
    python make_binary_split.py
    python train.py --split data/splits_binary.json
"""
import csv
import json
import random
from collections import Counter

from dataset import PROJECT_ROOT

CLASSES = ["bad", "good"]


def main(seed=42, val_ratio=0.15, test_ratio=0.15):
    labels = {}
    extra = []
    usk = {"train": [], "val": [], "test": []}
    with open(PROJECT_ROOT / "data" / "binary_labels.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            y = CLASSES.index(row["label"])
            if row["split"].startswith("usk_"):
                usk[row["split"][4:]].append([row["path"], y])
            elif row["split"] == "extra":
                extra.append((row["path"], y))
            else:
                labels[row["path"]] = y

    with open(PROJECT_ROOT / "data" / "splits_base.json", encoding="utf-8") as f:
        base = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for name, samples in base["splits"].items():
        for path, _ in samples:
            orig = path.replace("data/resized/", "data/")  # Normal 放大版 -> 原始路徑
            splits[name].append([path, labels[orig]])

    for name, items in usk.items():  # USK-Coffee 沿用它自己的 train/val/test
        splits[name] += items

    rng = random.Random(seed)
    rng.shuffle(extra)
    n_val, n_test = round(len(extra) * val_ratio), round(len(extra) * test_ratio)
    splits["val"] += [list(x) for x in extra[:n_val]]
    splits["test"] += [list(x) for x in extra[n_val:n_val + n_test]]
    splits["train"] += [list(x) for x in extra[n_val + n_test:]]

    with open(PROJECT_ROOT / "data" / "splits_binary.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASSES, "seed": seed, "splits": splits}, f, ensure_ascii=False, indent=1)
    for name, v in splits.items():
        print(name, len(v), {CLASSES[k]: c for k, c in sorted(Counter(y for _, y in v).items())})


if __name__ == "__main__":
    main()
