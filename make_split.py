"""
產生資料切分檔 data/splits.json（不會移動或刪除任何圖片）。

做的事情：
1. 讀取 data/train 與 data/test（一個類別一個資料夾）。
2. 用檔案內容的 MD5 找出「完全相同的圖片」：
   - 同一張圖出現在不同類別 → 標籤互相矛盾，全部排除。
   - 同一張圖同時出現在 train 與 test → 只保留 test 那份，避免「考題洩漏」到訓練資料。
   - 同一類別裡重複的圖 → 只保留一張。
3. 從 train 依類別比例（分層抽樣）切出 val，用來挑最佳權重、調整學習率；
   test 只在訓練結束後評估一次，確保分數客觀。

用法：
    python make_split.py
    python make_split.py --val-ratio 0.2 --seed 0
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from dataset import PROJECT_ROOT, get_class_names, list_images


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Create a deduplicated train/val/test split file.")
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "data" / "train")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "splits.json")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_names = get_class_names(args.train_dir)
    test_classes = get_class_names(args.test_dir)
    if class_names != test_classes:
        raise ValueError(
            f"Class folders differ between train and test:\n"
            f"  only in train: {sorted(set(class_names) - set(test_classes))}\n"
            f"  only in test:  {sorted(set(test_classes) - set(class_names))}"
        )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # 收集所有圖片：(來源 split, 類別, 路徑)，並依內容 hash 分組
    by_hash = defaultdict(list)
    for split, root in (("test", args.test_dir), ("train", args.train_dir)):
        for name in class_names:
            for path in list_images(Path(root) / name):
                by_hash[file_md5(path)].append((split, name, path))

    train_pool = defaultdict(list)
    test_samples = []
    conflicts, leaks, dups = [], [], []
    for copies in by_hash.values():
        labels = {name for _, name, _ in copies}
        if len(labels) > 1:
            conflicts.append(copies)
            continue
        # test 優先保留（test 排在前面），其餘副本都丟掉
        split, name, path = copies[0]
        if len(copies) > 1:
            splits = {s for s, _, _ in copies}
            (leaks if splits == {"train", "test"} else dups).append(copies)
        if split == "test":
            test_samples.append((path, class_to_idx[name]))
        else:
            train_pool[name].append(path)

    # 分層抽樣切出 val：每個類別各自抽 val_ratio 比例（至少 1 張），讓 val 的類別分布跟 train 一致
    rng = random.Random(args.seed)
    train_samples, val_samples = [], []
    for name in class_names:
        paths = sorted(train_pool[name])
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * args.val_ratio)) if len(paths) > 1 else 0
        val_samples += [(p, class_to_idx[name]) for p in paths[:n_val]]
        train_samples += [(p, class_to_idx[name]) for p in paths[n_val:]]

    def rel(samples):
        return [[Path(p).resolve().relative_to(PROJECT_ROOT).as_posix(), label] for p, label in sorted(samples)]

    output = {
        "class_names": class_names,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "splits": {"train": rel(train_samples), "val": rel(val_samples), "test": rel(test_samples)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)

    # ---------- 報告 ----------
    def show(title, groups):
        if groups:
            print(f"\n{title} ({len(groups)}):")
            for copies in groups:
                print("   " + "  ==  ".join(Path(p).relative_to(PROJECT_ROOT).as_posix() for _, _, p in copies))

    show("[WARN] Same image under different classes - excluded", conflicts)
    show("[WARN] Same image in train and test - kept only in test", leaks)
    show("[INFO] Duplicate images within a split - kept one copy", dups)

    counts = {s: defaultdict(int) for s in ("train", "val", "test")}
    for s, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        for _, label in samples:
            counts[s][label] += 1
    print(f"\n{'class':<24}{'train':>7}{'val':>6}{'test':>6}")
    for idx, name in enumerate(class_names):
        print(f"{name:<24}{counts['train'][idx]:>7}{counts['val'][idx]:>6}{counts['test'][idx]:>6}")
    print(f"{'TOTAL':<24}{len(train_samples):>7}{len(val_samples):>6}{len(test_samples):>6}")
    print(f"\nSplit saved to: {args.output}")


if __name__ == "__main__":
    main()
