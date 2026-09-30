"""
離線資料擴增：只針對「訓練集」的圖片產生擴增版本，存到 data/train_aug/<類別>/，
並把它們加進 data/splits.json 的 train。val / test 完全不動，避免同一張圖的變體洩漏到評估資料。

做的事情：
1. 第一次執行會把原本的切分檔備份成 data/splits_base.json，之後每次都從備份重新產生，
   所以重複執行不會越擴越多。
2. 每個類別補到 --target 張（已經超過的類別不擴增），順便緩和類別不平衡。
3. 擴增手法：隨機翻轉、旋轉、裁切縮放、亮度/對比/色調、輕微模糊、雜訊。

用法：
    python augment.py
    python augment.py --target 120 --seed 0
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFilter
from torchvision import transforms

from dataset import PROJECT_ROOT

AUG_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=180, fill=0),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
])


def augment_image(image, rng):
    out = AUG_TRANSFORM(image)
    # 裁切後縮回原尺寸（Normal 是 200x200，其他是 500x500，不能統一改成同一大小）
    out = transforms.RandomResizedCrop(image.size[::-1], scale=(0.75, 1.0), ratio=(0.9, 1.1))(out)
    if rng.random() < 0.3:  # 模擬對焦不準
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Offline augmentation of the training split.")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "splits.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "train_aug")
    parser.add_argument("--target", type=int, default=100, help="Images per class after augmentation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_path = args.split.with_name(args.split.stem + "_base.json")
    if not base_path.exists():
        shutil.copy(args.split, base_path)
        print(f"Original split backed up to {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    rng = random.Random(args.seed)
    random.seed(args.seed)  # torchvision 的隨機變換用全域 random
    by_class = defaultdict(list)
    for path, label in data["splits"]["train"]:
        by_class[label].append(path)

    new_samples = []
    for label, paths in sorted(by_class.items()):
        name = data["class_names"][label]
        folder = args.out_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(max(0, args.target - len(paths))):
            src = paths[i % len(paths)]
            with Image.open(PROJECT_ROOT / src) as im:
                aug = augment_image(im.convert("RGB"), rng)
            out_path = folder / f"{name}_aug{i:03d}.jpg"
            aug.save(out_path, quality=95)
            new_samples.append([out_path.relative_to(PROJECT_ROOT).as_posix(), label])

    data["splits"]["train"] = sorted(data["splits"]["train"] + new_samples)
    data["augmented"] = {"target": args.target, "seed": args.seed, "added": len(new_samples)}
    with open(args.split, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    print(f"Added {len(new_samples)} augmented images -> train = {len(data['splits']['train'])}")
    print(f"val={len(data['splits']['val'])}  test={len(data['splits']['test'])} (unchanged)")


if __name__ == "__main__":
    main()
