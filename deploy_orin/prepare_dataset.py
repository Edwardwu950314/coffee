"""
資料前處理：把 coffee/<method>/{good,bad}/ 的原圖裁成「單顆豆子」的方形小圖，並切成 train / val / test。

原圖是 4096x3072、豆子只佔畫面一小塊，直接縮成 224 會把瑕疵縮到看不見，所以先偵測豆子位置再裁切。
切分方式：以「類別 x 拍攝日期」為單位分層抽樣，因此
  - train / val / test 的 good:bad 比例都維持與整體相同（約 50:50）
  - 每一天拍的照片都會按比例分進三份，不會某一天全進 train 或 test

輸出：
  cache/<method>/<class>/*.jpg   裁好的圖（不進版本控制，可隨時重新產生）
  splits/<method>.json           切分結果（進版本控制）
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from dataset import PROJECT_ROOT, list_images

CLASSES = ["bad", "good"]
DATE_PATTERN = re.compile(r"(\d{8})")  # 檔名 P_20261007_195415.jpg 裡的拍攝日期


def largest_component(mask):
    """回傳 mask 中最大連通區域（8 鄰域）的 (xs, ys)。只走訪前景像素，速度很快。"""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue, comp = deque([(y0, x0)]), []
        while queue:
            y, x = queue.popleft()
            comp.append((y, x))
            for ny in range(max(y - 1, 0), min(y + 2, h)):
                for nx in range(max(x - 1, 0), min(x + 2, w)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    arr = np.array(best, dtype=int).reshape(-1, 2)
    return arr[:, 1], arr[:, 0]


def find_bean_box(img):
    """在縮小的圖上找豆子：背景是單色紙，與背景顏色差異大的區域就是豆子（含一點陰影）。回傳 (l, t, r, b) 與縮小圖尺寸。"""
    small = img.copy()
    small.thumbnail((512, 512))
    arr = np.asarray(small, dtype=np.float32)
    background = np.median(arr.reshape(-1, 3), axis=0)  # 豆子只佔一小塊，整張圖的中位數顏色就是背景色
    diff = np.sqrt(((arr - background) ** 2).sum(axis=2))
    mask = Image.fromarray(((diff > max(20.0, diff.mean() + 4 * diff.std())) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MedianFilter(5))  # 去掉雜點
    xs, ys = largest_component(np.asarray(mask) > 0)  # 只取最大的一塊，邊緣雜點不會拉歪裁切框
    if len(xs) < 30:
        return None, small.size
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), small.size


def crop_bean(img, out_size, margin=1.5):
    """以豆子為中心裁出正方形（邊長 = 豆子長邊 x margin），找不到豆子就取畫面中央。"""
    box, small_size = find_bean_box(img)
    w, h = img.size
    if box is None:
        cx, cy, side = w / 2, h / 2, min(w, h) * 0.5
    else:
        scale = w / small_size[0]
        left, top, right, bottom = (v * scale for v in box)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        side = max(right - left, bottom - top) * margin
    side = min(max(side, out_size), min(w, h))  # 至少 out_size，且不超出原圖
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    crop = img.crop((int(left), int(top), int(left + side), int(top + side)))
    return crop.resize((out_size, out_size), Image.LANCZOS)


def stratified_split(items, ratios, seed):
    """items: [(path, class, date)]。每個 (class, date) 組合各自依 ratios 切，再合併。"""
    groups = defaultdict(list)
    for item in items:
        groups[(item[1], item[2])].append(item)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        group = sorted(groups[key])
        rng.shuffle(group)
        n_val = round(len(group) * ratios[1])
        n_test = round(len(group) * ratios[2])
        splits["test"] += group[:n_test]
        splits["val"] += group[n_test:n_test + n_val]
        splits["train"] += group[n_test + n_val:]
    return splits


def main():
    parser = argparse.ArgumentParser(description="Crop beans and split a processing-method dataset.")
    parser.add_argument("--method", default="water", help="Folder name under coffee/ (water / sun / honey)")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "coffee")
    parser.add_argument("--size", type=int, default=256, help="Size of the cropped square images")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = args.raw_dir / args.method
    cache = PROJECT_ROOT / "cache" / args.method
    items = []
    for cls in CLASSES:
        files = list_images(raw / cls)
        if not files:
            raise FileNotFoundError(f"No images found in {raw / cls}")
        for path in files:
            match = DATE_PATTERN.search(path.name)
            items.append((path, cls, match.group(1) if match else "unknown"))
    print(f"{args.method}: {Counter(i[1] for i in items)}  dates: {dict(Counter(i[2] for i in items))}")

    # 1. 裁切（已存在就跳過，方便中斷後續跑）
    for n, (path, cls, _) in enumerate(items, 1):
        out = cache / cls / (path.stem + ".jpg")
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            crop_bean(img.convert("RGB"), args.size).save(out, quality=95)
        if n % 100 == 0:
            print(f"  cropped {n}/{len(items)}")

    # 2. 分層切分
    cache_items = [(cache / cls / (p.stem + ".jpg"), cls, date) for p, cls, date in items]
    splits = stratified_split(cache_items, args.ratios, args.seed)
    label_of = {c: i for i, c in enumerate(CLASSES)}
    out_json = PROJECT_ROOT / "splits" / f"{args.method}.json"
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "method": args.method,
            "class_names": CLASSES,
            "seed": args.seed,
            "splits": {name: [[p.relative_to(PROJECT_ROOT).as_posix(), label_of[c]] for p, c, _ in sorted(rows)]
                       for name, rows in splits.items()},
        }, f, indent=1, ensure_ascii=False)

    print(f"\n{'split':<7}{'total':>6}  {'bad':>5}{'good':>6}   by date")
    for name, rows in splits.items():
        by_class = Counter(c for _, c, _ in rows)
        by_date = Counter(d for _, _, d in rows)
        print(f"{name:<7}{len(rows):>6}  {by_class['bad']:>5}{by_class['good']:>6}   {dict(sorted(by_date.items()))}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
