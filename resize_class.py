"""
把指定類別的圖片放大成統一尺寸（預設 Normal -> 500x500），不修改原始圖片。

放大後的圖存到 data/resized/<原路徑>，並把 data/splits_base.json 與 data/splits.json
裡該類別的路徑改指向新圖。做完後請重新執行 `python augment.py`，
讓擴增圖也從 500x500 的版本產生。

用法：
    python resize_class.py
    python resize_class.py --cls Normal --size 500
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from dataset import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Resize one class's images to a fixed size.")
    parser.add_argument("--cls", default="Normal")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "resized")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    for name in ("splits_base.json", "splits.json"):
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        label = data["class_names"].index(args.cls)

        n = 0
        for samples in data["splits"].values():
            for item in samples:
                src = item[0]
                if item[1] != label or src.startswith("data/resized/"):
                    continue
                if src.startswith("data/train_aug/"):
                    continue  # 擴增圖由 augment.py 重新產生
                dst = args.out_dir / Path(src).relative_to("data")
                dst.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(PROJECT_ROOT / src) as im:
                    im.convert("RGB").resize((args.size, args.size), Image.BICUBIC).save(dst, quality=95)
                item[0] = dst.relative_to(PROJECT_ROOT).as_posix()
                n += 1

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"{name}: {n} '{args.cls}' images resized to {args.size}x{args.size}")


if __name__ == "__main__":
    main()
