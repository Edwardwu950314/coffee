import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不開視窗，直接存檔
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import PROJECT_ROOT, build_transform, list_images
from model import load_checkpoint
from prepare_dataset import crop_bean

METHODS = ("water", "sun", "honey")  # 水洗 / 日曬 / 蜜處理
CROP_SIZE = 256  # 要與 prepare_dataset.py 的 --size 相同


def model_path_for(method):
    """各處理法的模型固定放在 models/<method>/best.pth。"""
    return PROJECT_ROOT / "models" / method / "best.pth"


class Predictor:
    """依處理法載入對應模型。切換處理法就是換一個 Predictor（或呼叫 Predictor(method)）。"""

    def __init__(self, method="water", model_path=None, device=None):
        self.method = method
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(model_path) if model_path else model_path_for(method)
        if not path.exists():
            raise FileNotFoundError(f"No model for '{method}' at {path}. Train it with: python train.py --method {method}")
        self.model, self.class_names, image_size = load_checkpoint(path, self.device)
        self.model_path = path
        self.transform = build_transform(image_size, train=False)

    @torch.no_grad()
    def predict(self, images, crop=True, batch_size=32):
        """輸入 PIL 圖片列表，回傳 [(good/bad, 機率), ...]。crop=True 代表輸入是整張原照片，會先裁出豆子。"""
        if crop:
            images = [crop_bean(img.convert("RGB"), CROP_SIZE) for img in images]
        results = []
        for start in range(0, len(images), batch_size):
            x = torch.stack([self.transform(img.convert("RGB")) for img in images[start:start + batch_size]]).to(self.device)
            scores, indices = torch.softmax(self.model(x), dim=1).max(dim=1)
            results += [(self.class_names[i], s) for i, s in zip(indices.tolist(), scores.tolist())]
        return results


def save_result(image, label, score, save_path):
    """把裁出的豆子連同預測結果畫成圖存檔：好豆綠字、壞豆紅字。"""
    fig, ax = plt.subplots(figsize=(4, 4.4))
    ax.imshow(image)
    ax.set_title(f"{label} ({score:.1%})", color="green" if label == "good" else "red", fontsize=16)
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Classify a green coffee bean as good / bad.")
    parser.add_argument("--method", choices=METHODS, default="water", help="Processing method of the bean")
    parser.add_argument("--image", type=Path, nargs="+", default=None,
                        help="Image file(s) or folder (default: 5 random test-set images of the method, with results saved)")
    parser.add_argument("--model", type=Path, default=None, help="Override checkpoint path")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Save an annotated image (cropped bean + prediction) per input into this folder")
    parser.add_argument("--no-crop", action="store_true", help="Input is already a cropped bean image")
    args = parser.parse_args()

    paths = []
    if args.image is None:
        # 沒指定圖片：從該處理法的測試集抽 5 張示範，並自動存圖
        split_file = PROJECT_ROOT / "splits" / f"{args.method}.json"
        if not split_file.exists():
            parser.error(f"--image is required ({split_file} not found)")
        with open(split_file, encoding="utf-8") as f:
            test = json.load(f)["splits"]["test"]
        paths = [PROJECT_ROOT / p for p, _ in random.sample(test, min(5, len(test)))]
        args.no_crop = True  # cache 裡已是裁好的圖
        args.save_dir = args.save_dir or PROJECT_ROOT / "predict_results"
    for p in args.image or []:
        paths += list_images(p) if p.is_dir() else [p]
    predictor = Predictor(args.method, args.model)
    print(f"Model: {predictor.model_path}")
    images = [Image.open(p).convert("RGB") for p in paths]
    if not args.no_crop:
        images = [crop_bean(img, CROP_SIZE) for img in images]
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    for path, image, (label, score) in zip(paths, images, predictor.predict(images, crop=False)):
        print(f"{path.name}: {label} ({score:.1%})")
        if args.save_dir:
            save_result(image, label, score, args.save_dir / f"{path.stem}_{label}.jpg")
    if args.save_dir:
        print(f"Annotated images saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
