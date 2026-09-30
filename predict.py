import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不開視窗畫圖，直接存成圖片檔
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import PROJECT_ROOT, build_transform, get_class_names, list_images
from model import load_checkpoint


def find_latest_checkpoint():
    """預設使用最新一次訓練 (runs/<時間>/best.pth) 的權重；沒有的話退回舊版的 weights/best.pth。"""
    runs = sorted((PROJECT_ROOT / "runs").glob("*/best.pth"))
    if runs:
        return runs[-1]
    legacy = PROJECT_ROOT / "weights" / "best.pth"
    return legacy if legacy.exists() else None


class Predictor:
    """
    包裝「載入模型 + 前處理 + 預測」。只在建立時載入一次模型，之後可以重複呼叫 predict()，
    批次推論時不必每張圖都重新載入權重。
    """

    def __init__(self, model_path=None, class_dir=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_path = Path(model_path) if model_path else find_latest_checkpoint()
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Train a model first with train.py.")

        # 新版權重檔自帶類別名稱與圖片大小；舊版權重檔才需要用 class_dir 取得類別名稱
        legacy_classes = get_class_names(class_dir) if class_dir and Path(class_dir).is_dir() else None
        self.model, self.class_names, image_size = load_checkpoint(model_path, self.device, legacy_classes)
        self.model_path = model_path

        # 前處理跟訓練時的「驗證/測試」前處理共用同一個函式，保證完全一致
        self.transform = build_transform(image_size, train=False)

    @torch.no_grad()
    def predict(self, images, batch_size=32):
        """輸入 PIL 圖片列表，回傳 [(類別名稱, 信心分數), ...]。"""
        results = []
        for start in range(0, len(images), batch_size):
            x = torch.stack([self.transform(img) for img in images[start:start + batch_size]]).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)  # 把分數轉成加總為 100% 的機率，比較直覺
            scores, indices = probs.max(dim=1)           # 機率最高的那個類別與它的機率
            results += [(self.class_names[i], s) for i, s in zip(indices.tolist(), scores.tolist())]
        return results


def load_image(path):
    return Image.open(path).convert("RGB")


def save_visualization(image, title, save_path, color="black"):
    """把原圖跟預測結果一起畫出來、存檔，方便肉眼檢查模型猜得對不對。"""
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image)
    ax.set_title(title, color=color)
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # 關掉圖表釋放記憶體，批次處理大量圖片時很重要


def main():
    parser = argparse.ArgumentParser(description="Classify an image using the trained model.")
    parser.add_argument("--image", type=Path, default=None,
                        help="Image file to classify (default: first image under data/test)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Checkpoint file (default: latest runs/*/best.pth)")
    parser.add_argument("--class-dir", type=Path, default=PROJECT_ROOT / "data" / "train",
                        help="Class folders; only needed for legacy checkpoints without class names")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "result_output.jpg", help="Output image path")
    args = parser.parse_args()

    # 沒指定 --image 的話，就從 data/test 挑第一張圖片來用
    image_path = args.image
    if image_path is None:
        candidates = list_images(PROJECT_ROOT / "data" / "test")
        if not candidates:
            raise FileNotFoundError("No image was provided and no sample image was found under data/test.")
        image_path = candidates[0]

    predictor = Predictor(args.model, args.class_dir)
    image = load_image(image_path)
    label, score = predictor.predict([image])[0]

    save_visualization(image, f"Predicted: {label} ({score:.2%})", args.output)
    print(f"Model: {predictor.model_path}")
    print(f"Image: {image_path}")
    print(f"Prediction: {label} ({score:.2%})")
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
