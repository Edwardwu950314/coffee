"""
從 test 資料夾（各類別資料夾）隨機抽幾張圖片做預測，並把結果圖存起來。
用途：訓練完後想快速抽查模型表現，不用一張一張手動跑 predict.py。
因為圖片放在類別資料夾裡，資料夾名稱就是正確答案，所以也會順便統計猜對幾張。

用法：
    python batch_predict.py                 # 隨機抽 5 張
    python batch_predict.py --num-images 20
    python batch_predict.py --all           # 全部圖片
"""
import argparse
import random
from pathlib import Path

from dataset import PROJECT_ROOT, list_images
from predict import Predictor, load_image, save_visualization


def main():
    parser = argparse.ArgumentParser(description="Run the classifier on sample images from class folders.")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "test_results")
    parser.add_argument("--model", type=Path, default=None, help="Checkpoint file (default: latest runs/*/best.pth)")
    parser.add_argument("--class-dir", type=Path, default=PROJECT_ROOT / "data" / "train",
                        help="Class folders; only needed for legacy checkpoints without class names")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Predict every image instead of a random sample")
    parser.add_argument("--per-class", type=int, default=None,
                        help="Sample N images from every class folder (overrides --num-images)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    all_images = list_images(args.test_dir)
    if not all_images:
        print(f"No images found under {args.test_dir}")
        return

    # random.sample 不會抽到重複的圖片；min() 是避免要求的張數比實際圖片還多時噴錯
    if args.all:
        selected = all_images
    elif args.per_class:
        rng = random.Random(args.seed)
        by_class = {}
        for p in all_images:
            by_class.setdefault(p.parent.name, []).append(p)
        selected = [p for imgs in by_class.values() for p in rng.sample(imgs, min(args.per_class, len(imgs)))]
    else:
        # 預設：固定有 2~3 張 Normal，其餘從瑕疵類別抽，再打散順序
        rng = random.Random(args.seed)
        normal = [p for p in all_images if p.parent.name == "Normal"]
        others = [p for p in all_images if p.parent.name != "Normal"]
        n_normal = min(rng.randint(2, 3), args.num_images, len(normal))
        selected = rng.sample(normal, n_normal) + rng.sample(others, min(args.num_images - n_normal, len(others)))
        rng.shuffle(selected)

    # 模型只載入一次，所有圖片一起批次預測
    predictor = Predictor(args.model, args.class_dir)
    print(f"Model: {predictor.model_path}")
    binary = set(predictor.class_names) == {"bad", "good"}
    images = [load_image(p) for p in selected]
    predictions = predictor.predict(images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    correct = 0
    for idx, (path, image, (label, score)) in enumerate(zip(selected, images, predictions), 1):
        actual = path.parent.name  # 圖片所在的資料夾名稱就是正確答案
        # 二元模型（bad/good）：Normal 資料夾算 good，其餘瑕疵類別都算 bad
        expected = ("good" if actual == "Normal" else "bad") if binary else actual
        ok = label == expected
        correct += ok
        mark = "OK " if ok else "BAD"
        print(f"[{idx}/{len(selected)}] {mark} {path.name}: predicted={label} ({score:.2%}) actual={actual}")

        output_path = args.output_dir / f"{path.stem}_pred.jpg"  # 原檔名加上 _pred 後綴，避免蓋掉原圖
        title = f"Predicted: {label} ({score:.2%})\nActual: {actual}"
        save_visualization(image, title, output_path, color="green" if ok else "red")

    print(f"\nAccuracy on this sample: {correct}/{len(selected)} = {correct / len(selected):.2%}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
