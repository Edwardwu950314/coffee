import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="訓練 YOLOv8-OBB 咖啡豆瑕疵偵測模型")

    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s-obb.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project", type=str, default="runs/obb")
    parser.add_argument("--name", type=str, default="coffee_bean_defect")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "False"])

    parser.add_argument("--optimizer", type=str, default="auto",
                         choices=["auto", "SGD", "Adam", "AdamW"])
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--cos_lr", action="store_true", default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--degrees", type=float, default=180.0)
    parser.add_argument("--flipud", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close_mosaic", type=int, default=15)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--erasing", type=float, default=0.4)

    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--plots", action="store_true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到資料集設定檔: {data_path}")

    if args.resume:
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"--resume 但找不到 checkpoint: {last_ckpt}")
        print(f"[INFO] 從中斷點接續訓練: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"[INFO] 載入模型: {args.model}")
        model = YOLO(args.model)

    print(f"[INFO] 開始訓練，資料集設定: {data_path}")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        seed=args.seed,
        workers=args.workers,
        cache=False if args.cache == "False" else args.cache,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        label_smoothing=args.label_smoothing,
        degrees=args.degrees,
        flipud=args.flipud,
        fliplr=args.fliplr,
        scale=args.scale,
        translate=args.translate,
        shear=args.shear,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        erasing=args.erasing,
        multi_scale=args.multi_scale,
        amp=args.amp,
        plots=args.plots,
        exist_ok=args.resume,
    )

    print("[INFO] 訓練完成，開始在驗證集上評估...")
    metrics = model.val(data=str(data_path), imgsz=args.imgsz, plots=True)
    print(metrics)

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[INFO] 最佳模型權重路徑: {best_weights}")
    print(f"[INFO] 訓練圖表存於: {Path(args.project) / args.name}")
    print("[INFO] 之後可用 predict.py 搭配這個權重進行辨識，例如:")
    print(f"    python predict.py --weights {best_weights} --source path/to/image_or_folder")


if __name__ == "__main__":
    main()