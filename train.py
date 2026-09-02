"""
train.py
========
使用 Ultralytics YOLOv8 (OBB 旋轉框任務) 訓練咖啡豆瑕疵偵測模型。

背景說明
--------
你的資料集標註格式為每列 9 個數字:
    class  x1 y1  x2 y2  x3 y3  x4 y4   (皆為 0~1 正規化座標)
這是「4 個角點」的旋轉框 (Oriented Bounding Box, OBB) 格式，
因此必須使用 YOLOv8-OBB 模型 (yolov8*-obb.pt)，而不是一般的偵測模型，
否則 Ultralytics 會因為欄位數量對不上而報錯或訓練效果不佳。

資料集只有 1 個類別: "NG" (瑕疵)。也就是說模型的任務是找出
咖啡豆影像中「瑕疵區域 / 瑕疵豆」的位置。

使用方式
--------
1. 安裝套件:
    pip install ultralytics

2. 把本資料夾與你解壓縮出來的 train/ valid/ test/ 資料夾放在一起，
    確保目錄結構長這樣:

    coffee_bean_yolov8/
    ├── data.yaml
    ├── train.py
    ├── predict.py
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

3. 執行訓練 (預設會自動下載 yolov8n-obb.pt 預訓練權重):
    python train.py

    也可以自行調整參數，例如:
    python train.py --model yolov8s-obb.pt --epochs 150 --imgsz 224 --batch 32

訓練完成後，最佳權重會存放在:
    runs/obb/coffee_bean_defect/weights/best.pt
之後可以把這個路徑丟給 predict.py 做推論。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="訓練 YOLOv8-OBB 咖啡豆瑕疵偵測模型")
    parser.add_argument(
        "--data", type=str, default="data.yaml",
        help="資料集設定檔路徑 (預設: data.yaml)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n-obb.pt",
        help="預訓練權重/模型結構檔，可選 yolov8n-obb.pt / yolov8s-obb.pt / "
            "yolov8m-obb.pt / yolov8l-obb.pt / yolov8x-obb.pt (n最快, x最準)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="訓練回合數")
    parser.add_argument("--imgsz", type=int, default=224, help="輸入影像大小 (資料集原始為224)")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument(
        "--device", type=str, default="",
        help="訓練裝置，例如 '0' 表示第一張GPU，'cpu' 表示用CPU，留空自動偵測"
    )
    parser.add_argument("--patience", type=int, default=30, help="Early stopping 的耐心值")
    parser.add_argument(
        "--project", type=str, default="runs/obb",
        help="訓練結果輸出的上層資料夾"
    )
    parser.add_argument(
        "--name", type=str, default="coffee_bean_defect",
        help="本次訓練的實驗名稱 (結果存於 project/name 底下)"
    )
    parser.add_argument("--resume", action="store_true", help="從上次中斷的訓練繼續")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"找不到資料集設定檔: {data_path}\n"
            f"請確認 data.yaml 與 train/valid/test 資料夾放在正確位置。"
        )

    print(f"[INFO] 載入模型: {args.model}")
    model = YOLO(args.model)  # 使用 *-obb.pt 會自動以 OBB 任務初始化

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
    )

    print("[INFO] 訓練完成，開始在驗證集上評估...")
    metrics = model.val(data=str(data_path), imgsz=args.imgsz)
    print(metrics)

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[INFO] 最佳模型權重路徑: {best_weights}")
    print("[INFO] 之後可用 predict.py 搭配這個權重進行辨識，例如:")
    print(f"    python predict.py --weights {best_weights} --source path/to/image_or_folder")


if __name__ == "__main__":
    main()
