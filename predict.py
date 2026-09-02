

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="用YOLOv8-OBB模型辨識咖啡豆瑕疵")
    parser.add_argument(
        "--weights", type=str, required=True,
        help="訓練好的模型權重路徑，例如 runs/obb/coffee_bean_defect/weights/best.pt"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="要辨識的圖片路徑、資料夾路徑，或影片路徑"
    )
    parser.add_argument("--imgsz", type=int, default=224, help="推論時的輸入影像大小")
    parser.add_argument("--conf", type=float, default=0.25, help="信心閾值(0~1)，越高越嚴格")
    parser.add_argument("--device", type=str, default="", help="'0'=GPU0, 'cpu'=CPU，留空自動偵測")
    parser.add_argument(
        "--project", type=str, default="runs/obb",
        help="輸出結果的上層資料夾"
    )
    parser.add_argument(
        "--name", type=str, default="predict",
        help="本次推論結果的資料夾名稱"
    )
    parser.add_argument(
        "--csv", type=str, default="",
        help="要輸出的統計結果CSV檔路徑 (預設會存在輸出資料夾內的 summary.csv)"
    )
    return parser.parse_args()


def summarize(results, save_dir: Path, csv_path: str):
    """整理每張圖片的偵測結果，印出報表並存成CSV"""
    rows = []
    ng_image_count = 0

    for r in results:
        img_path = Path(r.path)
        # OBB 任務的偵測結果存在 r.obb，一般偵測任務則是 r.boxes
        det = r.obb if r.obb is not None else r.boxes
        num_defects = 0 if det is None else len(det)
        verdict = "NG (瑕疵)" if num_defects > 0 else "OK (正常)"
        if num_defects > 0:
            ng_image_count += 1

        avg_conf = 0.0
        if det is not None and len(det) > 0:
            avg_conf = float(det.conf.mean())

        rows.append({
            "image": img_path.name,
            "defect_count": num_defects,
            "avg_confidence": round(avg_conf, 4),
            "verdict": verdict,
        })

    # 印出報表
    print("\n" + "=" * 60)
    print(f"{'影像':40s} {'瑕疵數':>6s} {'平均信心':>8s}  判定")
    print("-" * 60)
    for row in rows:
        print(f"{row['image']:40s} {row['defect_count']:>6d} "
                f"{row['avg_confidence']:>8.3f}  {row['verdict']}")
    print("-" * 60)
    total = len(rows)
    print(f"總圖片數: {total}，判定為瑕疵(NG)的圖片: {ng_image_count}，"
            f"正常(OK): {total - ng_image_count}")
    print("=" * 60)

    # 存成 CSV
    out_csv = Path(csv_path) if csv_path else save_dir / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "defect_count", "avg_confidence", "verdict"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] 統計結果已存成 CSV: {out_csv}")


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到模型權重: {weights_path}")

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"找不到辨識來源: {source_path}")

    print(f"[INFO] 載入模型: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] 開始辨識: {source_path}")
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device or None,
        save=True,           # 儲存標好框的圖片
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(results[0].save_dir) if results else Path(args.project) / args.name
    print(f"[INFO] 標註後的圖片已存到: {save_dir}")

    summarize(results, save_dir, args.csv)


if __name__ == "__main__":
    main()
