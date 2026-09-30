import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不開視窗畫圖，直接存成圖片檔（伺服器/無畫面環境也能跑）
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PROJECT_ROOT, ImageListDataset, build_transform, load_split
from model import CoffeeBeanClassifier, save_checkpoint


def set_seed(seed):
    """固定所有亂數來源，讓同樣的設定每次訓練出來的結果都一樣，實驗才能重現、比較。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_confusion_matrix(preds, labels, num_classes):
    """
    混淆矩陣：一張表格，橫軸是「模型猜的類別」，縱軸是「真正的類別」。
    cm[真實類別, 預測類別] += 1，可以一眼看出模型「常常把哪一類誤判成哪一類」。
    對角線上的數字 = 猜對的數量；對角線以外的數字 = 猜錯、且錯成哪一類。
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels), np.asarray(preds)), 1)
    return cm


def metrics_from_cm(cm):
    """
    從混淆矩陣算出各項指標：
    - precision（精確率）：模型說是 A 的圖裡，真的是 A 的比例
    - recall（召回率 / 每類準確率）：真的是 A 的圖裡，被模型找出來的比例
    - F1：precision 與 recall 的調和平均
    - macro-F1：每個類別的 F1 直接平均，每類權重相同，
      不會因為某類圖片特別多就掩蓋少數類別的表現，比 accuracy 更適合類別不平衡的資料。
    用 np.maximum(..., 1) 避免分母為 0。
    """
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": tp.sum() / max(cm.sum(), 1),
        "macro_f1": f1.mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_epoch(model, loader, criterion, device, num_classes, optimizer=None, scaler=None):
    """
    跑完一個 epoch。有傳 optimizer 就是「訓練」（會更新參數），沒傳就是「評估」（只考試，不學習）。
    回傳 (平均 loss, 混淆矩陣)。
    """
    training = optimizer is not None
    model.train(training)  # 切換訓練/評估模式（會影響 BatchNorm 等層的行為）
    total_loss, all_preds, all_labels = 0.0, [], []

    # 評估時不需要計算梯度，省記憶體、跑更快
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # autocast：在 GPU 上自動用半精度 (float16) 運算，速度更快、更省顯示卡記憶體
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)             # 1. 模型猜答案
                loss = criterion(logits, labels)   # 2. 算猜錯多少

            if training:
                optimizer.zero_grad()              # 清空上一批留下的梯度，避免累加錯誤
                if scaler is not None:
                    scaler.scale(loss).backward()  # 3. 反向傳播，算出每個參數該往哪個方向調整
                    scaler.step(optimizer)         # 4. 真的去調整參數
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())  # 分數最高的類別 = 模型的預測結果
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(labels), compute_confusion_matrix(preds, labels, num_classes)


def plot_confusion_matrix(cm, class_names, save_path, title):
    """把混淆矩陣畫成熱力圖存檔，顏色越深代表數量越多，格子裡標上實際數字。"""
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_curves(history, save_path):
    """畫出 loss 曲線與驗證指標曲線。"""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="royalblue")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", color="darkorange")
    ax1.set_title("Loss")
    ax2.plot(epochs, [h["train_acc"] for h in history], label="Train Accuracy", color="royalblue")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="Val Accuracy", color="darkorange")
    ax2.plot(epochs, [h["val_macro_f1"] for h in history], label="Val Macro-F1", color="forestgreen")
    ax2.set_title("Validation Metrics")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def write_report(path, class_names, results):
    """把每個資料集（val/test）的整體指標與每類 precision/recall/F1 寫成文字報告。"""
    with open(path, "w", encoding="utf-8") as f:
        for split, (loss, cm) in results.items():
            m = metrics_from_cm(cm)
            f.write(f"===== {split} =====\n")
            f.write(f"loss={loss:.4f}  accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}\n\n")
            f.write(f"{'class':<24}{'precision':>10}{'recall':>10}{'f1':>8}{'support':>9}\n")
            for i, name in enumerate(class_names):
                f.write(f"{name:<24}{m['precision'][i]:>10.4f}{m['recall'][i]:>10.4f}"
                        f"{m['f1'][i]:>8.4f}{cm[i].sum():>9}\n")
            f.write("\n")


def main():
    # 用命令列參數讓使用者可以不改程式碼就調整訓練設定，例如：
    # python train.py --epochs 30 --lr 5e-5
    parser = argparse.ArgumentParser(description="Train a coffee bean defect classifier.")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data" / "splits.json",
                        help="Split file created by make_split.py")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output folder (default: runs/<timestamp>)")
    parser.add_argument("--epochs", type=int, default=30)       # 整份訓練資料最多重複看幾遍
    parser.add_argument("--patience", type=int, default=8)      # 驗證分數連續幾個 epoch 沒進步就提早停止
    parser.add_argument("--batch-size", type=int, default=16)   # 一次丟幾張圖片進模型
    parser.add_argument("--lr", type=float, default=1e-4)       # learning rate：每次調整參數的步伐大小
    parser.add_argument("--img-size", type=int, default=224)    # 圖片統一縮放成幾 x 幾
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "efficientnet_b0"])  # backbone 架構
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-balanced loss")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on GPU")
    parser.add_argument("--device", type=str, default="auto")   # 用 CPU 還是 GPU 訓練
    args = parser.parse_args()

    set_seed(args.seed)

    # 自動判斷有沒有可用的 GPU（cuda），有的話用 GPU 訓練會快非常多
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.split.exists():
        raise FileNotFoundError(f"{args.split} not found. Run `python make_split.py` first.")
    class_names, splits = load_split(args.split)
    num_classes = len(class_names)

    # 每次訓練都存到獨立的資料夾，不會覆蓋上一次的結果，方便比較不同實驗
    out_dir = args.out_dir or PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    print(f"Training on {num_classes} classes using {device}")
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"Outputs: {out_dir}")

    # 建立訓練集（會做資料增強）跟驗證/測試集（不做資料增強，維持穩定評估標準）
    train_dataset = ImageListDataset(splits["train"], build_transform(args.img_size, train=True))
    eval_transform = build_transform(args.img_size, train=False)
    val_dataset = ImageListDataset(splits["val"], eval_transform)
    test_dataset = ImageListDataset(splits["test"], eval_transform)

    # DataLoader 負責把 dataset 包裝成「一批一批」丟給模型，
    # shuffle=True 讓訓練集每個 epoch 的順序都打亂，避免模型記住資料的排列順序
    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # 建立模型，並搬到指定的裝置（CPU 或 GPU）上
    model = CoffeeBeanClassifier(num_classes=num_classes, arch=args.arch).to(device)

    # loss 函式：CrossEntropyLoss 是分類任務最常用的損失函數。
    # 類別權重：圖片越少的類別權重越高，避免模型只顧著猜「圖片多的類別」。
    class_weights = None
    if not args.no_class_weights:
        counts = np.bincount(train_dataset.targets(), minlength=num_classes).astype(float)
        weights = counts.sum() / (num_classes * np.maximum(counts, 1))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # optimizer：根據 loss.backward() 算出來的梯度，實際去調整模型參數
    # weight_decay 是一種讓參數不要長得太誇張的正則化手法，避免 overfitting
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # scheduler：如果驗證 loss 連續 3 個 epoch 都沒進步，就自動把 learning rate 減半，
    # 讓模型在快收斂時用更小的步伐微調，避免一直在最佳解附近震盪。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 混合精度只在 GPU 上啟用
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and not args.no_amp else None

    best_f1, best_loss, best_epoch = -1.0, float("inf"), 0
    history = []
    history_path = out_dir / "history.csv"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_cm = run_epoch(model, train_loader, criterion, device, num_classes, optimizer, scaler)
        val_loss, val_cm = run_epoch(model, val_loader, criterion, device, num_classes, scaler=scaler)
        scheduler.step(val_loss)  # 把這次驗證 loss 回報給 scheduler，讓它決定要不要降 learning rate

        train_m, val_m = metrics_from_cm(train_cm), metrics_from_cm(val_cm)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_m["accuracy"],
            "val_loss": val_loss,
            "val_acc": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
        }
        history.append(row)
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_m['accuracy']:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_m['accuracy']:.4f} val_f1={val_m['macro_f1']:.4f}"
        )

        # 以驗證集 macro-F1 挑最佳模型（分數相同時選 loss 較低者），
        # 最後留下的是「在沒看過的資料上表現最好」的版本，而不一定是最後一個 epoch。
        if (val_m["macro_f1"], -val_loss) > (best_f1, -best_loss):
            best_f1, best_loss, best_epoch = val_m["macro_f1"], val_loss, epoch
            save_checkpoint(out_dir / "best.pth", model, class_names, args.img_size, epoch=epoch)
            print("   * Best model saved.")

        # 每個 epoch 結束都覆寫一次「最後狀態」，方便需要時除錯
        save_checkpoint(out_dir / "last.pth", model, class_names, args.img_size, epoch=epoch)

        # Early stopping：驗證分數太久沒進步就停止，避免浪費時間、持續 overfitting
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    plot_curves(history, out_dir / "training_curves.png")

    # ---------- 用「最佳權重」做最後評估：val 與 test 各一次 ----------
    # test 集到這裡才第一次使用，不參與任何挑選模型的決定，所以分數才客觀。
    best = torch.load(out_dir / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    results = {
        "val": run_epoch(model, val_loader, criterion, device, num_classes),
        "test": run_epoch(model, test_loader, criterion, device, num_classes),
    }
    for split, (_, cm) in results.items():
        plot_confusion_matrix(cm, class_names, out_dir / f"confusion_matrix_{split}.png",
                              f"Confusion Matrix ({split}, epoch {best_epoch})")
    report_path = out_dir / "report.txt"
    write_report(report_path, class_names, results)

    test_m = metrics_from_cm(results["test"][1])
    print(f"\nBest epoch: {best_epoch}  (val macro-F1={best_f1:.4f})")
    print(f"Test accuracy={test_m['accuracy']:.4f}  macro-F1={test_m['macro_f1']:.4f}")
    print(f"Saved weights: {out_dir / 'best.pth'}, {out_dir / 'last.pth'}")
    print(f"Saved report:  {report_path}")


if __name__ == "__main__":
    main()
