# 咖啡豆瑕疵偵測 (YOLOv8-OBB)

根據你上傳的 `coffee_bean_defect_v1i_yolov8.zip` 資料集所建立的訓練與辨識程式。

## 資料集重點

- 共 1835 張影像 (train 1285 / valid 367 / test 183)，皆已縮放為 224x224。
- 只有 **1 個類別: `NG`**（瑕疵）。
- 標註格式是每列 `class x1 y1 x2 y2 x3 y3 x4 y4`（4個角點、正規化座標），
  這是 **YOLOv8-OBB（旋轉框）格式**，不是一般的 `x_center y_center w h` 格式。
  → 因此程式使用的是 `yolov8n-obb.pt` 這種 OBB 專用模型，而不是一般的 `yolov8n.pt`。

## 目錄結構

把本資料夾內容和你解壓縮出來的 `train/ valid/ test/` 放在同一層：

```
coffee_bean_yolov8/
├── data.yaml        # 資料集設定 (已修正好路徑)
├── train.py         # 訓練腳本
├── predict.py       # 辨識/推論腳本
├── requirements.txt
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 安裝

```bash
pip install -r requirements.txt
```

若有 NVIDIA GPU，建議先依你的 CUDA 版本安裝好對應的 PyTorch，再安裝 ultralytics，
訓練速度會快非常多。

## 訓練模型

```bash
python train.py
```

常用參數：

```bash
python train.py --model yolov8s-obb.pt --epochs 150 --imgsz 224 --batch 32 --device 0
```

- `--model`：`yolov8n-obb.pt`（最快）到 `yolov8x-obb.pt`（最準但最慢），依你的硬體選擇。
- `--epochs`：訓練回合數，資料量不到2000張，建議先跑100~150回合觀察。
- `--device`：有 GPU 就填 `0`，沒有就留空或填 `cpu`。

訓練完成後，最佳權重會存在：

```
runs/obb/coffee_bean_defect/weights/best.pt
```

訓練過程也會自動產生混淆矩陣、PR curve、loss 曲線等圖表在同一資料夾內，方便檢查訓練成效。

## 用訓練好的模型做辨識

單張圖片：

```bash
python predict.py --weights runs/obb/coffee_bean_defect/weights/best.pt \
    --source test/images/-003_png.rf.xxxx.jpg
```

整個資料夾（例如批次檢驗一批咖啡豆照片）：

```bash
python predict.py --weights runs/obb/coffee_bean_defect/weights/best.pt \
    --source test/images
```

執行後會：

1. 把畫好偵測框的圖片存到 `runs/obb/predict/`。
2. 在終端機印出每張圖的瑕疵數量、平均信心分數、判定結果 (OK / NG)。
3. 另外輸出一份 `summary.csv` 統計表，方便匯入 Excel 做後續分析。

## 調整靈敏度

如果覺得誤判太多（把正常豆判成瑕疵）或漏判太多，可以調整信心閾值：

```bash
python predict.py --weights best.pt --source test/images --conf 0.4   # 提高閾值，判斷更嚴格
python predict.py --weights best.pt --source test/images --conf 0.15  # 降低閾值，判斷更寬鬆
```

## 小提醒

- 因為只有一個類別 `NG`，模型的邏輯是「有偵測到框 = 該影像有瑕疵」，
  沒偵測到任何框則視為正常豆。如果你之後想同時分辨「瑕疵種類」（黑豆、蟲蛀、破損等），
  需要重新標註成多類別資料集，再修改 `data.yaml` 的 `names` 清單。
- 目前資料集裡每張圖是單顆豆的裁切照，若你未來想直接對「一整盤豆子的照片」做偵測，
  建議之後改用更高解析度、多顆豆同框的影像重新標註訓練，效果會更好。
