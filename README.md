# 咖啡豆瑕疵偵測系統 (YOLOv8-OBB)

一個基於 YOLOv8-OBB（旋轉框目標檢測）的自動化咖啡豆瑕疵檢測系統。採用深度學習技術，能夠快速、準確地識別咖啡豆上的瑕疵，適用於咖啡豆品質控制和分選流程。

---

## 📋 目錄

- [專案概述](#專案概述)
- [資料集信息](#資料集信息)
- [目錄結構](#目錄結構)
- [快速開始](#快速開始)
- [詳細說明](#詳細說明)
- [進階配置](#進階配置)
- [常見問題](#常見問題)

---

## 專案概述

### 核心特性

- **YOLOv8-OBB 模型**：使用旋轉框格式，能精確偵測各種角度的瑕疵
- **小型資料集優化**：使用 1,835 張訓練圖像，訓練高效快速
- **批量處理能力**：支援單張圖片或整個資料夾的批量推理
- **詳細報告輸出**：自動生成 CSV 統計表和可視化檢測結果
- **靈活調整**：支援信心閾值調整，適應不同品質標準

### 應用場景

- ✅ 咖啡豆分選線的自動檢測
- ✅ 品質控制和品質保證
- ✅ 瑕疵統計和趨勢分析
- ✅ 傳送帶異物檢測

---

## 資料集信息

### 資料集概況

| 指標 | 值 |
|------|-----|
| **總圖像數** | 1,835 張 |
| **訓練集** | 1,285 張 |
| **驗證集** | 367 張 |
| **測試集** | 183 張 |
| **圖像分辨率** | 224 × 224 px |
| **類別數** | 1 |
| **類別名稱** | `NG`（瑕疵/不合格） |
| **預處理** | 自動方向調整、拉伸至 224×224 |

### 標籤格式

採用 **YOLOv8-OBB（旋轉框）格式**，每行一個檢測框：

```
<class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

- `<class>`：類別索引（0 = NG）
- `<x1~y4>`：四邊形四個頂點的正規化座標（0~1）

**重要**：這與標準 YOLO 格式不同（標準格式為 `class x_center y_center w h`）。因此必須使用 **OBB 專用模型**（如 `yolov8n-obb.pt`），而不是通用的 YOLO 模型。

### 資料來源

資料集來自 Roboflow 平台：
- **匯出時間**：2024 年 1 月 25 日
- **資料集名稱**：Coffee Bean Defect - v1
- **標註工具**：Roboflow Annotation

---

## 目錄結構

```
coffee_bean_defect/
├── README.md                           # 本文件
├── data.yaml                           # 資料集配置文件
├── train.py                            # 訓練腳本
├── predict.py                          # 推理/預測腳本
├── test_cuda.py                        # CUDA 環境檢測工具
├── requirements.txt                    # 依賴包列表
├── yolov8n-obb.pt                      # 預訓練模型（nano 版本）
│
├── train/                              # 訓練集
│   ├── images/                         # 訓練圖片 (1285 張)
│   └── labels/                         # 訓練標籤 (OBB 格式)
│
├── valid/                              # 驗證集
│   ├── images/                         # 驗證圖片 (367 張)
│   └── labels/                         # 驗證標籤
│
├── test/                               # 測試集
│   ├── images/                         # 測試圖片 (183 張)
│   └── labels/                         # 測試標籤
│
├── weights/                            # 存放訓練好的模型
│   └── yolo26n.pt
│
└── runs/                               # 訓練和推理輸出
    └── obb/
        ├── coffee_bean_defect/         # 訓練結果（權重、圖表等）
        │   └── weights/
        │       ├── best.pt             # 最佳模型
        │       └── last.pt
        └── predict/                    # 推理結果
            ├── images/                 # 標註後的圖片
            └── summary.csv             # 統計報告
```

---

## 快速開始

### 1️⃣ 環境設置

#### 前置需求
- Python 3.8+
- pip（Python 包管理工具）
- (可選) NVIDIA GPU 與 CUDA 工具包（加速訓練）

#### 安裝依賴

```bash
# 基礎安裝
pip install -r requirements.txt

# 如果有 NVIDIA GPU，建議先安裝 GPU 支援版本的 PyTorch
# 根據你的 CUDA 版本選擇相應的命令
# 例如 CUDA 12.1：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 驗證 CUDA 環境（可選）

```bash
python test_cuda.py
```

### 2️⃣ 訓練模型

#### 基礎訓練

使用預設參數啟動訓練：

```bash
python train.py
```

#### 自訂訓練參數

```bash
python train.py \
  --model yolov8s-obb.pt \
  --epochs 150 \
  --imgsz 224 \
  --batch 32 \
  --device 0
```

#### 常用訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `yolov8s-obb.pt` | 模型大小（nano/small/medium/large/xlarge） |
| `--epochs` | 200 | 訓練回合數 |
| `--imgsz` | 640 | 輸入圖像大小 |
| `--batch` | 16 | 批量大小 |
| `--device` | 自動 | GPU ID（0, 1, ...）或 `cpu` |
| `--lr0` | 0.01 | 初始學習率 |
| `--patience` | 50 | 早停耐心值 |
| `--optimizer` | auto | 優化器（auto/SGD/Adam/AdamW） |
| `--cache` | ram | 緩存方式（ram/disk/False） |
| `--plots` | True | 是否生成訓練圖表 |

#### 模型尺寸對比

| 模型 | 速度 | 精度 | VRAM | 適用場景 |
|------|------|------|------|---------|
| `yolov8n-obb` | ⚡⚡⚡ 最快 | ★★☆ | 低 | 實時應用、邊緣設備 |
| `yolov8s-obb` | ⚡⚡ | ★★★ | 中 | **推薦** |
| `yolov8m-obb` | ⚡ | ★★★★ | 中高 | 高精度要求 |
| `yolov8l-obb` | 中等 | ★★★★★ | 高 | 優先精度 |
| `yolov8x-obb` | 最慢 | ★★★★★ | 非常高 | 最高精度要求 |

#### 訓練完成後

訓練完成時，最佳權重和其他輸出存放在：

```
runs/obb/coffee_bean_defect/
├── weights/
│   ├── best.pt          # ⭐ 最佳模型（驗證集上表現最好）
│   └── last.pt          # 最後一個 epoch 的模型
├── results.csv          # 每個 epoch 的訓練指標
├── confusion_matrix.png # 混淆矩陣
├── results.png          # 訓練曲線（loss、精度等）
└── ...
```

### 3️⃣ 模型推理/預測

#### 推理單張圖片

```bash
python predict.py \
  --weights runs/obb/coffee_bean_defect/weights/best.pt \
  --source test/images/-003_png.rf.xxxx.jpg
```

#### 推理整個資料夾

```bash
python predict.py \
  --weights runs/obb/coffee_bean_defect/weights/best.pt \
  --source test/images
```

#### 推理結果

推理完成後，輸出位置：`runs/obb/predict/`

輸出內容包括：
1. **標註圖片**：在 `images/` 資料夾，每張圖片都繪製了偵測框
2. **終端輸出**：每張圖片的瑕疵數量、信心分數、判定結果（OK/NG）
3. **統計報告**：`summary.csv` 檔案，包含所有圖片的彙總統計

#### 調整信心閾值

如果需要調整檢測的靈敏度：

```bash
# 嚴格模式：降低誤判率（漏判率可能上升）
python predict.py \
  --weights best.pt \
  --source test/images \
  --conf 0.4

# 寬鬆模式：降低漏判率（誤判率可能上升）
python predict.py \
  --weights best.pt \
  --source test/images \
  --conf 0.15
```

#### 推理參數說明

| 參數 | 預設 | 說明 |
|------|------|------|
| `--weights` | 必須 | 模型權重路徑 |
| `--source` | 必須 | 圖片/資料夾路徑 |
| `--imgsz` | 224 | 推理時的輸入圖像大小 |
| `--conf` | 0.25 | 信心閾值（0~1），越高越嚴格 |
| `--device` | 自動 | 設備（0 = GPU0，cpu = CPU） |
| `--project` | runs/obb | 輸出上層資料夾 |
| `--name` | predict | 輸出資料夾名稱 |
| `--csv` | 預設 | 自訂 CSV 輸出路徑 |

---

## 詳細說明

### 訓練流程詳解

1. **資料載入**：根據 `data.yaml` 配置載入訓練、驗證、測試集
2. **模型初始化**：載入指定的預訓練 YOLOv8-OBB 模型
3. **訓練迴圈**：
   - 每個 epoch 使用訓練集進行前向和反向傳播
   - 計算 loss（分類 loss + 迴歸 loss + objectness loss）
   - 每個 epoch 後在驗證集上評估性能
4. **模型選擇**：根據驗證集性能選擇最佳模型保存為 `best.pt`
5. **早停機制**：如果驗證集性能連續 50 個 epoch 不改善，自動停止訓練

### OBB 標籤格式解說

YOLOv8-OBB 標籤使用四個頂點座標表示旋轉框：

```
# 例子：
0 0.5 0.5 0.6 0.4 0.7 0.6 0.6 0.7
├─ 類別   0（NG）
├─ 點 1（x1, y1）：0.5, 0.5
├─ 點 2（x2, y2）：0.6, 0.4
├─ 點 3（x3, y3）：0.7, 0.6
└─ 點 4（x4, y4）：0.6, 0.7
```

所有座標都被正規化至 [0, 1] 範圍，與影像大小無關。

### CSV 報告格式

推理完成後生成的 `summary.csv` 包含以下列：

| 欄位 | 說明 |
|------|------|
| `image` | 圖片名稱 |
| `num_defects` | 偵測到的瑕疵數量 |
| `mean_confidence` | 所有檢測的平均信心分數 |
| `result` | 判定結果（OK = 無瑕疵，NG = 有瑕疵） |

---

## 進階配置

### 資料增強設定

在 `train.py` 中可調整資料增強參數：

```python
# 旋轉角度（度數）
parser.add_argument("--degrees", type=float, default=180.0)

# 上下翻轉概率
parser.add_argument("--flipud", type=float, default=0.5)

# 左右翻轉概率
parser.add_argument("--fliplr", type=float, default=0.5)

# 縮放比例
parser.add_argument("--scale", type=float, default=0.5)

# 平移幅度
parser.add_argument("--translate", type=float, default=0.1)

# 色調飽和度亮度調整
parser.add_argument("--hsv_h", type=float, default=0.015)  # 色調
parser.add_argument("--hsv_s", type=float, default=0.7)    # 飽和度
parser.add_argument("--hsv_v", type=float, default=0.4)    # 亮度

# Mosaic 和 Mixup
parser.add_argument("--mosaic", type=float, default=1.0)
parser.add_argument("--mixup", type=float, default=0.1)
```

### 恢復訓練

如果訓練被中斷，可以恢復上次的訓練進度：

```bash
python train.py --resume
```

此命令會自動載入最後保存的模型和訓練狀態。

### 多卡 GPU 訓練

```bash
# 使用 GPU 0
python train.py --device 0

# 使用 GPU 0 和 1（分佈式訓練）
python train.py --device 0,1
```

### 優化器選擇

```bash
# 自動選擇（推薦）
python train.py --optimizer auto

# 使用 SGD
python train.py --optimizer SGD --lr0 0.01

# 使用 Adam
python train.py --optimizer Adam --lr0 0.001

# 使用 AdamW
python train.py --optimizer AdamW --lr0 0.001
```

---

## 常見問題

### Q1: 訓練速度太慢

**A:** 如果沒有 GPU，訓練會非常慢。解決方案：
- 檢查是否正確安裝 GPU 版本的 PyTorch 和 CUDA
- 執行 `python test_cuda.py` 驗證 GPU 可用性
- 使用更小的模型（如 `yolov8n-obb.pt`）
- 減少 epoch 數量進行測試

### Q2: 顯示記憶體不足 (OOM) 錯誤

**A:** 減少批量大小或使用更小的模型：

```bash
python train.py --model yolov8n-obb.pt --batch 8
```

### Q3: 推理結果中所有圖片都被判定為 NG

**A:** 可能是信心閾值設定過低。嘗試增加閾值：

```bash
python predict.py --weights best.pt --source test/images --conf 0.5
```

### Q4: 模型訓練不收斂（loss 沒有下降）

**A:** 可能的原因和解決方案：
- 學習率過高：`python train.py --lr0 0.001`
- 數據標籤有問題：檢查標籤文件格式
- 模型不適合資料：嘗試更大的模型
- 資料量不足：考慮加入資料增強或收集更多資料

### Q5: 如何在自訂 GPU 上進行推理

**A:** 指定 device 參數：

```bash
# 使用 GPU 0
python predict.py --weights best.pt --source test/images --device 0

# 使用 CPU
python predict.py --weights best.pt --source test/images --device cpu
```

### Q6: 能否修改類別名稱或新增類別

**A:** 當前項目只有 1 個類別（NG），如需修改需要：
1. 重新標註數據集（修改 `labels/*.txt` 檔案）
2. 更新 `data.yaml` 中的 `nc` 和 `names`
3. 重新訓練模型

---

## 許可證和引用

此項目基於以下技術：

- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **資料集**: Roboflow 平台提供的咖啡豆數據集

### 引用

如在學術或商業項目中使用，請引用：

```bibtex
@article{ultralytics2023,
  title={YOLOv8: A New Frontier in Real-Time Object Detection},
  author={Ultralytics},
  year={2023}
}
```

---

## 聯絡與支援

如有問題或建議，歡迎：
- 檢查本 README 的[常見問題](#常見問題)部分
- 查看 [Ultralytics 官方文件](https://docs.ultralytics.com)
- 參考 [YOLOv8 GitHub 倉庫](https://github.com/ultralytics/ultralytics)

---

**最後更新**: 2024 年 1 月 25 日  
**YOLOv8 版本**: 8.1.0+  
**Python 版本**: 3.8+
  沒偵測到任何框則視為正常豆。如果你之後想同時分辨「瑕疵種類」（黑豆、蟲蛀、破損等），
  需要重新標註成多類別資料集，再修改 `data.yaml` 的 `names` 清單。
- 目前資料集裡每張圖是單顆豆的裁切照，若你未來想直接對「一整盤豆子的照片」做偵測，
  建議之後改用更高解析度、多顆豆同框的影像重新標註訓練，效果會更好。
