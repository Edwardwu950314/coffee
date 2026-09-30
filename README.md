# Coffee Bean Defect Classification Project

這是一個針對咖啡豆缺陷類別辨識的影像分類專案。專案使用預訓練的 ResNet18 作為主幹網路，並透過遷移學習方式對 18 種咖啡豆狀態進行分類。

本專案的資料集不是 YOLO 目標偵測格式，而是「每個類別一個資料夾」的分類資料集，因此訓練與推論均採用影像分類流程。

---

## 專案目標

判斷咖啡豆屬於哪一種缺陷類別或正常狀態：

Broken、Cut、Dry Cherry、Fade、Floater、Full Black、Full Sour、Fungus Damage、Husk、Immature、Normal、Parchment、Partial Black、Partial Sour、Severe Insect Damage、Shell、Slight Insect Damage、Withered

---

## 模型架構

- Backbone: ResNet18（ImageNet 預訓練權重）
- 分類頭: `Linear(512, num_classes)`
- Loss: 依類別數量加權的 Cross Entropy Loss（處理類別不平衡）
- Optimizer: AdamW + ReduceLROnPlateau
- 選模指標: 驗證集 macro-F1，並有 early stopping
- GPU 上自動啟用混合精度 (AMP)

---

## 資料集結構與切分

```text
data/
├── train/<類別>/*.jpg     # 原始訓練資料
├── test/<類別>/*.jpg      # 原始測試資料
└── splits.json            # make_split.py 產生的切分檔
```

`make_split.py` 會產生 `data/splits.json`（**不會移動或刪除任何圖片**）：

- 用 MD5 找出內容完全相同的圖片：
  - 出現在不同類別的同一張圖 → 標籤矛盾，全部排除
  - 同時出現在 train 與 test → 只保留 test 那份，避免資料洩漏
  - 同類別內重複 → 只保留一張
- 從 train 依類別比例切出 **val**（預設 15%），用於挑選最佳權重與調整學習率
- **test 只在訓練結束後評估一次**，不參與任何模型選擇，分數才客觀

目前切分結果：train 773 / val 134 / test 108 張。

---

## 專案檔案說明

```text
.
├── dataset.py         # 資料集、前處理（訓練與推論共用）
├── make_split.py      # 產生去重後的 train/val/test 切分檔
├── model.py           # 模型定義、checkpoint 存取
├── train.py           # 訓練主程式
├── predict.py         # 單張圖片推論
├── batch_predict.py   # 批次推論與抽查
├── requirements.txt
└── runs/<時間戳>/      # 每次訓練的輸出（不進版本控制）
```

---

## 安裝環境

Python 3.10+：

```bash
pip install -r requirements.txt
```

GPU 版 PyTorch 請依 CUDA 版本安裝，參考 `requirements.txt` 開頭的說明。

---

## 使用方式

所有預設路徑都以專案根目錄為基準，從任何目錄執行都可以。

### 1. 產生資料切分（資料有變動時重新執行）

```bash
python make_split.py
```

### 2. 訓練

```bash
python train.py
python train.py --epochs 50 --lr 5e-5 --batch-size 32
```

常用參數：`--patience`（early stopping）、`--seed`、`--img-size`、`--no-class-weights`、`--no-amp`。

每次訓練輸出到獨立的 `runs/<時間戳>/`，不會覆蓋之前的結果：

| 檔案 | 內容 |
|---|---|
| `best.pth` / `last.pth` | 權重檔（內含類別名稱與圖片大小，推論只需這一個檔案） |
| `config.json` | 本次訓練參數 |
| `history.csv` | 每個 epoch 的 loss、accuracy、macro-F1、learning rate |
| `training_curves.png` | loss 與驗證指標曲線 |
| `confusion_matrix_val.png` / `confusion_matrix_test.png` | 混淆矩陣（含數字） |
| `report.txt` | val / test 的每類 precision、recall、F1 |

### 3. 推論

```bash
python predict.py --image "path/to/image.jpg"
python predict.py --image img.jpg --model runs/20260923-120000/best.pth
```

未指定 `--model` 時自動使用最新的 `runs/*/best.pth`；未指定 `--image` 時使用 `data/test` 中的第一張圖。舊版 `weights/best.pth`（只存參數）仍可載入，類別名稱會從 `--class-dir`（預設 `data/train`）讀取。

### 4. 批次抽查

```bash
python batch_predict.py                  # 隨機抽 5 張
python batch_predict.py --num-images 20 --seed 0
python batch_predict.py --all
```

結果圖存到 `data/test_results/`，並顯示正確答案與抽樣準確率。

---

## 已知資料問題

- **Normal 類別的圖片是 200×200，其他類別都是 500×500**。模型可能學到「解析度/模糊程度 = Normal」這種捷徑，而不是豆子本身的特徵。建議補拍同規格的 Normal 圖片，或檢查混淆矩陣中 Normal 的表現是否異常地好。
- `Fade_08.jpg` 與 `Partial Sour_06.jpg` 是同一張圖，標籤矛盾，已在切分時排除，建議人工確認正確類別。
- test 集每類只有 6 張，單類準確率波動大（錯 1 張 ≈ 17%），解讀時需注意。

---

## 下一步建議

- 擴充資料量，特別是 test 集與 Normal 類別
- 嘗試 ResNet50 / EfficientNet / ConvNeXt 等 backbone
- 用 k-fold cross-validation 取得更穩定的評估

---

## 下週目標

- [ ] **補拍自己的豆子**：好豆與壞豆都要，目前自拍好豆只剩 35 張（測試集僅 6 張），無法可靠評估實際表現。拍完後裁成單顆、統一 500×500，依同樣標準標好壞，併入 `data/merged/`。
- [ ] **統一好壞標準**：「偏黃算不算壞」還沒決定（影響 USK 裡約 60 張壞豆漏抓）；「銀皮斑塊算壞豆」目前只套用在自拍資料，USK 的好豆尚未依此複查。
- [ ] **用新資料重訓並評估**：以 ResNet50 重訓（目前測試 accuracy 94.7%、macro-F1 0.936，`runs/merged_resnet50_v2/`），分開看自拍與 USK 的好豆／壞豆表現。
