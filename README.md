# 咖啡生豆好壞辨識（依處理法分模型）

拍一顆咖啡生豆的照片，判斷它是**好豆（good）**還是**壞豆（bad）**。
不同處理法的豆子外觀差異大，所以每種處理法各訓練一個模型，使用時用 `--method` 切換：

| 處理法 | `--method` | 資料集 | 狀態 |
|---|---|---|---|
| 水洗 | `water` | 666 bad + 680 good | 已完成 |
| 日曬 | `sun` | 506 bad + 509 good | 已完成 |
| 蜜處理 | `honey` | 503 bad + 509 good | 已完成 |

## 水洗模型結果

ResNet18（ImageNet 預訓練）二分類，train/val/test = 942 / 202 / 202 張。測試集只在訓練結束後評估一次。

| | precision | recall | F1 | 張數 |
|---|---|---|---|---|
| bad | 0.990 | 0.950 | 0.969 | 100 |
| good | 0.953 | 0.990 | 0.971 | 102 |

測試 accuracy 97.0%、macro-F1 0.970（驗證集最佳 macro-F1 0.985，在第 10 個 epoch）；test 判錯 6 張（漏抓 5、誤判 1）。

## 日曬 / 蜜處理模型結果

同水洗的設定（ResNet18、70/15/15 分層切分），測試集只評估一次。

| | train/val/test | test accuracy | macro-F1 | test 判錯 |
|---|---|---|---|---|
| 日曬 | 711 / 152 / 152 | 99.3% | 0.993 | 漏抓 1、誤判 0 |
| 蜜處理 | 710 / 151 / 151 | 99.3% | 0.993 | 漏抓 1、誤判 0 |

## 壞豆信心度與錯誤清單

- **信心度 p_bad**：模型認為該顆是壞豆的機率（0~1）。`predict.py` 每張都會印出 `bad 信心度`，`Predictor.predict` 回傳 `(label, 判定機率, p_bad)`。可自訂門檻（如 p_bad > 0.3 即視為壞豆）來減少漏抓。
- **每次訓練結束**會在 `runs/<method>/<時間戳>/` 產生：
  - `predictions_val.csv` / `predictions_test.csv`：每張照片的實際/預測類別與 p_bad
  - `errors_val.txt` / `errors_test.txt`：所有判錯的照片，分成「漏抓（壞豆判成好豆）」與「誤判（好豆判成壞豆）」，依離譜程度排序
  - `errors/<split>/{missed,false_alarm}/`：把判錯的照片複製出來，檔名開頭是 p_bad，方便直接檢視
  - 訓練的終端機輸出也會直接列出 test 的錯誤清單

## 資料夾結構

```text
coffee/<method>/{good,bad}/*.jpg   原始照片（4096x3072，一張一顆豆子；不進版本控制）
cache/<method>/{good,bad}/*.jpg    裁成單顆豆子的 256x256 圖（自動產生）
splits/<method>.json               train/val/test 切分結果
models/<method>/best.pth           各處理法的部署模型（predict.py 讀這裡）
runs/<method>/<時間戳>/            每次訓練的紀錄：曲線、混淆矩陣、report.txt
```

程式：`prepare_dataset.py`（裁切＋切分）、`dataset.py`（資料集與前處理）、`model.py`（模型）、`train.py`（訓練）、`predict.py`（推論）。

## 安裝

Python 3.10+：`pip install -r requirements.txt`（GPU 版 PyTorch 請依 CUDA 版本另行安裝）。

## 使用方式

### 1. 準備資料（資料有變動時重跑）

照片依處理法放進 `coffee/<method>/good` 與 `coffee/<method>/bad`，然後：

```bash
python prepare_dataset.py --method water
```

- 偵測豆子位置並裁成方形（原圖豆子很小，直接縮小會看不到瑕疵）。
- 以「類別 × 拍攝日期（取自檔名 `P_YYYYMMDD_*`）」分層抽樣切成 70/15/15，因此每份的 good:bad 都約 50:50，且每天的照片都按比例分進三份。

### 2. 訓練

```bash
python train.py --method water
python train.py --method water --arch resnet50 --epochs 40
```

以驗證集 macro-F1 挑最佳權重並 early stopping，最後評估 test，並把權重複製到 `models/<method>/best.pth`。常用參數：`--epochs --lr --batch-size --patience --img-size --arch --seed`。

### 3. 推論（依豆子處理法切換模型）

```bash
python predict.py --method water --image 豆子照片.jpg
python predict.py --method sun --image 資料夾/        # 也可以給資料夾
```

加 `--save-dir predict_results` 會把每張的裁切圖標上預測結果與信心度存進該資料夾（綠字 good、紅字 bad）。

輸入整張原照片即可，會自動裁出豆子；已裁好的圖加 `--no-crop`。程式中可用 `Predictor("water").predict([PIL 圖片])`。

## 新增日曬 / 蜜處理

1. 照片放進 `coffee/sun/{good,bad}`（或 `honey`）。
2. `python prepare_dataset.py --method sun`
3. `python train.py --method sun`

## 注意事項

- **同一顆豆子若連拍多張**，照片可能同時落在 train 與 test，測試分數會偏高。建議每顆豆子只拍一張，或日後以「豆子編號」為單位切分。
- 資料只有兩天、同一種背景，換拍攝環境（光線、背景紙、手機）時準確率可能下降，建議另拍一批新環境的照片驗證。
- 測試集 202 張，單類錯 1 張約 1%，數字有些波動屬正常。
