# 講稿｜model.py 與 train.py 程式說明

> 預估時間：約 8 ~ 10 分鐘（model.py 約 3 分鐘，train.py 約 6 分鐘）
> 口語稿，念的時候可自行調整。行數以目前的 `model.py`（102 行）與 `train.py`（387 行）為準。

---

## 開場（約 30 秒）

接下來我用兩支程式來說明整個訓練流程。**model.py** 定義「模型長什麼樣子」，**train.py** 定義「模型怎麼被訓練出來」。可以把 model.py 想成一個學生，train.py 則是教他、考他、幫他打成績的整套流程。

---

# Part 1｜model.py：模型長什麼樣子（共 102 行）

## 1-1 整體架構（約 1 分鐘）

📍 `model.py` 第 6–51 行（class `CoffeeBeanClassifier`）；backbone 第 20–29 行、分類頭第 37–38 行

核心是一個 class，叫 `CoffeeBeanClassifier`。它由兩個部分組成：

1. **骨幹網路 backbone**：負責「看懂圖片」。
2. **分類頭 classifier head**：負責「根據看到的東西，猜是哪一類」。

一張圖進來，先過 backbone 變成一串特徵數字，再過分類頭變成各類別的分數，分數最高的就是預測結果。

## 1-2 骨幹網路與遷移學習（約 1 分鐘）

📍 `model.py` 第 20–29 行；拿掉最後一層在第 28 行；`arch` 切換在第 21、25 行

骨幹我們用 torchvision 內建的 **ResNet18**，而且載入的是**在 ImageNet 上訓練過的預訓練權重**。ImageNet 有上百萬張照片，所以這個網路已經學會怎麼認邊緣、紋理、形狀。這等於借用別人訓練好的「眼睛」，我們只有一千多張咖啡豆照片，不可能從零學起，這就是**遷移學習**。

程式裡有一行 `list(net.children())[:-1]`，意思是把 ResNet18 最後那層「ImageNet 1000 類」的分類層拿掉，只留下抽特徵的部分。這個 class 也支援換成 ResNet50 或 EfficientNet-B0，用 `arch` 參數切換，方便之後做比較。

## 1-3 分類頭與 Dropout（約 1 分鐘）

📍 `model.py` 第 37 行（Dropout）、第 38 行（線性層）

backbone 輸出的是一串特徵數字，ResNet18 是 512 個。分類頭就是一個全新的**線性層**，把 512 個數字轉成 `num_classes` 個分數。我們目前是好豆／壞豆，所以 `num_classes = 2`。

這一層是整個模型裡**唯一需要從頭學**的部分，因為 ImageNet 沒教過咖啡豆瑕疵長什麼樣。

分類頭前面加了 **Dropout 0.5**：訓練時隨機丟掉一半特徵，逼模型不要只依賴少數特徵，用來防止小資料集的 overfitting。推論時它會自動關閉。

## 1-4 forward 與存取權重（約 1 分鐘）

📍 `forward` 第 40–51 行；`save_checkpoint` 第 56–68 行；`load_checkpoint` 第 71–92 行（`weights_only=True` 在第 77 行、`pretrained=False` 在第 89 行）；形狀檢查 `__main__` 第 95–102 行

`forward` 的流程很短：圖片進 backbone → 用 `flatten` 把多餘的維度攤平 → 過 Dropout → 過分類頭 → 輸出 logits。

下面兩個函式處理權重檔：

- `save_checkpoint`：除了模型參數，**還一起存類別名稱、圖片大小和架構**。好處是推論時只需要這一個檔案，不會因為前處理設定和訓練時不一致而出錯。
- `load_checkpoint`：載入時用 `weights_only=True`，是比較安全的載入方式；推論時 `pretrained=False`，因為馬上會載入自己訓練的權重，不需要再下載 ImageNet 權重，離線也能用。

最底下的 `__main__` 只是用假資料檢查輸入輸出形狀，輸入 2 張圖，應該輸出 (2, 類別數)。

---

# Part 2｜train.py：模型怎麼被訓練（共 387 行）

我把 train.py 分成五段來講：**輔助工具、一個 epoch、主流程、最終評估、錯誤分析**。

## 2-1 輔助工具：亂數種子與評估指標（約 1 分鐘）

📍 `train.py` `set_seed` 第 23–28 行；`compute_confusion_matrix` 第 31–39 行；`metrics_from_cm` 第 42–62 行

- `set_seed`：固定所有亂數來源（Python、NumPy、PyTorch），同樣設定跑兩次結果一樣，實驗才能比較。
- `compute_confusion_matrix`：建立**混淆矩陣**。縱軸是真實類別，橫軸是預測類別，對角線是猜對，其他是猜錯，一眼看出誤判的方向。
- `metrics_from_cm`：從混淆矩陣算出 accuracy、precision、recall、F1 和 **macro-F1**。macro-F1 是每類 F1 直接平均，不會被某一類的數量掩蓋。

## 2-2 run_epoch：跑完一輪資料（約 1.5 分鐘）

📍 `train.py` 第 65–101 行。訓練／評估切換第 70–71 行、不算梯度第 75 行、autocast 第 81 行；四步驟：猜答案第 82 行、算 loss 第 83 行、`zero_grad` 第 86 行、`backward` 第 88 行（或 92 行）、`step` 第 89 行（或 93 行）

這是訓練的核心。有傳 optimizer 就是**訓練**，沒傳就是**評估**，同一個函式兩用。

訓練時每一批資料做四件事：

1. **模型猜答案**：`logits = model(images)`
2. **算猜錯多少**：`loss = criterion(logits, labels)`
3. **反向傳播**：`loss.backward()`，算出每個參數該往哪個方向調。
4. **更新參數**：`optimizer.step()`。

每批開始前要 `zero_grad()` 清掉上一批的梯度，避免累加。

幾個細節：

- 評估時用 `set_grad_enabled(False)`，不算梯度，省記憶體。
- `model.train()` 與 `model.eval()` 切換，會影響 Dropout 和 BatchNorm 的行為。
- 有 GPU 時開 **autocast 與 GradScaler**，用半精度運算，更快更省顯存。

最後回傳這個 epoch 的平均 loss 與混淆矩陣。

## 2-3 main()：訓練設定（約 1.5 分鐘）

📍 `train.py` `main()` 第 212–383 行。參數 argparse 第 215–237 行；讀切分檔第 247–251 行；輸出資料夾與 config 第 254–257 行；資料集（增強／不增強）第 264–267 行；DataLoader 第 271–278 行；建模型第 281 行；Loss 與類別權重第 286–291 行；Optimizer 第 295 行；Scheduler 第 299 行；GradScaler 第 302 行

用 `argparse` 讓所有設定都能從命令列調整，例如 `python train.py --method sun`，不用改程式碼。預設值就是週報裡的設定：ResNet18、學習率 1e-4、batch size 16、最多 30 個 epoch、dropout 0.5、weight decay 0.05、label smoothing 0.1。

接著做準備工作：

- 讀入 `prepare_dataset.py` 產生的**切分 json**，固定 train / val / test。
- 每次訓練存到 `runs/<method>/<時間戳>`，**不會覆蓋之前的實驗**，並把設定存成 config.json。
- **訓練集做資料增強，驗證與測試集不做**，評估標準才穩定。
- 訓練集 `shuffle=True`，每個 epoch 順序都打亂。

然後建立三樣東西：

- **Loss**：CrossEntropy，加上**類別權重**（樣本少的類別權重高）和 **label smoothing 0.1**（不讓模型對答案過度自信）。
- **Optimizer**：AdamW，負責根據梯度實際更新參數，weight decay 是另一種防 overfitting 的手法。
- **Scheduler**：ReduceLROnPlateau，驗證 loss 連續 3 個 epoch 沒進步，學習率減半，後期用小步伐微調。

## 2-4 訓練迴圈：挑最佳模型與提早停止（約 1.5 分鐘）

📍 `train.py` 第 308–348 行。訓練＋驗證第 309–311 行；寫 `history.csv` 第 324–327 行；挑最佳模型存 `best.pth` 第 337–340 行；存 `last.pth` 第 343 行；Early stopping 第 346–348 行

迴圈每個 epoch 做：先訓練一輪、再驗證一輪，把 loss、accuracy、macro-F1 寫進 `history.csv`。

這裡有兩個重要設計：

1. **用驗證集 macro-F1 挑最佳模型**（分數相同比 loss），存成 `best.pth`。所以留下的是「在沒看過的資料上表現最好」的版本，不一定是最後一個 epoch。另外每個 epoch 也會覆寫 `last.pth` 方便除錯。
2. **Early stopping**：連續 8 個 epoch 沒有刷新最佳成績就停，避免浪費時間，也避免繼續 overfitting。

## 2-5 最終評估與錯誤分析（約 1.5 分鐘）

📍 `train.py` 畫訓練曲線第 350 行；載回最佳權重並評估 val／test 第 354–359 行；混淆矩陣圖與報告第 360–364 行；`predict_probs` 第 162–168 行；`analyze_errors` 第 171–209 行（p_bad 第 188 行、漏抓／誤判分類第 190–191 行、排序第 193–194 行、複製錯誤照片第 204–208 行）；呼叫錯誤分析第 367–370 行；部署權重到 `models/<method>/best.pth` 第 377–380 行

訓練結束後，載回 `best.pth`，對 val 與 test **各評估一次**。重點是：**測試集到這一步才第一次使用**，沒有參與任何挑選，所以分數是客觀的。評估會輸出混淆矩陣圖、訓練曲線圖和 `report.txt`。

這週新增的是**錯誤分析**（`analyze_errors`）：

- `predict_probs` 對每張圖算 softmax 機率，其中壞豆那一欄就是 **p_bad，壞豆信心度**。
- 把判錯的分成兩種：
  - **漏抓**：真的是壞豆卻判成好豆，最危險，因為壞豆會混進成品。p_bad 越低，代表模型錯得越「理直氣壯」。
  - **誤判**：好豆被當成壞豆，損失較小。
- 輸出每張的預測 csv、錯誤清單 txt，並把錯誤照片複製到 `errors/` 資料夾，檔名前面加 p_bad，方便直接挑出最離譜的來看。

最後，把最佳權重複製到固定位置 `models/<method>/best.pth`，之後 `predict.py` 或 App 用 `--method` 切換處理法時，就是讀這裡。

---

## 收尾（約 30 秒）

總結一下：**model.py** 是「ImageNet 預訓練的 ResNet18 加一個新的分類頭」，並且把類別名稱和圖片大小跟權重存在一起；**train.py** 是完整可重現的訓練流程，包含正則化、用驗證集挑模型、測試集只評估一次，以及這週新增的壞豆信心度與錯誤分析。

謝謝，歡迎提問。
