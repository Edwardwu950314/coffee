# 教授問答小抄（ResNet50 版，附程式碼行數）

> 行數對應目前版本的 `model.py`、`train.py`、`make_binary_split.py`，程式改動後行數會位移。

## 一分鐘版專案流程

人工標註好豆/壞豆（`binary_labels.csv`）→ `make_binary_split.py` 產生固定的 train/val/test 切分（`splits_binary.json`）→ `train.py` 以 ResNet50 遷移學習訓練 → 存 checkpoint（含權重、類別名、圖片大小、arch）→ 推論時載入 checkpoint 預測新照片。

執行指令：

```
python make_binary_split.py
python train.py --split data/splits_binary.json
```

注意：`train.py` 的 `--split` 預設是 `splits.json`（[train.py:164](train.py#L164)），二分類要明確指定 `splits_binary.json`。

---

## 一、基本概念

| 問題 | 答案 | 程式碼 |
|---|---|---|
| 專案在做什麼？ | 影像分類：輸入咖啡豆照片，輸出類別（好/壞或缺陷種類） | [model.py:6](model.py#L6) |
| 輸入和輸出？ | 輸入 (批次, 3, 高, 寬) 的圖片張量，輸出每個類別一個分數 | [model.py:41](model.py#L41) |
| 訓練 vs 推論？ | 訓練是用有標籤資料調權重；推論是用訓練好的模型預測新圖 | [train.py:64](train.py#L64)、[model.py:91](model.py#L91) |
| 遷移學習？ | 借用 ImageNet 預訓練的 ResNet50，資料少也能訓練好 | [model.py:20](model.py#L20) |
| overfitting？ | 背訓練資料，對沒看過的資料表現差 | [train.py:175-177](train.py#L175-L177) |
| 為什麼切三份？ | train 學習、val 調參與選模型、test 最後只評估一次 | [make_binary_split.py:38](make_binary_split.py#L38) |
| epoch / batch size？ | 看完全部資料一輪 / 每次丟進模型的張數 | [train.py:168-170](train.py#L168-L170) |
| loss？ | 預測與答案的差距，訓練就是讓它變小 | [train.py:236](train.py#L236) |
| 只看 accuracy 的問題？ | 類別不平衡時會誤導，要看 precision、recall、F1 | [train.py:41-61](train.py#L41-L61) |
| `if __name__ == "__main__":`？ | 只有直接執行該檔案才會跑，被 import 時不會跑 | [train.py:318](train.py#L318)、[model.py:95](model.py#L95) |
| 為什麼固定 random seed？ | 讓結果可重現 | [train.py:22-27](train.py#L22-L27) |

---

## 二、模型架構（model.py）

| 問題 | 答案 | 程式碼 |
|---|---|---|
| 形狀變化？ | (B,3,224,224) → backbone → (B,2048,1,1) → flatten → (B,2048) → Dropout → Linear(2048,N) → (B,N) logits | [model.py:40-51](model.py#L40-L51) |
| 為什麼選 ResNet50？ | 比 ResNet18 深，特徵更豐富；週報有比較表（`merged_resnet50_v2`：test accuracy 0.947、macro-F1 0.936） | [train.py:173](train.py#L173) |
| 哪部分從頭學？ | 只有分類頭 `Linear(2048, N)`；backbone 用 ImageNet 權重 | [model.py:38](model.py#L38) |
| `children()[:-1]`？ | 拿掉最後的 1000 類 fc 層，只留特徵抽取 | [model.py:28](model.py#L28) |
| `flatten(features, 1)`？ | 把 (B,2048,1,1) 攤平成 (B,2048)，Linear 才吃得進去；`1` 表示保留 batch 維 | [model.py:47](model.py#L47) |
| logits vs 機率？ | logits 是未正規化分數，softmax 後才是機率；`CrossEntropyLoss` 內部已含 softmax | [model.py:51](model.py#L51)、[train.py:236](train.py#L236) |
| Dropout？ | 訓練時隨機丟特徵防 overfitting；`model.eval()` 時自動關閉 | [model.py:37](model.py#L37) |
| `class ...(nn.Module)`？ | 繼承 PyTorch 模型基底類別 | [model.py:6](model.py#L6) |
| `__init__` / `forward`？ | `__init__` 定義有哪些層，`forward` 定義資料怎麼流過 | [model.py:11](model.py#L11)、[model.py:40](model.py#L40) |
| `super().__init__()`？ | 初始化父類別，否則 PyTorch 無法追蹤參數 | [model.py:12](model.py#L12) |
| `pretrained=False` 推論時為何可以？ | 馬上會 `load_state_dict` 載入自己的權重，不必再下載 ImageNet 權重 | [model.py:89-90](model.py#L89-L90) |
| 預設為何曾是 resnet18？ | 歷史遺留，舊 checkpoint 沒存 arch，載入時 fallback 為 resnet18；新 checkpoint 會存 arch。類別預設已改為 resnet50，與 train.py 一致 | [model.py:11](model.py#L11)、[model.py:84-85](model.py#L84-L85) |

---

## 三、資料切分（make_binary_split.py）

| 問題 | 答案 | 程式碼 |
|---|---|---|
| 為什麼獨立成腳本？ | 切分存成 JSON，每次訓練用同一份，實驗才能公平比較 | [make_binary_split.py:54-55](make_binary_split.py#L54-L55) |
| 標籤來源？ | 人工逐張判斷，記錄在 `binary_labels.csv`（欄位 path、label、split） | [make_binary_split.py:25-33](make_binary_split.py#L25-L33) |
| 類別順序？ | `["bad", "good"]`，用 index 當標籤 | [make_binary_split.py:18](make_binary_split.py#L18)、[make_binary_split.py:27](make_binary_split.py#L27) |
| USK-Coffee？ | 沿用它原本的 train/val/test | [make_binary_split.py:28-29](make_binary_split.py#L28-L29)、[make_binary_split.py:44-45](make_binary_split.py#L44-L45) |
| 沿用 base 切分？ | 讀 `splits_base.json`，把 resized 路徑還原成原始路徑去查標籤 | [make_binary_split.py:35-42](make_binary_split.py#L35-L42) |
| extra（整盤裁切圖）怎麼分？ | shuffle 後切片，val 15%、test 15%、其餘 train，三份不重疊 | [make_binary_split.py:47-52](make_binary_split.py#L47-L52) |
| 固定 seed / `random.Random(seed)`？ | 讓切分可重現；獨立亂數物件不受全域亂數狀態影響 | [make_binary_split.py:47](make_binary_split.py#L47) |
| leakage 風險？ | 同一顆豆的多張照片若分到不同集合，test 分數會虛高 | — |
| 印出 Counter？ | 檢查類別是否平衡 | [make_binary_split.py:56-57](make_binary_split.py#L56-L57) |

---

## 四、你的實際訓練設定（train.py）

| 項目 | 設定 | 程式碼 |
|---|---|---|
| 架構 | ResNet50（預訓練） | [train.py:173](train.py#L173)、[train.py:226](train.py#L226) |
| Optimizer | AdamW，lr=1e-4，weight_decay=0.05 | [train.py:240](train.py#L240) |
| Loss | CrossEntropyLoss + 類別權重 + label smoothing 0.1 | [train.py:229-236](train.py#L229-L236) |
| Dropout | 0.5 | [train.py:175](train.py#L175) |
| Batch size / epochs | 16 / 最多 30 | [train.py:168-170](train.py#L168-L170) |
| Scheduler | ReduceLROnPlateau：val loss 連續 3 個 epoch 沒進步，lr 減半 | [train.py:244](train.py#L244) |
| Early stopping | 8 個 epoch 沒進步就停 | [train.py:291-293](train.py#L291-L293) |
| 混合精度 | GPU 上自動啟用（autocast + GradScaler） | [train.py:80](train.py#L80)、[train.py:247](train.py#L247) |
| 圖片大小 | 224 | [train.py:172](train.py#L172) |
| 裝置選擇 | 有 GPU 用 cuda，否則 cpu | [train.py:188-191](train.py#L188-L191) |
| DataLoader | train 打亂、val/test 不打亂 | [train.py:216-223](train.py#L216-L223) |
| 資料增強 | 只有 train 做，val/test 不做 | [train.py:209-212](train.py#L209-L212) |

### 訓練迴圈（`run_epoch`，[train.py:64-100](train.py#L64-L100)）

有傳 optimizer 就是訓練，沒傳就是評估（[train.py:69](train.py#L69)）。

1. 圖片送進模型得到 logits，算 loss（[train.py:81-82](train.py#L81-L82)）
2. `optimizer.zero_grad()` 清梯度（[train.py:85](train.py#L85)）
3. `loss.backward()` 反向傳播，GPU 上經 scaler（[train.py:87](train.py#L87)、[train.py:91](train.py#L91)）
4. `optimizer.step()` 更新權重（[train.py:88](train.py#L88)、[train.py:92](train.py#L92)）
5. 每個 epoch 後跑 val，把 val loss 交給 scheduler（[train.py:255-256](train.py#L255-L256)）

### 選哪個 checkpoint

- 標準是 **val macro-F1**，不是 loss，也不是最後一個 epoch；分數相同時選 val loss 較低者（[train.py:282-285](train.py#L282-L285)）
- 存兩份：`best.pth`（[train.py:284](train.py#L284)）、`last.pth`（[train.py:288](train.py#L288)）
- test 集只在訓練結束後、載入 `best.pth` 才用一次，沒參與任何選模型的決定（[train.py:297-304](train.py#L297-L304)）

---

## 五、教授可能追問

| 問題 | 答案 | 程式碼 |
|---|---|---|
| 為什麼用 macro-F1 選模型，不用 accuracy？ | 每個類別權重相同，不會被數量多的類別掩蓋少數類別的表現 | [train.py:282](train.py#L282) |
| 類別權重怎麼算？ | `總數 / (類別數 × 該類數量)`，圖片越少的類別權重越高 | [train.py:233-235](train.py#L233-L235) |
| label smoothing 是什麼？ | 不讓模型對答案過度自信，是一種正則化 | [train.py:177](train.py#L177)、[train.py:236](train.py#L236) |
| 用了哪些防 overfitting 的方法？ | Dropout 0.5、weight decay 0.05、label smoothing、資料增強（只在 train）、early stopping | [train.py:175-177](train.py#L175-L177)、[train.py:291](train.py#L291) |
| `model.train(training)` 在做什麼？ | 切換訓練與評估模式，影響 Dropout 和 BatchNorm | [train.py:70](train.py#L70) |
| 為什麼 val/test 不做資料增強？ | 評估標準要穩定，才能公平比較 | [train.py:208-212](train.py#L208-L212) |
| GradScaler 為什麼需要？ | float16 的梯度容易下溢成 0，先放大 loss 再還原，保持數值穩定 | [train.py:86-89](train.py#L86-L89) |
| `set_seed` 為什麼設四個？ | Python、NumPy、PyTorch CPU、PyTorch GPU 各有獨立的亂數來源 | [train.py:22-27](train.py#L22-L27) |
| 為什麼每個 batch 要 `zero_grad()`？ | PyTorch 梯度預設會累加，不清掉會混入上一個 batch 的梯度 | [train.py:85](train.py#L85) |
| 評估時為何不計算梯度？ | 省記憶體也加速（用 `set_grad_enabled(training)` 達成） | [train.py:74](train.py#L74) |
| 為什麼要 early stopping？ | 驗證分數太久沒進步就停，避免浪費時間與持續 overfitting | [train.py:291-293](train.py#L291-L293) |
| ReduceLROnPlateau 做什麼？ | val loss 連續 3 個 epoch 沒進步就把 lr 減半，收斂時用小步伐微調 | [train.py:244](train.py#L244)、[train.py:256](train.py#L256) |
| precision / recall / F1 / macro-F1？ | precision：說是 A 的裡面真的是 A 的比例；recall：真的是 A 的裡面被找出來的比例；F1 是兩者調和平均；macro-F1 是各類 F1 直接平均 | [train.py:51-54](train.py#L51-L54)、[train.py:56-57](train.py#L56-L57) |
| 混淆矩陣怎麼看？ | 縱軸真實類別、橫軸預測類別，對角線是猜對，其餘是錯成哪一類 | [train.py:30-38](train.py#L30-L38) |
| 預測類別怎麼來？ | logits 取 `argmax`，分數最高的類別 | [train.py:95](train.py#L95) |
| 為什麼分母用 `np.maximum(..., 1)`？ | 避免某類沒有樣本時除以 0 | [train.py:52-54](train.py#L52-L54) |
| 每次訓練的結果存哪？ | 獨立的時間戳資料夾，不覆蓋舊實驗；同時存 `config.json` | [train.py:199-202](train.py#L199-L202) |

---

## 六、存取權重（model.py）

| 問題 | 答案 | 程式碼 |
|---|---|---|
| 存什麼？ | model_state、arch、class_names、image_size、dropout | [model.py:56-68](model.py#L56-L68) |
| 為什麼連 class_names 和 image_size 一起存？ | 推論只靠一個檔案，前處理才不會和訓練不一致 | [model.py:57-60](model.py#L57-L60) |
| 權重為什麼先 `.cpu()`？ | 存檔不綁定 GPU，在沒有 GPU 的機器也能載入 | [model.py:62](model.py#L62) |
| `weights_only=True`？ | 避免 pickle 載入時執行夾帶的程式碼 | [model.py:76-77](model.py#L76-L77) |
| 如何相容舊權重檔？ | 沒有 `model_state` 鍵就視為舊版，arch 設 resnet18，並要求傳入 class_names | [model.py:78-87](model.py#L78-L87) |
| 載入後為何呼叫 `.eval()`？ | 切到推論模式，關閉 Dropout | [model.py:91](model.py#L91) |

---

## 七、結果與除錯

- **train loss 降、val loss 升：** overfitting → 加 Dropout、資料增強、早停、增加資料
- **怎麼判斷好壞：** 在 test 集看 accuracy、confusion matrix、各類 precision/recall（[train.py:305-309](train.py#L305-L309)）
- **你的結果：** `merged_resnet50_v2`，bad→good 誤判 49、good→bad 誤判 42
- **新照片表現變差：** 光線、背景、相機不同造成 domain shift
- **壞豆偵測為何 recall 比 accuracy 重要？** 漏掉壞豆（bad→good）的代價通常高於誤殺好豆
- **learning rate 太大/太小？** 太大會震盪甚至發散；太小收斂很慢、容易卡住（目前 1e-4，[train.py:171](train.py#L171)）
- **怎麼用假資料檢查模型形狀？** 直接執行 `python model.py`，輸出應為 (2, 18)（[model.py:95-102](model.py#L95-L102)）

---

## 答不出來時的句型

「我的理解是……，細節我再確認。」
