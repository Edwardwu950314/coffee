import torch
import torch.nn as nn
from torchvision import models  # torchvision 內建很多常見的模型架構，這裡用它拿現成的 ResNet18


class CoffeeBeanClassifier(nn.Module):
    """
    影像分類模型：輸入一張咖啡豆照片，輸出它屬於各個缺陷類別的分數。
    """

    def __init__(self, num_classes=18, pretrained=True, arch="resnet18", dropout=0.0):
        super().__init__()
        self.arch = arch

        # 1. 骨幹網路 (backbone)：負責「看懂圖片」
        #    訓練時使用已經在 ImageNet（上百萬張照片）訓練過的 ResNet18，
        #    等於借用別人已經訓練好的「眼睛」，不用從零開始學怎麼辨認邊緣、形狀、紋理。
        #    這就是「遷移學習 (transfer learning)」的概念。
        #    推論時設 pretrained=False：反正馬上會載入自己訓練好的權重，不需要再下載 ImageNet 權重（離線也能用）。
        weights = "DEFAULT" if pretrained else None
        if arch == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights)
            self.backbone = nn.Sequential(net.features, net.avgpool)
            feat_dim = net.classifier[1].in_features
        else:  # resnet18 / resnet50 ...
            net = getattr(models, arch)(weights=weights)
            # 拿掉最後那層 ImageNet 1000 類的分類層，只留下「抽取圖片特徵」的部分
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            feat_dim = net.fc.in_features

        # 2. 分類頭 (classifier head)：負責「猜是哪一類」
        #    backbone 輸出的是 512 個數字（圖片的特徵摘要），
        #    這裡用一個全新的線性層，把 512 個數字轉成 num_classes 個分數。
        #    這一層是模型裡「唯一需要從頭學」的部分，因為 ImageNet 沒教過咖啡豆缺陷長怎樣。
        #    前面加 Dropout：訓練時隨機丟掉一部分特徵，防止模型太依賴少數特徵而 overfitting。
        #    Dropout 沒有參數，所以不影響權重檔的相容性；推論（eval）時自動關閉。
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x 的形狀是 (批次大小, 3, 圖片高, 圖片寬)，3 代表 RGB 三個顏色通道

        features = self.backbone(x)
        # backbone 輸出形狀是 (批次大小, 512, 1, 1)，
        # 也就是每張圖被壓縮成 512 個數字，但外面還包著多餘的維度

        features = torch.flatten(features, 1)
        # 把 (批次大小, 512, 1, 1) 攤平成 (批次大小, 512)，
        # 這樣才能丟進下面的線性層做矩陣運算

        return self.classifier(self.dropout(features))
        # 回傳形狀 (批次大小, num_classes) 的分數 (logits)，
        # 分數最高的那個類別，就是模型認為最可能的答案


def save_checkpoint(path, model, class_names, image_size, **extra):
    """
    權重檔裡除了模型參數，也一起存「類別名稱」與「圖片大小」，
    這樣推論時只需要這一個檔案，不必再去掃描訓練資料夾，也不怕前處理設定跟訓練時不一致。
    """
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "arch": getattr(model, "arch", "resnet18"),
        "class_names": list(class_names),
        "image_size": image_size,
        **extra,
    }, path)


def load_checkpoint(path, device, class_names=None, image_size=224):
    """
    載入 save_checkpoint() 存的檔案，回傳 (model, class_names, image_size)。
    也相容舊版只存 state_dict 的權重檔：那種檔案沒有類別名稱，必須由呼叫端提供 class_names。
    """
    # weights_only=True 是比較安全的載入方式（不會執行檔案裡夾帶的任意程式碼）
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        class_names = ckpt["class_names"]
        image_size = ckpt.get("image_size", image_size)
    else:
        state_dict = ckpt
        if not class_names:
            raise ValueError(f"{path} is a legacy checkpoint without class names; please pass --class-dir.")

    model = CoffeeBeanClassifier(num_classes=len(class_names), pretrained=False,
                                 arch=ckpt.get("arch", "resnet18") if "model_state" in ckpt else "resnet18")
    model.load_state_dict(state_dict)  # 把訓練好的權重灌進模型架構裡
    model.to(device).eval()  # 推論模式
    return model, class_names, image_size


if __name__ == "__main__":
    # 這段只有直接執行 `python model.py` 時才會跑，
    # 用假資料快速檢查模型輸入輸出的形狀對不對，方便偵錯。
    mock_input = torch.randn(2, 3, 224, 224)  # 假裝有 2 張 224x224 的彩色圖片
    model = CoffeeBeanClassifier(num_classes=18, pretrained=False)
    logits = model(mock_input)
    print(f"input shape: {mock_input.shape}")
    print(f"output shape: {logits.shape}")  # 應該會是 (2, 18)：2張圖，各18個類別分數
