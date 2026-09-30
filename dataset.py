import json
import os
from pathlib import Path

from PIL import Image  # 用來開啟、讀取圖片檔案
from torch.utils.data import Dataset  # PyTorch 規定的「資料集」基底類別
from torchvision import transforms  # 圖片前處理工具（縮放、轉張量、正規化等）

# 專案根目錄：所有預設路徑都以這裡為基準，不管從哪個資料夾執行程式都能找到檔案
PROJECT_ROOT = Path(__file__).resolve().parent

# 只接受這些副檔名的圖片，避免資料夾裡混到其他無關檔案
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ImageNet 資料集算出來的平均值/標準差。
# 因為 backbone 是用同樣方式正規化的資料訓練出來的，輸入格式要一致模型才看得懂。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_class_names(data_dir):
    """
    類別名稱 = data_dir 底下每一個子資料夾的名字，例如 data/train/Broken、data/train/Cut ...
    用 sorted() 排序是為了確保每次執行「類別名稱 -> 數字編號」的對應關係都一致，
    否則今天訓練時 Broken=0，明天推論時 Broken 卻變成 3，模型就會亂猜。
    """
    return sorted([
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ])


def is_image_file(file_name):
    return str(file_name).lower().endswith(IMAGE_EXTENSIONS)


def list_images(folder):
    """回傳資料夾（含子資料夾）底下所有圖片的路徑，已排序。"""
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and is_image_file(p.name))


def build_transform(image_size, train):
    """
    訓練與推論共用同一個函式建立前處理，確保兩邊的 resize 大小、正規化數值永遠一致。
    """
    if train:
        # 訓練用：故意加入隨機變化（資料增強 / augmentation），
        # 讓模型看到「同一顆豆子的不同角度、大小、亮度版本」，避免死記照片、提升泛化能力。
        # 咖啡豆拍照沒有固定方向，所以上下翻轉、任意旋轉都是合理的變化。
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),  # 隨機裁切一部分再縮放
            transforms.RandomHorizontalFlip(),                                            # 隨機左右翻轉
            transforms.RandomVerticalFlip(),                                              # 隨機上下翻轉
            transforms.RandomRotation(degrees=30),                                        # 隨機旋轉 ±30 度
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),  # 隨機調整亮度/對比/飽和度/色調
            transforms.ToTensor(),                                                        # 圖片轉成 PyTorch 看得懂的張量 (Tensor)
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),         # 隨機遮住一小塊，逼模型不能只靠單一局部特徵
        ])

    # 驗證/測試/推論用：不做隨機變化，每次看同一張圖都應該得到一樣的結果
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_split(split_file):
    """
    讀取 make_split.py 產生的切分檔，回傳 (class_names, {"train": [...], "val": [...], "test": [...]})。
    每筆樣本是 (圖片絕對路徑, 標籤數字)。切分檔裡存的是相對專案根目錄的路徑，換電腦也能用。
    """
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    class_names = data["class_names"]
    splits = {
        name: [(str(PROJECT_ROOT / path), label) for path, label in samples]
        for name, samples in data["splits"].items()
    }
    return class_names, splits


class ImageListDataset(Dataset):
    """
    給 PyTorch 用的資料集類別，負責「每次要拿一張圖片時，去哪裡拿、怎麼處理」。
    samples 是 [(圖片路徑, 標籤數字), ...]，由 load_split() 提供。
    """

    def __init__(self, samples, transform):
        if not samples:
            raise ValueError("Dataset is empty")
        self.samples = samples
        self.transform = transform

    def __len__(self):
        # PyTorch 需要知道這個資料集總共有幾筆資料
        return len(self.samples)

    def __getitem__(self, idx):
        # PyTorch 訓練時會不斷呼叫這個方法，用索引 idx 拿「第幾筆資料」
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # 統一轉成 RGB，避免灰階或帶透明通道的圖片格式不一致
        return self.transform(image), label  # 回傳「處理好的圖片張量」和「正確答案(數字標籤)」

    def targets(self):
        return [label for _, label in self.samples]
