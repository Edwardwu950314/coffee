"""找豆子與追蹤（webcam.py 與 bean_classifier.py 共用）。"""
import cv2
import numpy as np
from PIL import Image


def find_beans(frame, min_area_ratio=0.0003, max_area_ratio=0.05):
    """
    依顏色找出畫面中每一顆生豆，回傳 [(l, t, r, b), ...]（原圖座標）。
    生豆是黃綠 / 橄欖色（HSV 色相約 10~45），白紙與灰色金屬飽和度低、藍色零件色相不同，所以能被排除。
    豆子互相碰在一起時，用距離轉換 + watershed 切開。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 45, 25), (45, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(cv2.dilate(mask, np.ones((3, 3), np.uint8)), sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0
    markers = cv2.watershed(frame, markers)

    area = frame.shape[0] * frame.shape[1]
    boxes = []
    for label in range(2, n + 1):
        ys, xs = np.nonzero((markers == label) & (mask > 0))
        if len(xs) < area * min_area_ratio or len(xs) > area * max_area_ratio:
            continue
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        if box[0] <= 1 or box[1] <= 1 or box[2] >= frame.shape[1] - 1 or box[3] >= frame.shape[0] - 1:
            continue  # 碰到畫面邊緣的豆子只拍到一半，不判斷
        w, h = box[2] - box[0], box[3] - box[1]
        if max(w, h) > 2.2 * min(w, h):
            continue  # 豆子約是橢圓形，太細長的多半是反光或邊緣
        fill = len(xs) / (w * h)
        if fill < 0.55:
            continue  # 豆子填滿外框的比例很高；太低代表是碎片或不規則的雜物
        boxes.append(box)
    if len(boxes) >= 3:
        # 好幾顆豆子黏在一起會被當成一塊：比同畫面中位數大太多的就捨棄
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        median = float(np.median(areas))
        boxes = [b for b, a in zip(boxes, areas) if a <= 2.0 * median]
    return boxes


def crop_square(pil, box, size, margin=1.5):
    """與訓練資料相同的裁法（見 prepare_dataset.crop_bean）：以豆子為中心、邊長 = 長邊 x 1.5，縮成 size x size。"""
    w, h = pil.size
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = min(max(max(box[2] - box[0], box[3] - box[1]) * margin, size), min(w, h))
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    return pil.crop((int(left), int(top), int(left + side), int(top + side))).resize((size, size), Image.LANCZOS)


class Track:
    """一顆被追蹤的豆子：位置、速度、以及累積的壞豆機率。"""

    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box
        self.velocity = (0.0, 0.0)  # 每次更新移動的像素數（輸送帶上大致固定）
        self.probs = []
        self.missed = 0
        self.bean_id = None  # 感應器編號；None 代表還沒被認領（還沒對應到任何一次 S）
        self.p_bad_result = None  # 認領時辨識出的壞豆機率
        self.result = None   # 認領後的辨識結果 "GOOD" / "BAD"

    @property
    def center(self):
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def size(self):
        return max(self.box[2] - self.box[0], self.box[3] - self.box[1])

    @property
    def p_bad(self):
        return float(np.mean(self.probs)) if self.probs else 0.0

    def predicted_center(self):
        cx, cy = self.center
        return cx + self.velocity[0], cy + self.velocity[1]


class Tracker:
    """
    以「預測位置 + 最近距離」配對前後兩次偵測，讓同一顆豆子在輸送帶上移動時維持同一個編號。
    配對成功的豆子沿用編號並更新速度；沒配到的新框給新編號；連續 max_missed 次沒出現的編號就刪除。
    """

    def __init__(self, max_missed=3, gate=1.2):
        self.tracks = []
        self.next_id = 1
        self.max_missed = max_missed
        self.gate = gate  # 配對距離上限 = gate x 豆子邊長
        self.total = 0    # 累計看過幾顆不同的豆子

    def update(self, boxes):
        """boxes: [(l, t, r, b), ...]。回傳 [(track, 這次是否為新偵測到的框), ...]，順序與 boxes 對應。"""
        pairs = sorted(
            (np.hypot(t.predicted_center()[0] - (b[0] + b[2]) / 2, t.predicted_center()[1] - (b[1] + b[3]) / 2), ti, bi)
            for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes))
        used_t, matched = set(), {}
        for dist, ti, bi in pairs:
            if ti in used_t or bi in matched:
                continue
            if dist <= self.gate * max(self.tracks[ti].size, boxes[bi][2] - boxes[bi][0]):
                used_t.add(ti)
                matched[bi] = self.tracks[ti]

        for bi, track in matched.items():
            old = track.center
            track.box = boxes[bi]
            new = track.center
            track.velocity = (0.5 * track.velocity[0] + 0.5 * (new[0] - old[0]),
                              0.5 * track.velocity[1] + 0.5 * (new[1] - old[1]))
            track.missed = 0
        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        out = []
        for bi, box in enumerate(boxes):
            if bi not in matched:
                matched[bi] = Track(self.next_id, box)
                self.next_id += 1
                self.total += 1
                self.tracks.append(matched[bi])
            out.append(matched[bi])
        return out
