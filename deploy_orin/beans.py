"""找豆子與追蹤（webcam.py 與 bean_classifier.py 共用）。"""
import cv2
import numpy as np
from PIL import Image


def find_beans(frame, min_area_ratio=0.0003, max_area_ratio=0.05, split_area_ratio=0.02):
    """
    依顏色找出畫面中每一顆生豆，回傳 [(l, t, r, b), ...]（原圖座標）。
    生豆是黃綠 / 橄欖色（HSV 色相約 10~45），白紙與灰色金屬飽和度低、藍色零件色相不同，所以能被排除。
    豆子互相碰在一起時，用距離轉換 + watershed 切開。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 25, 25), (45, 255, 255))  # 飽和度門檻放低，豆子陰影那一半（S≈40）才不會被漏掉
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    area = frame.shape[0] * frame.shape[1]
    # 一般大小的連通區塊直接當一顆豆子；只有特別大（可能黏在一起）的才用 watershed 切開，避免單顆被切成兩塊
    n_cc, cc = cv2.connectedComponents(mask)
    regions = []
    for label in range(1, n_cc):
        comp = (cc == label).astype(np.uint8) * 255
        if cv2.countNonZero(comp) <= area * split_area_ratio:
            regions.append(np.nonzero(comp))
            continue
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(cv2.dilate(comp, np.ones((3, 3), np.uint8)), sure_fg)
        n, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown > 0] = 0
        markers = cv2.watershed(frame, markers)
        for sub in range(2, n + 1):
            regions.append(np.nonzero((markers == sub) & (comp > 0)))

    boxes = []
    for ys, xs in regions:
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
    side = min(max(max(box[2] - box[0], box[3] - box[1]) * margin, 32), min(w, h))  # 不墊到 size：豆子佔裁切圖的比例要和訓練時一致
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    return pil.crop((int(left), int(top), int(left + side), int(top + side))).resize((size, size), Image.LANCZOS)


class Track:
    """一顆被追蹤的豆子：位置、速度、以及累積的壞豆機率。"""

    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box      # 平滑後的框，只用來顯示
        self.raw_box = box  # 最近一次實際偵測到的原始框，裁圖辨識用
        self.velocity = (0.0, 0.0)  # 每次更新移動的像素數（輸送帶上大致固定）
        self.probs = []
        self.missed = 0
        self.hits = 1  # 累計被偵測到幾次；太少代表可能只是雜訊，不顯示也不認領
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
        steps = self.missed + 1  # 漏偵測期間豆子仍在移動
        return cx + self.velocity[0] * steps, cy + self.velocity[1] * steps


class Tracker:
    """
    以「預測位置 + 最近距離」配對前後兩次偵測，讓同一顆豆子在輸送帶上移動時維持同一個編號。
    配對成功的豆子沿用編號並更新速度；沒配到的新框給新編號；連續 max_missed 次沒出現的編號就刪除。
    """

    def __init__(self, max_missed=8, gate=1.2, smooth=0.5):
        self.smooth = smooth  # 框的平滑係數：新偵測框的權重，越小越穩
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
            steps = track.missed + 1
            raw = boxes[bi]
            # 把舊框依速度推到現在的位置，再與新偵測框混合，避免框大小與位置逐幀抖動
            dx, dy = track.velocity[0] * steps, track.velocity[1] * steps
            prev = (track.box[0] + dx, track.box[1] + dy, track.box[2] + dx, track.box[3] + dy)
            a = self.smooth
            track.box = tuple(a * r + (1 - a) * p for r, p in zip(raw, prev))
            new = ((raw[0] + raw[2]) / 2, (raw[1] + raw[3]) / 2)
            track.velocity = (0.5 * track.velocity[0] + 0.5 * (new[0] - old[0]) / steps,
                              0.5 * track.velocity[1] + 0.5 * (new[1] - old[1]) / steps)
            track.raw_box = raw
            track.missed = 0
            track.hits += 1
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
