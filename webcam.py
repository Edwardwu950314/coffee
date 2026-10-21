import argparse
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from predict import CROP_SIZE, METHODS, Predictor

METHOD_NAMES = {"water": "水洗", "sun": "日曬", "honey": "蜜處理"}
LABEL_NAMES = {"good": "好豆", "bad": "壞豆"}
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體，OpenCV 的 putText 不支援中文所以改用 PIL 畫字


ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def draw_text(frame, text, xy, color, size=28):
    """在 OpenCV 影像（BGR）上畫中文字。"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(xy, text, font=ImageFont.truetype(FONT_PATH, size), fill=color[::-1])
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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


def crop_square(pil, box, margin=1.5):
    """與訓練資料相同的裁法（見 prepare_dataset.crop_bean）：以豆子為中心、邊長 = 長邊 x 1.5，縮成 CROP_SIZE。"""
    w, h = pil.size
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = min(max(max(box[2] - box[0], box[3] - box[1]) * margin, CROP_SIZE), min(w, h))
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    return pil.crop((int(left), int(top), int(left + side), int(top + side))).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description="Real-time good / bad green bean detection from a webcam.")
    parser.add_argument("--method", choices=METHODS, default="water", help="Initial processing method")
    parser.add_argument("--source", default="http://192.168.1.146:8080/video",
                        help="Webcam index (0, 1, ...) or a phone stream URL, e.g. http://192.168.1.5:8080/video")
    parser.add_argument("--bad-threshold", type=float, default=0.5, help="p_bad above this is shown as bad")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between predictions")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=90,
                        help="Rotate the image clockwise by this many degrees")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--focus", type=int, default=None, help="Start in manual focus with this value (usually 0~255)")
    args = parser.parse_args()

    predictors = {}

    def get_predictor(method):
        if method not in predictors:
            predictors[method] = Predictor(method)
        return predictors[method]

    method = args.method
    get_predictor(method)

    is_url = not args.source.isdigit()
    if is_url:
        cap = cv2.VideoCapture(args.source)  # 手機串流：解析度與對焦都在手機 App 裡設定
    else:
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")

    autofocus = args.focus is None
    focus = args.focus if args.focus is not None else 100
    if not is_url:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if not autofocus:
            cap.set(cv2.CAP_PROP_FOCUS, focus)

    last_time, results = 0.0, []
    print("按鍵：1 水洗 / 2 日曬 / 3 蜜處理 / 空白鍵 暫停 / s 存圖 / q 離開")
    if not is_url:
        print("對焦：f 切換自動/手動對焦 / [ ] 手動調整焦距 / p 開啟相機驅動設定視窗")
    paused, frozen = False, None
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if args.rotate:
                frame = cv2.rotate(frame, ROTATIONS[args.rotate])
        else:
            frame = frozen.copy()

        now = time.time()
        if not paused and now - last_time >= args.interval:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = find_beans(frame)
            if boxes:
                crops = [crop_square(pil, box) for box in boxes]
                preds = get_predictor(method).predict(crops, crop=False)
                results = [(box, p_bad) for box, (_, _, p_bad) in zip(boxes, preds)]
            else:
                results = []
            last_time = now

        display = frame.copy()
        n_bad = 0
        for box, p_bad in results:
            is_bad = p_bad >= args.bad_threshold
            n_bad += is_bad
            color = (0, 0, 255) if is_bad else (0, 200, 0)
            cv2.rectangle(display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
            cv2.putText(display, f"{p_bad:.0%}", (int(box[0]), max(int(box[1]) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if results:
            display = draw_text(display, f"好豆 {len(results) - n_bad}　壞豆 {n_bad}　（框上數字 = 壞豆信心度）", (10, 40), (255, 255, 255))
        else:
            display = draw_text(display, "未偵測到豆子", (10, 40), (200, 200, 200))
        focus_text = "手機串流" if is_url else ("自動對焦" if autofocus else f"手動對焦 {focus}")
        display = draw_text(display, f"處理法：{METHOD_NAMES[method]}　{focus_text}" + ("（暫停）" if paused else ""),
                            (10, 5), (255, 255, 0), size=22)
        cv2.imshow("Coffee Bean Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused, frozen = not paused, frame
        elif key == ord("s"):
            path = f"capture_{int(time.time())}.png"
            cv2.imwrite(path, display)
            print("saved", path)
        elif is_url and key in (ord("f"), ord("["), ord("]"), ord("p")):
            pass  # 手機串流的對焦要在手機 App 裡操作
        elif key == ord("f"):
            autofocus = not autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            if not autofocus:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
        elif key in (ord("["), ord("]")):
            autofocus = False
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            focus = min(max(focus + (5 if key == ord("]") else -5), 0), 255)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
        elif key == ord("p"):
            cap.set(cv2.CAP_PROP_SETTINGS, 1)  # DirectShow 的相機設定視窗，裡面通常有對焦滑桿
        elif key in (ord("1"), ord("2"), ord("3")):
            new_method = METHODS[key - ord("1")]
            try:
                get_predictor(new_method)
                method, last_time = new_method, 0.0
            except FileNotFoundError as e:
                print(e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
