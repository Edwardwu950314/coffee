import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from beans import Tracker, crop_square, find_beans
from predict import CROP_SIZE, METHODS, Predictor

METHOD_NAMES = {"water": "水洗", "sun": "日曬", "honey": "蜜處理"}
LABEL_NAMES = {"good": "好豆", "bad": "壞豆"}
# 微軟正黑體 / Linux 的 Noto CJK；OpenCV 的 putText 不支援中文所以改用 PIL 畫字
FONT_CANDIDATES = ["C:/Windows/Fonts/msjh.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]
FONT_PATH = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)


ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def draw_text(frame, text, xy, color, size=28):
    """在 OpenCV 影像（BGR）上畫中文字。"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(FONT_PATH, size) if FONT_PATH else ImageFont.load_default()
    ImageDraw.Draw(img).text(xy, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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
    parser.add_argument("--samples", type=int, default=5,
                        help="Predictions averaged per tracked bean; after this many it is not re-classified")
    parser.add_argument("--max-missed", type=int, default=3, help="Updates a bean may go undetected before its ID is dropped")
    parser.add_argument("--headless", action="store_true",
                        help="No window (for SSH): print counts and save the annotated frame to --out. Auto-enabled on Linux without DISPLAY")
    parser.add_argument("--out", default="latest.jpg", help="Headless: file the annotated frame is saved to")
    parser.add_argument("--save-every", type=float, default=1.0, help="Headless: seconds between saves")
    args = parser.parse_args()
    headless = args.headless or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))

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

    tracker = Tracker(max_missed=args.max_missed)
    last_time, last_saved, results = 0.0, 0.0, []
    if headless:
        print(f"Headless 模式：Ctrl+C 離開，標註影像存到 {args.out}")
    else:
        print("按鍵：1 水洗 / 2 日曬 / 3 蜜處理 / 空白鍵 暫停 / s 存圖 / q 離開")
    if not is_url and not headless:
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
            tracks = tracker.update(boxes)
            # 只對還沒累積夠次數的豆子跑模型；累積夠了就沿用平均值，同一顆不會被重複辨識
            todo = [(t, b) for t, b in zip(tracks, boxes) if len(t.probs) < args.samples]
            if todo:
                crops = [crop_square(pil, box, CROP_SIZE) for _, box in todo]
                for (t, _), (_, _, p_bad) in zip(todo, get_predictor(method).predict(crops, crop=False)):
                    t.probs.append(p_bad)
            results = [(t.box, t.p_bad, t.id) for t in tracks]
            last_time = now

        display = frame.copy()
        n_bad = 0
        for box, p_bad, track_id in results:
            is_bad = p_bad >= args.bad_threshold
            n_bad += is_bad
            color = (0, 0, 255) if is_bad else (0, 200, 0)
            cv2.rectangle(display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
            cv2.putText(display, f"#{track_id} {p_bad:.0%}", (int(box[0]), max(int(box[1]) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if results:
            display = draw_text(display, f"好豆 {len(results) - n_bad}　壞豆 {n_bad}　（#編號 壞豆信心度）　累計 {tracker.total} 顆", (10, 40), (255, 255, 255))
        else:
            display = draw_text(display, "未偵測到豆子", (10, 40), (200, 200, 200))
        focus_text = "手機串流" if is_url else ("自動對焦" if autofocus else f"手動對焦 {focus}")
        display = draw_text(display, f"處理法：{METHOD_NAMES[method]}　{focus_text}" + ("（暫停）" if paused else ""),
                            (10, 5), (255, 255, 0), size=22)
        if headless:
            if now - last_saved >= args.save_every:
                cv2.imwrite(args.out, display)
                print(f"好豆 {len(results) - n_bad}　壞豆 {n_bad}", flush=True)
                last_saved = now
            continue
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
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
