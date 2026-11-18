#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""咖啡豆辨識網頁版（電腦用，不需要 Orin / Arduino / 序列埠）。

用電腦 webcam、手機串流或影片檔當輸入，持續找出畫面裡的豆子並追蹤，每顆豆子累積
--samples 次辨識後就定案並計入統計（同一顆不會重複計算）。操作畫面在瀏覽器：

    python web_app.py                              # 手機串流（同 webcam.py 的預設網址）
    python web_app.py --source 0                   # 電腦 webcam
    python web_app.py --source "2026-10-21 21-10-19.mp4"   # 影片檔（播完會重頭播）
    python web_app.py --lan                        # 讓同網路的手機 / 筆電也能連

網頁畫面與 bean_classifier.py（Orin 版）共用 web_ui.py。
"""

import argparse
import os
import threading
import time
import webbrowser

import cv2
from PIL import Image

from beans import Tracker, crop_square, find_beans
from predict import CROP_SIZE, METHODS, Predictor
from web_ui import Stats, start_web_ui

ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
METHOD_KEYS = {"1": 0, "2": 1, "3": 2}  # 與網頁介面約定的按鍵：1 水洗 / 2 日曬 / 3 蜜處理


class BeanApp:
    """相機讀取、豆子偵測 / 追蹤 / 辨識、畫面標註；提供 web_ui 需要的介面。"""

    def __init__(self, args):
        self.args = args
        self.stats = Stats()
        self.threshold = args.bad_threshold
        self.method = args.method
        self.loading = False
        self.paused = False
        self.mode = "standalone"
        self.hint = "框上的 #編號 是追蹤編號、百分比是壞豆機率；每顆累積 {} 次辨識後定案並計入統計。".format(args.samples)

        self._predictors = {}
        self._get_predictor(self.method)

        source = args.source
        self._is_file = os.path.isfile(source)
        if self._is_file:
            self._cap = cv2.VideoCapture(source)
        elif source.isdigit():
            self._cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise SystemExit("無法開啟相機 / 影片：{}（用 --source 指定，例如 0 或影片路徑）".format(source))

        self._lock = threading.Lock()
        self._frame = None      # 最新的原始畫面
        self._frozen = None     # 暫停時凍結的已標註畫面
        self._results = []      # 最近一次辨識的 [(box, p_bad, track_id, final_result)]
        self._tracker = Tracker(max_missed=args.max_missed)
        self._busy = threading.Lock()
        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    # ---- 模型 ----
    def _get_predictor(self, method):
        if method not in self._predictors:
            self._predictors[method] = Predictor(method)
        return self._predictors[method]

    # ---- 背景執行緒 ----
    def _read_loop(self):
        fps = self._cap.get(cv2.CAP_PROP_FPS) if self._is_file else 0
        delay = 1.0 / fps if fps and fps > 1 else 1.0 / 30
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if self._is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 影片播完就重頭播
                else:
                    time.sleep(0.05)
                continue
            if self.args.rotate:
                frame = cv2.rotate(frame, ROTATIONS[self.args.rotate])
            with self._lock:
                self._frame = frame
            if self._is_file:
                time.sleep(delay)

    def _detect_loop(self):
        samples = self.args.samples
        while self._running:
            started = time.time()
            if self.paused or self.loading:
                time.sleep(0.05)
                continue
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes = find_beans(frame)
                tracks = self._tracker.update(boxes)
                # 只對還沒累積夠次數的豆子跑模型；累積夠了就沿用平均值，不重複辨識
                todo = [(t, b) for t, b in zip(tracks, boxes) if t.result is None]
                if todo:
                    crops = [crop_square(pil, box, CROP_SIZE) for _, box in todo]
                    outputs = self._get_predictor(self.method).predict(crops, crop=False)
                    for (t, _), crop, (_, _, p_bad) in zip(todo, crops, outputs):
                        t.probs.append(p_bad)
                        t.crop = crop
                        if len(t.probs) >= samples:
                            t.result = "BAD" if t.p_bad >= self.threshold else "GOOD"
                            t.p_bad_result = t.p_bad
                            self.stats.add(t.id, t.result, t.p_bad, crop)
                self._results = [(t.box, t.p_bad, t.id, t.result) for t in tracks]
            except Exception as error:  # 單次失敗不要讓整個辨識停掉
                print("辨識發生錯誤：{}".format(error))
            time.sleep(max(0.0, self.args.interval - (time.time() - started)))

    # ---- web_ui 需要的介面 ----
    def _annotate(self, frame):
        for box, p_bad, track_id, result in self._results:
            is_bad = (result == "BAD") if result else (p_bad >= self.threshold)
            color = (0, 0, 255) if is_bad else (0, 200, 0)
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if result else 1)
            cv2.putText(frame, "#{} {:.0%}".format(track_id, p_bad), (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def frame(self):
        if self.paused and self._frozen is not None:
            return self._frozen.copy()
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        return None if frame is None else self._annotate(frame)

    def state_info(self):
        return {"method": self.method, "loading": self.loading, "paused": self.paused,
                "threshold": self.threshold, "camera_ok": self._frame is not None,
                "arduino_ok": None, "mode": self.mode, "hint": self.hint}

    def set_threshold(self, value):
        self.threshold = min(max(value, 0.0), 1.0)

    def handle_key(self, key):
        if key == " ":
            if self.paused:
                self.paused, self._frozen = False, None
            else:
                self._frozen = self.frame()
                self.paused = True
        elif key == "s":
            frame = self.frame()
            if frame is not None:
                path = "capture_{}.png".format(int(time.time()))
                cv2.imwrite(path, frame)
                print("saved", path)
        elif key in METHOD_KEYS:
            method = METHODS[METHOD_KEYS[key]]
            if method == self.method or not self._busy.acquire(blocking=False):
                return
            self.loading = True
            try:
                self._get_predictor(method)
                self.method = method
                self._tracker = Tracker(max_missed=self.args.max_missed)  # 換模型後豆子重新辨識
                self._results = []
                print("處理法：{}".format(method))
            except FileNotFoundError as error:
                print(error)
            finally:
                self.loading = False
                self._busy.release()

    def close(self):
        self._running = False
        self._cap.release()


def main():
    parser = argparse.ArgumentParser(description="咖啡豆辨識網頁版（不需要 Orin / Arduino）")
    parser.add_argument("--method", choices=METHODS, default="water", help="起始處理法")
    parser.add_argument("--source", default="http://192.168.1.146:8080/video",
                        help="電腦 webcam 編號（0, 1, ...）、手機串流網址，或影片檔路徑")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="畫面順時針旋轉幾度")
    parser.add_argument("--bad-threshold", type=float, default=0.5, help="壞豆機率 >= 此值判 BAD（網頁上可即時調整）")
    parser.add_argument("--interval", type=float, default=0.3, help="每次辨識的間隔秒數")
    parser.add_argument("--samples", type=int, default=5, help="每顆豆子累積幾次辨識後定案")
    parser.add_argument("--max-missed", type=int, default=3, help="豆子連續幾次沒偵測到就丟掉它的編號")
    parser.add_argument("--port", type=int, default=8000, help="網頁埠號")
    parser.add_argument("--lan", action="store_true", help="允許同網路的其他裝置連線（預設只有本機）")
    parser.add_argument("--no-browser", action="store_true", help="不要自動開啟瀏覽器")
    args = parser.parse_args()

    app = BeanApp(args)
    server = start_web_ui(app, args.port, "0.0.0.0" if args.lan else "127.0.0.1")
    if server is None:
        raise SystemExit("網頁介面啟動失敗，請換一個 --port")
    if not args.no_browser:
        webbrowser.open("http://localhost:{}".format(args.port))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
