#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson Nano 咖啡豆辨識程式：收到 Arduino 的 S,編號 後拍照辨識，回傳 GOOD,編號 或 BAD,編號。

通訊格式與時間限制見 Jetson Nano 端程式需求文件。序列埠上只能送 GOOD,編號 / BAD,編號，
所以所有除錯訊息一律 print 到終端機。相容 Python 3.6（Jetson Nano 內建版本）。
"""

import os
import queue
import threading
import time

import cv2
import numpy as np
import serial
import torch
from PIL import Image

from dataset import PROJECT_ROOT, build_transform
from model import load_checkpoint
from prepare_dataset import crop_bean

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# USB 相機填編號（通常是 0）；CSI 相機請改成 GStreamer 字串並把 CAMERA_BACKEND 改成 cv2.CAP_GSTREAMER。
CAMERA_SOURCE = 0
CAMERA_BACKEND = cv2.CAP_ANY

# 模型：水洗 water / 日曬 sun / 蜜處理 honey，檔案放在 models/<method>/best.pth。
METHOD = "water"
# 壞豆機率 >= 這個值就判 BAD。調低 = 寧可錯殺也不漏抓壞豆。
BAD_THRESHOLD = 0.5
# 要與 prepare_dataset.py 的 --size 相同（訓練時裁出的豆子小圖大小）。
CROP_SIZE = 256

# 收到 S 之後延遲多久才取畫面。實測發現照片裡豆子還沒進到畫面中央時再調大。
CAPTURE_DELAY_S = 0.0

# 設成資料夾路徑（例如 "captures"）就會把每張辨識用的照片存下來，方便檢查拍照時機與蒐集訓練資料。
SAVE_DIR = None

# 從收到 S 到送出結果超過這個秒數就印出警告。
SLOW_WARNING_S = 1.0


class LatestFrameCamera:
    """背景執行緒持續讀取相機，隨時取得最新畫面，避免讀到相機緩衝區裡的舊畫面。"""

    def __init__(self, source, backend):
        self._capture = cv2.VideoCapture(source, backend)
        if not self._capture.isOpened():
            raise RuntimeError("無法開啟相機：{}".format(source))
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self._capture.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()


class ArduinoLink:
    """管理與 Arduino 的序列埠連線；讀取在主執行緒，寫入在辨識執行緒，所以寫入要上鎖。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._port = None

    def connect(self):
        while True:
            try:
                port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            except serial.SerialException as error:
                print("無法開啟 {}，2 秒後重試：{}".format(SERIAL_PORT, error))
                time.sleep(2)
                continue
            # 開啟序列埠會讓 Arduino Mega 自動重新開機，要等它開完再開始收資料。
            time.sleep(2)
            port.reset_input_buffer()
            with self._lock:
                self._port = port
            print("已連線 Arduino：{}".format(SERIAL_PORT))
            return port

    def disconnect(self):
        with self._lock:
            port = self._port
            self._port = None
        if port is not None:
            try:
                port.close()
            except serial.SerialException:
                pass

    def send(self, text):
        with self._lock:
            if self._port is None:
                return False
            try:
                self._port.write(text.encode("ascii"))
                return True
            except serial.SerialException as error:
                print("送出失敗：{}".format(error))
                return False


class BeanModel:
    """訓練好的好豆 / 壞豆分類模型，前處理與訓練（predict.py）完全相同。"""

    def __init__(self, method=METHOD):
        path = PROJECT_ROOT / "models" / method / "best.pth"
        if not path.exists():
            raise FileNotFoundError("找不到模型 {}，請先訓練：python train.py --method {}".format(path, method))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names, image_size = load_checkpoint(path, self.device)
        self.transform = build_transform(image_size, train=False)
        self.bad_index = self.class_names.index("bad")

    @torch.no_grad()
    def p_bad(self, frame):
        """frame 是 OpenCV 的 BGR 影像，回傳畫面中那顆豆子是壞豆的機率。"""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = crop_bean(image, CROP_SIZE)  # 與訓練相同：先在畫面中找出豆子並裁成方形
        x = self.transform(image).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0, self.bad_index].item()


def load_model():
    """程式啟動時只呼叫一次：載入 AI 模型，並先用一張影像跑一次推論暖機。"""
    model = BeanModel()
    model.p_bad(np.full((480, 640, 3), 200, dtype=np.uint8))  # 第一次推論特別慢（CUDA 初始化等），先跑掉
    print("模型已載入：{}（{}），裝置：{}".format(METHOD, model.class_names, model.device))
    return model


def classify_bean(model, frame):
    """只能回傳 "GOOD" 或 "BAD"。"""
    return "BAD" if model.p_bad(frame) >= BAD_THRESHOLD else "GOOD"


def parse_request(line):
    """解析 "S,12" 這種拍照通知，成功回傳編號，不是拍照通知就回傳 None。"""
    if not line.startswith("S,"):
        return None
    try:
        bean_id = int(line[2:])
    except ValueError:
        return None
    if 0 <= bean_id <= 65535:
        return bean_id
    return None


def classify_worker(model, jobs, link):
    while True:
        bean_id, received_at, frame = jobs.get()

        if frame is None:
            print("豆子 {}：沒有相機畫面，回傳 BAD".format(bean_id))
            result = "BAD"
        else:
            try:
                result = classify_bean(model, frame)
            except Exception as error:
                print("豆子 {}：辨識發生錯誤（{}），回傳 BAD".format(bean_id, error))
                result = "BAD"

        if result not in ("GOOD", "BAD"):
            print("豆子 {}：辨識結果 {!r} 不是 GOOD/BAD，回傳 BAD".format(bean_id, result))
            result = "BAD"

        sent = link.send("{},{}\n".format(result, bean_id))
        elapsed = time.monotonic() - received_at
        note = "（太慢！）" if elapsed > SLOW_WARNING_S else ""
        status = "" if sent else "（未送出）"
        print("豆子 {} -> {}，耗時 {:.0f} ms{}{}".format(bean_id, result, elapsed * 1000, note, status))

        if SAVE_DIR and frame is not None:
            filename = "{:05d}_{}_{}.jpg".format(bean_id, result, int(time.time() * 1000))
            cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)


def main():
    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)

    model = load_model()
    camera = LatestFrameCamera(CAMERA_SOURCE, CAMERA_BACKEND)
    jobs = queue.Queue()
    link = ArduinoLink()

    worker = threading.Thread(target=classify_worker, args=(model, jobs, link), daemon=True)
    worker.start()

    def capture(bean_id, received_at):
        jobs.put((bean_id, received_at, camera.read()))

    port = link.connect()
    buffer = b""
    try:
        while True:
            try:
                chunk = port.read(port.in_waiting or 1)
            except serial.SerialException as error:
                print("Arduino 斷線（{}），重新連線中…".format(error))
                link.disconnect()
                port = link.connect()
                buffer = b""
                continue

            if not chunk:
                continue
            buffer += chunk

            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("ascii", errors="ignore").strip()
                bean_id = parse_request(line)

                if bean_id is not None:
                    # 一收到 S 就取畫面，辨識交給另一個執行緒；
                    # 否則前一顆還在辨識時，這一顆的照片會拍得太晚、拍到別的位置。
                    received_at = time.monotonic()
                    if CAPTURE_DELAY_S > 0:
                        threading.Timer(CAPTURE_DELAY_S, capture, args=(bean_id, received_at)).start()
                    else:
                        capture(bean_id, received_at)
                elif line.startswith("ERR,"):
                    print("Arduino 回報：{}".format(line))
                elif line:
                    print("忽略無法辨識的訊息：{}".format(line))
    except KeyboardInterrupt:
        pass
    finally:
        link.disconnect()
        camera.close()


if __name__ == "__main__":
    main()
