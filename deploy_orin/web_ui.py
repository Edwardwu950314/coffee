#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""咖啡豆分選機網頁介面（只用標準函式庫，不需額外安裝套件）。

bean_classifier.py 啟動時會呼叫 start_web_ui()，在背景提供：
  /              操作畫面（即時影像、統計、模型切換、門檻、辨識紀錄）
  /stream        MJPEG 即時影像
  /api/state     目前狀態（JSON，畫面每 0.5 秒輪詢一次）
  /thumb/<seq>   某筆辨識紀錄的豆子縮圖
  POST /api/...  method / pause / save / reset / threshold

Orin 螢幕開瀏覽器連 http://localhost:8000；手機或筆電連 http://<Orin 的 IP>:8000。
本模組不 import bean_classifier（避免循環），所需功能都由 controls 物件提供。
相容 Python 3.6。
"""

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

STREAM_FPS = 15
JPEG_QUALITY = 70
THUMB_SIZE = 96
RECENT_LIMIT = 30
RATE_WINDOW_S = 60


class Stats:
    """辨識統計與最近的辨識紀錄（辨識執行緒寫入，網頁執行緒讀取）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._clear()

    def _clear(self):
        self.good = 0
        self.bad = 0
        self.not_found = 0  # 收到 S 但畫面裡找不到豆子（回傳 BAD）
        self.errors = 0     # 辨識發生例外（回傳 BAD）
        self.recent = deque(maxlen=RECENT_LIMIT)
        self.times = deque()
        self.seq = 0

    def reset(self):
        with self._lock:
            self._clear()

    def add(self, bean_id, result, p_bad, image=None, elapsed_ms=None, note=""):
        """image 是 PIL 方形豆子圖（可為 None）。"""
        thumb = None
        if image is not None:
            try:
                small = np.asarray(image.convert("RGB").resize((THUMB_SIZE, THUMB_SIZE)))
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                thumb = buf.tobytes() if ok else None
            except Exception:
                thumb = None
        now = time.time()
        with self._lock:
            self.seq += 1
            if result == "GOOD":
                self.good += 1
            else:
                self.bad += 1
            if note == "not_found":
                self.not_found += 1
            elif note == "error":
                self.errors += 1
            self.times.append(now)
            self.recent.appendleft({"seq": self.seq, "bean_id": bean_id, "result": result, "p_bad": p_bad,
                                    "ms": elapsed_ms, "note": note, "time": now, "thumb": thumb})

    def thumb(self, seq):
        with self._lock:
            for item in self.recent:
                if item["seq"] == seq:
                    return item["thumb"]
        return None

    def snapshot(self):
        now = time.time()
        with self._lock:
            while self.times and now - self.times[0] > RATE_WINDOW_S:
                self.times.popleft()
            total = self.good + self.bad
            return {
                "good": self.good, "bad": self.bad, "total": total,
                "bad_rate": (self.bad / total) if total else 0.0,
                "per_minute": len(self.times) * 60.0 / RATE_WINDOW_S,
                "not_found": self.not_found, "errors": self.errors,
                "recent": [{"seq": r["seq"], "bean_id": r["bean_id"], "result": r["result"], "p_bad": r["p_bad"],
                            "ms": r["ms"], "note": r["note"], "time": r["time"], "has_thumb": r["thumb"] is not None}
                           for r in self.recent],
            }


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(app):
    """app 需提供：stats、frame()、state_info()、handle_key(key)、set_threshold(v)。"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # 不要把每個請求都印到終端機
            pass

        def _send(self, code, body, content_type):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/api/state":
                state = app.state_info()
                state["stats"] = app.stats.snapshot()
                state["server_time"] = time.time()
                self._json(state)
            elif path.startswith("/thumb/"):
                try:
                    data = app.stats.thumb(int(path[len("/thumb/"):]))
                except ValueError:
                    data = None
                if data is None:
                    self._send(404, "not found", "text/plain")
                else:
                    self._send(200, data, "image/jpeg")
            elif path == "/stream":
                self._stream()
            else:
                self._send(404, "not found", "text/plain")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            delay = 1.0 / STREAM_FPS
            try:
                while True:
                    frame = app.frame()
                    if frame is not None:
                        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        if ok:
                            data = buf.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                             str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass  # 瀏覽器關掉分頁

        def do_POST(self):
            path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            except ValueError:
                body = {}
            if path == "/api/method":
                key = {"water": "1", "sun": "2", "honey": "3"}.get(body.get("method"))
                if key is None:
                    return self._json({"ok": False, "error": "unknown method"}, 400)
                # 載入模型要幾秒，丟到背景去做，畫面用 state 裡的 loading 顯示進度
                threading.Thread(target=app.handle_key, args=(key,), daemon=True).start()
            elif path == "/api/pause":
                app.handle_key(" ")
            elif path == "/api/save":
                app.handle_key("s")
            elif path == "/api/reset":
                # 電腦版（web_app）還要一併重建追蹤器讓編號歸零；Orin 版的編號由 Arduino 決定，只清統計
                (getattr(app, "reset", None) or app.stats.reset)()
            elif path == "/api/threshold":
                try:
                    app.set_threshold(float(body.get("value")))
                except (TypeError, ValueError):
                    return self._json({"ok": False, "error": "bad value"}, 400)
            else:
                return self._json({"ok": False, "error": "not found"}, 404)
            self._json({"ok": True})

    return Handler


def start_web_ui(app, port=8000, host="0.0.0.0"):
    """在背景執行緒啟動網頁伺服器；埠被占用等失敗時印訊息並回傳 None，不影響辨識。"""
    try:
        server = _Server((host, port), _make_handler(app))
    except OSError as error:
        print("無法啟動網頁介面（port {}）：{}".format(port, error))
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("網頁介面：http://localhost:{}（同網路的裝置請連 http://<本機 IP>:{}）".format(port, port))
    return server


PAGE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>咖啡豆分選機</title>
<style>
:root{
  --bg:#14110f; --panel:#1f1b18; --panel2:#2a2521; --line:#3a332d; --text:#f2ece6; --muted:#a39a90;
  --good:#3fb950; --bad:#f85149; --warn:#d29922; --accent:#c8915a;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);
  font-family:"Noto Sans TC","Microsoft JhengHei",system-ui,-apple-system,sans-serif;font-size:16px}
header{display:flex;align-items:center;gap:16px;padding:12px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{margin:0;font-size:20px;font-weight:700;letter-spacing:.05em;flex:1;min-width:160px}
.lights{display:flex;gap:14px;flex-wrap:wrap}
.light{display:flex;align-items:center;gap:6px;font-size:14px;color:var(--muted)}
.dot{width:12px;height:12px;border-radius:50%;background:var(--bad)}
.dot.on{background:var(--good)} .dot.wait{background:var(--warn)}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(300px,2fr);gap:16px;padding:16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.panel h2{margin:0 0 10px;font-size:14px;font-weight:600;color:var(--muted);letter-spacing:.08em}
.video{position:relative;background:#000;border-radius:8px;overflow:hidden;aspect-ratio:4/3}
.video img{width:100%;height:100%;object-fit:contain;display:block}
.badge{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.65);padding:4px 10px;border-radius:6px;font-size:14px}
.counts{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.count{background:var(--panel2);border-radius:10px;padding:12px;text-align:center}
.count .n{font-size:44px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.count .l{font-size:14px;color:var(--muted)}
.count.good .n{color:var(--good)} .count.bad .n{color:var(--bad)}
.meta{display:flex;justify-content:space-between;margin-top:10px;font-size:15px;color:var(--muted)}
.meta b{color:var(--text);font-variant-numeric:tabular-nums}
.bar{height:8px;border-radius:4px;background:var(--good);margin-top:10px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--bad);width:0;transition:width .3s}
.seg{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
button{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  min-height:52px;padding:8px 12px;cursor:pointer;touch-action:manipulation}
button:hover{border-color:var(--accent)}
button:active{transform:scale(.98)}
button.active{background:var(--accent);color:#1a1209;border-color:var(--accent);font-weight:700}
button:disabled{opacity:.55;cursor:wait}
button:focus-visible,input:focus-visible{outline:3px solid #7cb7ff;outline-offset:2px}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}
.stack{display:flex;flex-direction:column;gap:16px}
.slider{display:flex;align-items:center;gap:12px}
.slider input{flex:1;height:32px;accent-color:var(--accent)}
.slider output{min-width:48px;text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
.hint{font-size:13px;color:var(--muted);margin-top:6px}
.log{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;grid-column:1/-1}
.item{background:var(--panel2);border-radius:10px;overflow:hidden;border:2px solid var(--line)}
.item.GOOD{border-color:var(--good)} .item.BAD{border-color:var(--bad)}
.item img,.item .noimg{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#000}
.noimg{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:13px;text-align:center;padding:6px}
.item .t{padding:6px 8px;font-size:13px;display:flex;justify-content:space-between;gap:4px}
.item .r{font-weight:700} .item.GOOD .r{color:var(--good)} .item.BAD .r{color:var(--bad)}
.empty{color:var(--muted);padding:20px;text-align:center;grid-column:1/-1}
#toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:var(--panel2);border:1px solid var(--line);
  padding:10px 18px;border-radius:10px;opacity:0;transition:opacity .2s;pointer-events:none}
#toast.show{opacity:1}
.offline{display:none;background:var(--bad);color:#fff;text-align:center;padding:8px;font-weight:600}
</style>
</head>
<body>
<div class="offline" id="offline" role="alert">與主機的連線中斷，重新連線中…</div>
<header>
  <h1>☕ 咖啡豆分選機</h1>
  <div class="lights" aria-live="polite">
    <span class="light"><i class="dot" id="l-cam"></i>相機</span>
    <span class="light"><i class="dot" id="l-ard"></i>Arduino</span>
    <span class="light"><i class="dot" id="l-model"></i><span id="l-model-t">模型</span></span>
  </div>
</header>
<main>
  <section class="panel">
    <h2>即時影像</h2>
    <div class="video">
      <img id="stream" alt="即時影像" src="/stream">
      <div class="badge" id="badge">—</div>
    </div>
    <div class="hint" id="hint-video">框的顏色：黃＝尚未辨識、綠＝好豆、紅＝壞豆；框上數字是感應器編號。</div>
  </section>

  <div class="stack">
    <section class="panel">
      <h2>統計（目前處理法：<span id="stat-m">—</span>）</h2>
      <div class="counts">
        <div class="count good"><div class="n" id="n-good">0</div><div class="l">好豆 GOOD</div></div>
        <div class="count bad"><div class="n" id="n-bad">0</div><div class="l">壞豆 BAD</div></div>
      </div>
      <div class="bar" aria-hidden="true"><i id="bar"></i></div>
      <div class="meta"><span>壞豆率 <b id="rate">0%</b></span><span>總計 <b id="total">0</b> 顆</span><span><b id="pm">0</b> 顆/分</span></div>
      <div class="meta" id="meta2"><span>找不到豆子 <b id="nf">0</b></span><span>辨識錯誤 <b id="er">0</b></span></div>
    </section>

    <section class="panel">
      <h2>處理法（模型）</h2>
      <div class="seg" role="group" aria-label="處理法">
        <button data-m="water">水洗</button>
        <button data-m="sun">日曬</button>
        <button data-m="honey">蜜處理</button>
      </div>
    </section>

    <section class="panel">
      <h2>壞豆門檻</h2>
      <div class="slider">
        <input id="th" type="range" min="0.05" max="0.95" step="0.05" value="0.5" aria-label="壞豆門檻">
        <output id="th-v">0.50</output>
      </div>
      <div class="hint">壞豆機率 ≥ 門檻就判 BAD。調低＝寧可錯殺也不漏抓；只影響之後的豆子。</div>
    </section>

    <section class="panel">
      <h2>控制</h2>
      <div class="row">
        <button id="b-pause">⏸ 暫停畫面</button>
        <button id="b-save">📷 存圖</button>
        <button id="b-reset">↺ 重置計數</button>
      </div>
      <div class="hint">暫停只凍結畫面，辨識與 Arduino 通訊不受影響。</div>
    </section>
  </div>

  <section class="panel log-wrap" style="grid-column:1/-1">
    <h2>最近辨識紀錄</h2>
    <div class="log" id="log"><div class="empty">尚無紀錄</div></div>
  </section>
</main>
<div id="toast" role="status"></div>

<script>
const $ = id => document.getElementById(id);
const NAMES = {water:"水洗", sun:"日曬", honey:"蜜處理"};
let thDragging = false, lastSeq = "", fails = 0, curMethod = "";

function toast(msg){ const t=$("toast"); t.textContent=msg; t.classList.add("show"); clearTimeout(toast.h); toast.h=setTimeout(()=>t.classList.remove("show"),1800); }
async function post(path, body){
  try{ const r = await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})}); return r.ok; }
  catch(e){ toast("送出失敗"); return false; }
}
function light(id, state){ $(id).className = "dot " + state; }

function render(s){
  const st = s.stats;
  $("n-good").textContent = st.good; $("n-bad").textContent = st.bad; $("total").textContent = st.total;
  $("rate").textContent = Math.round(st.bad_rate*100) + "%";
  $("bar").style.width = (st.bad_rate*100) + "%";
  $("pm").textContent = st.per_minute.toFixed(1);
  $("nf").textContent = st.not_found; $("er").textContent = st.errors;

  const solo = s.mode === "standalone";  // 電腦版：沒有 Arduino、沒有「收到 S」的流程
  $("l-ard").parentElement.style.display = solo ? "none" : "";
  $("meta2").style.display = solo ? "none" : "";
  if(s.hint) $("hint-video").textContent = s.hint;
  light("l-cam", s.camera_ok ? "on" : "");
  light("l-ard", s.arduino_ok ? "on" : "");
  light("l-model", s.loading ? "wait" : "on");
  $("l-model-t").textContent = s.loading ? "模型載入中…" : "模型 " + (NAMES[s.method]||s.method);
  $("badge").textContent = (NAMES[s.method]||s.method) + (s.paused ? "　⏸ 已暫停" : "") + (s.loading ? "　載入中…" : "");

  document.querySelectorAll("[data-m]").forEach(b=>{
    b.classList.toggle("active", b.dataset.m === s.method);
    b.disabled = !!s.loading;
    const n = s.counts ? s.counts[b.dataset.m] : null;  // 各處理法各自的累計顆數
    b.textContent = NAMES[b.dataset.m] + (n == null ? "" : "（" + n + "）");
  });
  curMethod = s.method;
  $("stat-m").textContent = NAMES[s.method] || s.method;
  $("b-pause").textContent = s.paused ? "▶ 繼續畫面" : "⏸ 暫停畫面";
  if(!thDragging){ $("th").value = s.threshold; $("th-v").textContent = s.threshold.toFixed(2); }

  const top = s.method + ":" + (st.recent.length ? st.recent[0].seq : 0);  // 換處理法時紀錄也要重畫
  if(top !== lastSeq){
    lastSeq = top;
    const log = $("log");
    if(!st.recent.length){ log.innerHTML = '<div class="empty">尚無紀錄</div>'; return; }
    log.innerHTML = st.recent.map(r=>{
      const img = r.has_thumb ? `<img src="/thumb/${r.seq}" alt="豆子 ${r.bean_id}">`
        : `<div class="noimg">${r.note==="not_found"?"畫面找不到豆子":"無影像"}</div>`;
      const p = r.p_bad==null ? "" : Math.round(r.p_bad*100)+"%";
      return `<div class="item ${r.result}">${img}<div class="t"><span>#${r.bean_id}</span><span class="r">${r.result}</span><span>${p}</span></div></div>`;
    }).join("");
  }
}

async function poll(){
  try{
    const r = await fetch("/api/state",{cache:"no-store"});
    render(await r.json());
    if(fails>0){ $("stream").src = "/stream?" + Date.now(); }  // 斷線後恢復，重連影像
    fails = 0; $("offline").style.display = "none";
  }catch(e){ if(++fails>=3) $("offline").style.display = "block"; }
  setTimeout(poll, 500);
}

document.querySelectorAll("[data-m]").forEach(b=> b.onclick = async ()=>{ await post("/api/method",{method:b.dataset.m}); toast("切換到 " + NAMES[b.dataset.m]); });
$("b-pause").onclick = ()=> post("/api/pause");
$("b-save").onclick = async ()=>{ if(await post("/api/save")) toast("已存圖到主機"); };
$("b-reset").onclick = async ()=>{ if(confirm("確定要把「" + (NAMES[curMethod]||curMethod) + "」的計數歸零嗎？（其他處理法不受影響）")){ await post("/api/reset"); lastSeq=""; toast("已重置 " + (NAMES[curMethod]||curMethod)); } };
$("th").oninput = e=>{ thDragging = true; $("th-v").textContent = (+e.target.value).toFixed(2); };
$("th").onchange = async e=>{ await post("/api/threshold",{value:+e.target.value}); thDragging = false; };
poll();
</script>
</body>
</html>
"""
