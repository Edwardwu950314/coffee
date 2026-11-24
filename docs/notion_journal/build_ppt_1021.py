from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor as C
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

ESP = C(0x2B, 0x1D, 0x14); AMB = C(0xC7, 0x7D, 0x2E); RED = C(0xB0, 0x3A, 0x2E)
INK = C(0x2B, 0x2B, 0x2B); MUT = C(0x6B, 0x6B, 0x6B); TINT = C(0xF3, 0xF0, 0xEC)
WH = C(255, 255, 255); LIGHT = C(0xE8, 0xDC, 0xCB)
FONT = "Microsoft JhengHei"
p = Presentation(); p.slide_width = Inches(13.333); p.slide_height = Inches(7.5)
blank = p.slide_layouts[6]


def bg(s, color):
    s.background.fill.solid(); s.background.fill.fore_color.rgb = color


def box(s, x, y, w, h, text, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, font=FONT, anchor=MSO_ANCHOR.TOP):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = t.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, l in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(6)
        r = para.add_run(); r.text = l; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold; r.font.name = font
    return t


def rect(s, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = fill; r.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: r.adjustments[0] = 0.06
    r.shadow.inherit = False
    return r


def slide(title, dark=False):
    s = p.slides.add_slide(blank); bg(s, ESP if dark else WH)
    box(s, 0.6, 0.4, 12, 0.8, title, 32, WH if dark else ESP, True)
    return s


def pic(s, f, x, y, w=None, h=None):
    return s.shapes.add_picture('img/' + f, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def code(s, x, y, w, h, text, size=12):
    rect(s, x, y, w, h, TINT)
    t = box(s, x + 0.25, y + 0.2, w - 0.5, h - 0.4, text.split('\n'), size, INK, font="Consolas")
    for para in t.text_frame.paragraphs: para.space_after = Pt(0)


def table(s, x, y, w, rows, colw, size=13, rowh=0.42, hl=None):
    sh = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(rowh * len(rows)))
    tb = sh.table
    for j, cw in enumerate(colw): tb.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tb.rows[i].height = Inches(rowh)
        for j, v in enumerate(row):
            c = tb.cell(i, j); c.text = str(v); c.margin_left = c.margin_right = Inches(0.1)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            f = c.text_frame.paragraphs[0].runs[0].font; f.size = Pt(size); f.name = FONT
            c.fill.solid()
            if i == 0: c.fill.fore_color.rgb = ESP; f.color.rgb = WH; f.bold = True
            elif hl == i: c.fill.fore_color.rgb = LIGHT; f.color.rgb = INK; f.bold = True
            else: c.fill.fore_color.rgb = TINT if i % 2 == 0 else WH; f.color.rgb = INK
    return tb



# 1 title
s = p.slides.add_slide(blank); bg(s, ESP)
box(s, 0.9, 2.2, 11.5, 1.2, "咖啡豆分選機", 54, WH, True)
box(s, 0.9, 3.5, 11.5, 0.8, "週工作進度報告：即時辨識與 Jetson 端程式", 30, LIGHT)
box(s, 0.9, 4.6, 11.5, 0.5, "2026/10/15 – 10/21　｜　報告人：吳泰羲", 18, AMB)

# 2 headline
s = slide("本週成果：從「離線模型」走到「即時辨識」")
stats = [("2", "支新程式", "webcam.py、bean_classifier.py"), ("6 顆", "同畫面逐顆偵測", "分散的豆子全部框出"),
         ("12 / 12", "原圖判對", "水洗 好豆 6、壞豆 6"), ("70~100 ms", "單顆辨識時間", "本機 GPU；Jetson 待測")]
for i, (n, l, sub) in enumerate(stats):
    x = 0.6 + i * 3.08; rect(s, x, 1.7, 2.85, 3.3, TINT)
    box(s, x + 0.2, 2.1, 2.45, 1, n, 38, AMB, True)
    box(s, x + 0.2, 3.4, 2.45, 0.5, l, 16, ESP, True)
    box(s, x + 0.2, 3.95, 2.45, 0.8, sub, 13, MUT)
box(s, 0.6, 5.5, 12.1, 1.2, ["沿用既有模型，本週沒有重新訓練。",
                             "注意：Jetson 端序列埠與相機流程尚未實機驗證；手機拍攝的判斷偏壞豆，原因待確認。"], 17, INK)

# 3 goals
s = slide("本週目標")
goals = ["寫一個用鏡頭即時辨識咖啡豆好壞的程式（先用電腦 webcam，再改手機鏡頭）",
         "依 Arduino 端提供的通訊規格，寫好 Jetson Nano 端的辨識程式"]
for i, t in enumerate(goals):
    y = 2.0 + i * 1.6; rect(s, 0.9, y, 0.7, 0.7, AMB, MSO_SHAPE.OVAL)
    box(s, 0.9, y, 0.7, 0.7, "✓", 22, WH, True, PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 1.9, y, 10.5, 0.9, t, 22, INK, anchor=MSO_ANCHOR.MIDDLE)
box(s, 0.9, 5.4, 11.5, 1, "兩項皆完成；Jetson 端因缺少實機，只驗證了辨識部分。", 18, MUT)

# 4 webcam.py
s = slide("即時辨識程式 webcam.py")
rows = [["項目", "說明"], ["模型", "沿用 predict.py 的 Predictor 與 models/<處理法>/best.pth，不需重訓"],
        ["顯示", "好豆綠框、壞豆紅框，框上標壞豆信心度 p_bad；上方顯示好 / 壞數量"],
        ["按鍵", "1 / 2 / 3 切換水洗 / 日曬 / 蜜處理；空白鍵暫停；s 存圖；q 離開"],
        ["效能", "每 0.3 秒推論一次（--interval），其餘時間只更新畫面，避免卡頓"],
        ["門檻", "--bad-threshold 可調，預設 0.5"]]
table(s, 0.6, 1.7, 12.1, rows, [1.8, 10.3], 16, 0.7)

# 5 phone
s = slide("改用手機鏡頭")
box(s, 0.6, 1.5, 6.0, 0.5, "為什麼：webcam 無法對焦", 20, RED, True)
box(s, 0.6, 2.1, 6.0, 2.5, ["多數 webcam 最近對焦距離約 10 ~ 15 公分，豆子太近會糊。",
                            "改用 Android「IP Webcam」App，程式讀取手機串流網址。"], 16, INK)
code(s, 6.9, 1.5, 5.85, 4.3, '''parser.add_argument("--source",
    default="http://192.168.1.146:8080/video")
parser.add_argument("--rotate", type=int,
    choices=(0, 90, 180, 270), default=90)

is_url = not args.source.isdigit()
cap = cv2.VideoCapture(args.source) if is_url \\
      else cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)

frame = cv2.rotate(frame, ROTATIONS[args.rotate])''', 13)
box(s, 0.6, 4.7, 6.0, 2, ["--source 同時接受鏡頭編號或網址；--rotate 預設順時針轉 90 度。", "仍保留 webcam 的對焦控制（f、[ ]、p）。"], 14, MUT)

# 6 real run
s = slide("實際執行畫面：逐顆偵測並標出壞豆信心度")
box(s, 0.6, 1.25, 4.7, 0.4, "t = 8 秒：好豆 6、壞豆 1", 15, ESP, True)
box(s, 5.45, 1.25, 4.7, 0.4, "t = 36 秒：好豆 1、壞豆 5", 15, RED, True)
pic(s, 'video_8s.png', 0.6, 1.75, h=4.4); pic(s, 'video_36s.png', 5.45, 1.75, h=4.4)
box(s, 10.45, 1.75, 2.4, 4.4, ["6 顆分散的豆子都被獨立框出，其中 2 顆相貼的也被 watershed 切開。",
                              "框上數字 = p_bad；紅框為壞豆，綠框為好豆。",
                              "舊做法（find_bean_box）假設畫面只有一顆豆子，在同類畫面一顆都找不到。"], 13, INK)
box(s, 0.6, 6.7, 12.1, 0.5, "來源：10/21 webcam.py 螢幕錄影（水洗模型、手機串流）", 12, MUT)

# 7 detect principle
s = slide("逐顆偵測的原理")
steps = [("1", "HSV 顏色", "色相 10~45 → 黃綠 / 橄欖色\n白紙、灰色金屬、藍色零件排除"),
         ("2", "形態學", "開運算去雜點\n閉運算補洞"),
         ("3", "Watershed", "距離轉換找每顆中心\n切開互相碰到的豆子"),
         ("4", "過濾", "碰邊緣、太細長、填滿比例低、\n面積過大者一律捨棄")]
for i, (n, t, d) in enumerate(steps):
    x = 0.6 + i * 3.08; rect(s, x, 1.7, 2.85, 3.2, TINT)
    rect(s, x + 0.2, 1.95, 0.6, 0.6, AMB, MSO_SHAPE.OVAL)
    box(s, x + 0.2, 1.95, 0.6, 0.6, n, 18, WH, True, PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, x + 0.2, 2.8, 2.45, 0.5, t, 20, ESP, True)
    box(s, x + 0.2, 3.4, 2.45, 1.4, d.split("\n"), 13, INK)
box(s, 0.6, 5.3, 12.1, 1.5, ["裁切方式與訓練資料相同：以豆子為中心、邊長 = 長邊 × 1.5、縮成 256，", "確保模型看到的輸入格式一致。"], 16, MUT)

# 8 code detect
s = slide("程式碼 ①　webcam.py：找豆子與裁切")
code(s, 0.6, 1.5, 6.4, 5.2, '''hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (10, 45, 25), (45, 255, 255))
mask = cv2.morphologyEx(mask, MORPH_OPEN, k5)
mask = cv2.morphologyEx(mask, MORPH_CLOSE, k9)

dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(
    dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
...
markers = cv2.watershed(frame, markers)

# 每個 label 取外框後過濾（前三種條件）
if touches_border(box):         continue
if max(w, h) > 2.2 * min(w, h): continue
if len(xs) / (w * h) < 0.55:    continue
# 第四種「面積過大」另外與同畫面中位數比較''', 12.5)
code(s, 7.3, 1.5, 5.45, 3.6, '''boxes = find_beans(frame)
crops = [crop_square(pil, box) for box in boxes]
preds = get_predictor(method).predict(
    crops, crop=False)   # 已經裁好
results = [(box, p_bad)
    for box, (_, _, p_bad) in zip(boxes, preds)]''', 12.5)
box(s, 7.3, 5.4, 5.45, 1.3, "crop_square 與 prepare_dataset.crop_bean 相同，訓練與推論輸入一致。", 14, MUT)

# 9 jetson overview
s = slide("Jetson Nano 端 bean_classifier.py：通訊流程")
flow = [("Arduino", "豆子經過感測器\n送出 S,編號"), ("Jetson", "收到 S 立刻取畫面\n交辨識執行緒"),
        ("模型", "裁出豆子 → ResNet18\np_bad ≥ 門檻 → BAD"), ("回傳", "GOOD,編號\n或 BAD,編號")]
for i, (t, d) in enumerate(flow):
    x = 0.6 + i * 3.08; rect(s, x, 1.8, 2.7, 2.2, TINT)
    box(s, x + 0.2, 2.0, 2.3, 0.5, t, 20, ESP, True)
    box(s, x + 0.2, 2.6, 2.3, 1.3, d.split("\n"), 14, INK)
    if i < 3: box(s, x + 2.72, 2.6, 0.35, 0.6, "→", 24, AMB, True)
pts = ["序列埠 /dev/ttyACM0、115200；相機背景執行緒；USB 斷線自動重連",
       "辨識另開執行緒：避免前一顆還在辨識，下一顆照片拍太晚",
       "辨識失敗一律回 BAD（寧可誤殺，不放過）",
       "只依賴 torch、torchvision、PIL、numpy；不用 predict.py（會匯入 matplotlib）"]
for i, t in enumerate(pts):
    box(s, 0.6, 4.4 + i * 0.6, 12.1, 0.5, "• " + t, 16, INK)

# 10 code jetson
s = slide("程式碼 ②　bean_classifier.py：模型與通訊")
code(s, 0.6, 1.5, 6.3, 5.2, '''@torch.no_grad()
def p_bad(self, frame):
    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = crop_bean(image, CROP_SIZE)
    x = self.transform(image).unsqueeze(0)
    return torch.softmax(self.model(x.to(self.device)),
        dim=1)[0, self.bad_index].item()

def classify_bean(model, frame):
    return "BAD" if model.p_bad(frame) \\
        >= BAD_THRESHOLD else "GOOD"''', 12.5)
code(s, 7.2, 1.5, 5.55, 5.2, '''def parse_request(line):
    if not line.startswith("S,"):
        return None
    try:
        bean_id = int(line[2:])
    except ValueError:
        return None
    return bean_id if 0 <= bean_id <= 65535 \\
        else None

# 主迴圈：收到 S 立刻取畫面
jobs.put((bean_id, received_at, camera.read()))
# 辨識失敗 -> BAD
link.send("{},{}\\n".format(result, bean_id))''', 12.5)

# 11 verify
s = slide("驗證結果")
rows = [["項目", "結果"], ["水洗模型 × 原圖", "好豆 6 張、壞豆 6 張，12 / 12 判對"],
        ["單顆辨識時間", "約 70 ~ 100 ms（本機 GPU）"],
        ["手機畫面偵測", "錄影畫面中 6 顆分散的豆子全部框出；金屬反光與豆堆中心被排除"],
        ["Jetson 實機（序列埠 / 相機 / 執行緒）", "尚未驗證：沒有 Arduino 與 Jetson 可測"]]
table(s, 0.6, 1.7, 12.1, rows, [4.2, 7.9], 16, 0.8, hl=4)

# 12 problems
s = slide("遇到的問題與檢討")
items = [("手機拍攝判斷偏壞豆（原因未明）", "錄影 t=8 秒 6 顆豆子 p_bad 為 19%~52%，一顆 52% 剛好超過門檻；靜態畫面中豆堆邊緣 3 顆達 78%~93% 且未被排除。不確定是豆子真有問題，或光線 / 色調與訓練照片不同，也可能受相鄰豆子影響。", RED),
         ("Jetson 相容性風險", "model.py 用 torch.load(weights_only=True)，需 PyTorch ≥ 1.13；JetPack 4.x 多為 Python 3.6 + 舊版 PyTorch，可能不支援。", RED),
         ("Jetson 耗時未知", "本機 70~100 ms，Nano 一定更慢，需與「D2→D3 行進時間 − 0.3 秒」比較。", ESP),
         ("偵測前提較嚴格", "需白紙背景、豆子分開；其他顏色物件、黏在一起、被邊緣切到都不判斷。", ESP)]
for i, (t, d, c) in enumerate(items):
    y = 1.5 + i * 1.35; rect(s, 0.6, y, 12.1, 1.2, TINT)
    box(s, 0.9, y + 0.12, 11.5, 0.4, t, 18, c, True)
    box(s, 0.9, y + 0.55, 11.5, 0.6, d, 14, INK)

# 13 next
s = slide("下週計劃（10/22 – 10/28）", dark=True)
plans = ["Jetson Nano 實機測試：先確認 PyTorch 版本與 weights_only 問題，再依 Arduino 端規格逐項驗收（序列埠、相機、辨識回傳）並量測耗時",
         "確認手機拍攝是否偏壞豆：已知好壞的豆子各數顆測試；若系統性偏壞，用手機照片補訓練資料或調門檻",
         "檢查輸送帶拍照時機與背景：用 SAVE_DIR 存下實際照片，必要時調整 CAPTURE_DELAY_S"]
for i, t in enumerate(plans):
    y = 1.8 + i * 1.7; rect(s, 0.9, y, 0.7, 0.7, AMB, MSO_SHAPE.OVAL)
    box(s, 0.9, y, 0.7, 0.7, str(i + 1), 20, WH, True, PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 1.9, y, 10.5, 1.2, t, 20, WH, anchor=MSO_ANCHOR.MIDDLE)
p.save('weekly_log_2026-10-21.pptx')
