import cv2
import threading
import queue
import time
import os
import requests
import numpy as np
from collections import deque
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CẤU HÌNH — Chỉnh tại đây
# ─────────────────────────────────────────────
# Đường dẫn model TFLite của bạn
MODEL_PATH   = r"D:\1. STUDY\3. NCKH 2526\4. SRC\yolo26-tflite\pdisease1_int8.tflite"
CAMERA_INDEX = 1                

# Cấu hình Telegram
TELEGRAM_BOT_TOKEN = '8687455390:AAGZrx6wPya2kSpQGyRG2B9VTd59JHZ30Rg'
TELEGRAM_CHAT_ID   = '7846403532'
COOLDOWN_TIME      = 15         # Giây giữa 2 lần gửi tin
FOLDER_LOG         = "nhat_ky_benh"

# Thông số Camera & AI
CAP_WIDTH    = 640
CAP_HEIGHT   = 480
IMGSZ        = 640              # TFLite int8 bắt buộc 640
CONF_THRESH  = 0.5
MAX_DET      = 5                
INITIAL_SKIP = 3                
TARGET_FPS   = 20               
# ─────────────────────────────────────────────

os.makedirs(FOLDER_LOG, exist_ok=True)

def send_telegram_alert(image_path, caption):
    """Hàm gửi tin nhắn Telegram (chạy độc lập để không gây lag loop chính)"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as image_file:
            files = {"photo": image_file}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            requests.post(url, files=files, data=data, timeout=5)
    except Exception as e:
        print(f"[Lỗi Telegram] {e}")

class CameraReader(threading.Thread):
    def __init__(self, source, width, height):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret: self.frame = frame

    def read(self):
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

class InferenceWorker(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self.in_q = queue.Queue(maxsize=1)
        self.out_q = queue.Queue(maxsize=1)
        self.running = True
        self.last_sent_time = 0

    def run(self):
        while self.running:
            try:
                frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            results = self.model(frame, imgsz=640, conf=CONF_THRESH, max_det=MAX_DET, verbose=False)
            result = results[0]
            annotated = result.plot()
            detections = result.boxes

            # Logic xử lý Telegram ngay trong luồng AI
            if len(detections) > 0:
                curr_time = time.time()
                if curr_time - self.last_sent_time > COOLDOWN_TIME:
                    class_id = int(detections.cls[0].item())
                    class_name = self.model.names[class_id]
                    
                    # Lưu ảnh và gửi (dùng threading cho request để không treo AI)
                    fname = f"detect_{int(curr_time)}.jpg"
                    fpath = os.path.join(FOLDER_LOG, fname)
                    cv2.imwrite(fpath, annotated)
                    
                    msg = f"🚨 Phát hiện: {class_name}\n📍 ID: {class_id}"
                    threading.Thread(target=send_telegram_alert, args=(fpath, msg)).start()
                    self.last_sent_time = curr_time

            if self.out_q.full(): self.out_q.get_nowait()
            self.out_q.put((annotated, len(detections)))

    def submit(self, frame):
        if not self.in_q.full(): self.in_q.put_nowait(frame)

    def get_result(self):
        return self.out_q.get_nowait() if not self.out_q.empty() else None

def main():
    print(f"⏳ Loading: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")
    
    cam = CameraReader(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
    worker = InferenceWorker(model)
    cam.start()
    worker.start()

    fps_deque = deque(maxlen=30)
    last_time = time.perf_counter()
    last_annotated, last_num_boxes = None, 0
    frame_count, skip_frames, adjust_counter = 0, INITIAL_SKIP, 0

    while True:
        frame = cam.read()
        if frame is None: continue

        frame_count += 1
        if frame_count % (skip_frames + 1) == 0:
            worker.submit(frame)

        res = worker.get_result()
        if res: last_annotated, last_num_boxes = res

        # Tính FPS
        now = time.perf_counter()
        fps_deque.append(1.0 / max(now - last_time, 1e-6))
        last_time = now
        fps_curr = sum(fps_deque) / len(fps_deque)

        # Adaptive skip logic
        adjust_counter += 1
        if adjust_counter >= 60:
            adjust_counter = 0
            if fps_curr < TARGET_FPS * 0.7: skip_frames = min(skip_frames + 1, 8)
            elif fps_curr > TARGET_FPS * 1.1: skip_frames = max(skip_frames - 1, 0)

        # Hiển thị
        disp = last_annotated if last_annotated is not None else frame
        cv2.putText(disp, f"FPS: {fps_curr:.1f} | Skip: {skip_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("AMR SMART MONITOR - TFLite", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()