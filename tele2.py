import cv2
import time
import os
import requests
from ultralytics import YOLO

# ================= CẤU HÌNH TELEGRAM =================
TELEGRAM_BOT_TOKEN = '8687455390:AAGZrx6wPya2kSpQGyRG2B9VTd59JHZ30Rg'
TELEGRAM_CHAT_ID = '7846403532'

def send_telegram_alert(image_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as image_file:
            files = {"photo": image_file}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            requests.post(url, files=files, data=data, timeout=5)
    except Exception as e:
        print(f"[Lỗi Telegram] {e}")

# =====================================================

def main():
    folder_lich_su = "nhat_ky_benh"
    os.makedirs(folder_lich_su, exist_ok=True)
    
    # --- THÔNG SỐ TỐI ƯU ---
    SKIP_COUNT = 5        # Lược bỏ 5 frame, xử lý frame thứ 6
    COOLDOWN_TIME = 15 
    last_sent_time = 0

    print("[Thông báo] Đang tải mô hình...")
    model = YOLO(r"D:\1. STUDY\3. NCKH 2526\4. SRC\yolo26-tflite\pdisease1_int8.tflite") 

    # Khởi tạo Camera với backend phù hợp
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Tắt buffer của camera để tránh bị trễ hình ảnh cũ (Rất quan trọng)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[Thông báo] Bắt đầu luồng video. Nhấn 'q' để thoát.")
    
    while True:
        start_time = time.time()

        # 1. Bỏ qua n-frame nhanh bằng .grab() 
        # (Chỉ lấy tín hiệu, không giải mã điểm ảnh -> Cực nhanh)
        for _ in range(SKIP_COUNT):
            cap.grab()

        # 2. Đọc frame thực tế để xử lý
        ret, frame = cap.retrieve() # Giải mã frame hiện tại sau khi đã grab
        if not ret:
            break

        # 3. Dự đoán với stream=True để tối ưu bộ nhớ
        results = model.predict(source=frame, conf=0.5, verbose=False, stream=True)
        
        for r in results:
            annotated_frame = r.plot()
            detections = r.boxes

            if len(detections) > 0:
                current_time = time.time()
                if current_time - last_sent_time > COOLDOWN_TIME:
                    class_id = int(detections.cls[0].item())
                    class_name = model.names[class_id]
                    
                    ten_file = f"phat_hien_{int(current_time)}.jpg"
                    duong_dan_luu = os.path.join(folder_lich_su, ten_file)
                    cv2.imwrite(duong_dan_luu, annotated_frame)
                    
                    send_telegram_alert(duong_dan_luu, f"🚨 Phát hiện: {class_name}")
                    last_sent_time = current_time

        # 4. Tính toán FPS thực tế
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 5. Hiển thị
        cv2.imshow("AMR Camera - Optimized", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()