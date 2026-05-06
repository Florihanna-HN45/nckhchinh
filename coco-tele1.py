import cv2
import time
import os
import requests
from ultralytics import YOLO

# ================= CẤU HÌNH TELEGRAM =================
# Bạn cần thay thế bằng Token và Chat ID thực tế của bạn
TELEGRAM_BOT_TOKEN = '8687455390:AAGZrx6wPya2kSpQGyRG2B9VTd59JHZ30Rg'
TELEGRAM_CHAT_ID = '7846403532'

def send_telegram_alert(image_path, caption):
    """Gửi ảnh và tin nhắn cảnh báo qua Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as image_file:
            files = {"photo": image_file}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            response = requests.post(url, files=files, data=data)
            return response.json()
    except Exception as e:
        print(f"[Lỗi Telegram] Không thể gửi cảnh báo: {e}")
        return None
# =====================================================

def main():
    # 1. Khởi tạo thư mục và biến trạng thái
    folder_lich_su = "nhat_ky_benh"
    os.makedirs(folder_lich_su, exist_ok=True)
    
    last_sent_time = 0
    COOLDOWN_TIME = 15 # Thời gian chờ (giây) giữa 2 lần gửi tin nhắn
    FRAMES_TO_SKIP = 3 # Bỏ qua 3 frame, xử lý 1 frame để tăng hiệu năng
    frame_counter = 0

    # 2. Khởi tạo mô hình YOLOv8
    # Trên PC, nên dùng file .pt (PyTorch) thay vì .tflite để có độ chính xác và tốc độ tốt nhất
    print("[Thông báo] Đang tải mô hình YOLOv8...")
    model = YOLO(r"C:\Users\This pc\Downloads\mau.pt") 

    # 3. Khởi tạo Camera
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[Thông báo] Bắt đầu luồng video...")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[Lỗi] Không thể đọc từ camera.")
            break

        frame_counter += 1

        # 4. Xử lý bỏ qua khung hình (Frame Skipping)
        # Chỉ đưa vào model dự đoán nếu đến đúng frame cần xử lý
        if frame_counter % (FRAMES_TO_SKIP + 1) == 0:
            # model.predict tự động xử lý resize, chuẩn hóa, dự đoán và NMS
            # conf=0.5: Chỉ lấy các vật thể có độ tin cậy >= 50%
            results = model.predict(source=frame, conf=0.5, verbose=False)
            
            # Lấy frame đã được Ultralytics tự động vẽ sẵn bounding box và nhãn
            annotated_frame = results[0].plot()
            
            # Lấy danh sách các vật thể phát hiện được
            detections = results[0].boxes

            # 5. Logic gửi cảnh báo qua Telegram
            if len(detections) > 0:
                current_time = time.time()
                
                # Kiểm tra thời gian Cooldown
                if current_time - last_sent_time > COOLDOWN_TIME:
                    # Lấy ID của vật thể đầu tiên phát hiện được
                    class_id = int(detections.cls[0].item())
                    class_name = model.names[class_id] # Lấy tên class thay vì chỉ lấy ID
                    
                    # Lưu ảnh
                    ten_file = f"phat_hien_{int(current_time)}.jpg"
                    duong_dan_luu = os.path.join(folder_lich_su, ten_file)
                    cv2.imwrite(duong_dan_luu, annotated_frame)
                    
                    # Gửi Telegram
                    print(f"[Cảnh báo] Phát hiện {class_name}. Đang gửi Telegram...")
                    send_telegram_alert(
                        image_path=duong_dan_luu, 
                        caption=f"🚨 Phát hiện vật thể: {class_name} (ID: {class_id})\n📍 Yêu cầu kiểm tra ngay!"
                    )
                    
                    last_sent_time = current_time
                    
            frame_counter = 0 # Reset counter
        else:
            # Nếu đang ở frame bị skip, giữ nguyên frame cuối cùng có vẽ box (nếu muốn) 
            # hoặc trong trường hợp này, ta chỉ hiển thị frame gốc để tránh giật lag.
            annotated_frame = frame 

        # 6. Tính toán và hiển thị FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 7. Hiển thị Camera
        cv2.imshow("AMR Camera - Ultralytics YOLOv8", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()