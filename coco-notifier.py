import requests

def send_telegram_alert(photo_path, ten_benh, cach_khac_phuc):
    bot_token = '8687455390:AAGZrx6wPya2kSpQGyRG2B9VTd59JHZ30Rg'  
    chat_id = '7846403532'
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    # Tạo nội dung tin nhắn đẹp mắt với Emoji
    tin_nhan = f"🚨 PHÁT HIỆN SÂU BỆNH!\n\n"
    tin_nhan += f"🦠 Tên bệnh: {ten_benh}\n"
    tin_nhan += f"💡 Cách khắc phục: {cach_khac_phuc}"
    
    try:
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            # Truyền tin_nhan vào biến caption
            data = {'chat_id': chat_id, 'caption': tin_nhan} 
            
            print("Đang gửi báo cáo chi tiết lên Telegram...")
            requests.post(url, files=files, data=data)
            print("✅ Đã gửi thành công!")
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy ảnh.")