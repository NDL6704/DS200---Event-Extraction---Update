# 🚀 Module Streaming Thời Gian Thực - Event Extraction

Module này hiện thực hóa pipeline xử lý luồng (stream processing) thời gian thực cho bài toán **Trích xuất Sự kiện Tin tức Chính trị Việt Nam (Vietnamese Political Events Extraction)** sử dụng **Apache Kafka**, **Docker Compose**, và mô hình **Joint Learning (XLM-RoBERTa OneIE-based)**.

---

## 🏛️ Kiến Trúc Hệ Thống (Pipeline Architecture)

```
[ dataset.json / Web Crawl ]
            │
            ▼
┌───────────────────────┐
│ streaming/producer.py │  ──(Mỗi 2-3s push 1 bài báo, UTF-8)──►  [ Kafka: political_news_raw ]
└───────────────────────┘                                                      │
                                                                               ▼
                                                                  ┌─────────────────────────┐
                                                                  │  streaming/consumer.py  │
                                                                  │ (news_event_extraction) │
                                                                  └─────────────────────────┘
                                                                               │
                                                                               ▼
                                                                  ┌─────────────────────────┐
                                                                  │   extractor.py (OneIE)  │
                                                                  │ - JointModel (XLM-R)    │
                                                                  │ - best_model.pt         │
                                                                  │ - oneie_encoders/       │
                                                                  └─────────────────────────┘
                                                                               │
                                                                               ▼
[ Kafka UI : http://localhost:8080 ]  ◄──(Bản ghi sự kiện JSON)──  [ Kafka: political_news_extracted ]
```

---

## 📁 Cấu Trúc Thư Mục Liên Quan

```
DS200/
├── data/
│   └── dataset.json                # 5.814 bài báo mẫu đầu vào cho producer
├── models/
│   ├── model.py                    # Kiến trúc OneIE JointModel (XLM-RoBERTa + 3 heads)
│   ├── checkpoints/
│   │   └── best_model.pt           # Checkpoint trọng số mô hình đã huấn luyện
│   └── encoders/                   # 3 bộ mã hóa nhãn OneIE:
│       ├── trigger_encoder.pkl     # 26 nhãn BIO cho Trigger
│       ├── arg_encoder.pkl         # 7 nhãn BIO cho Argument
│       └── event_encoder.pkl       # 14 nhãn Event Type
├── streaming/
│   ├── producer.py                 # Đọc data/dataset.json & phát vào 'political_news_raw'
│   ├── extractor.py                # Engine suy luận OneIE kết nối models/
│   ├── consumer.py                 # Lắng nghe tin tức, trích xuất & đẩy sang 'political_news_extracted'
│   ├── __init__.py
│   └── README.md                   # Hướng dẫn vận hành streaming
├── docker-compose.yml              # Dựng Kafka broker, Zookeeper và Kafka UI (Port 8080)
└── requirements.txt                # Danh sách dependencies chung
```

---

## 🛠️ Hướng Dẫn Vận Hành Từng Bước (Step-by-Step)

### Bước 1: Cài đặt thư viện môi trường
Từ thư mục gốc dự án, cài đặt các dependency:
```bash
pip install -r streaming/requirements.txt
```

### Bước 2: Khởi động Kafka Cluster & Kafka UI với Docker
Mở terminal tại thư mục gốc dự án (`DS200`) và thực hiện:
```bash
docker compose up -d
```
Kiểm tra trạng thái các container đang chạy:
```bash
docker compose ps
```
Bạn sẽ thấy 3 container:
- `kafka_zookeeper` (Port `2181`)
- `kafka_broker` (Port `9092`)
- `kafka_ui` (Port `8080`)

### Bước 3: Mở Terminal 1 — Khởi chạy Consumer
Chạy Consumer để sẵn sàng đón nhận dữ liệu và thực hiện suy luận mô hình:
```bash
python streaming/consumer.py
```
> **Log màn hình sẽ hiển thị trực tiếp theo đúng định dạng Event Extraction**:
> ```json
> --- Processed: Ban hành vị trí việc làm của đại biểu Quốc hội hoạ... ---
> {
>   "event_type": "Legislation",
>   "entities": [
>     {
>       "token": "Ủy",
>       "trigger_tag": "O",
>       "argument_tag": "B-Arg-Subject"
>     },
>     {
>       "token": "ban",
>       "trigger_tag": "O",
>       "argument_tag": "I-Arg-Subject"
>     },
>     {
>       "token": "ban",
>       "trigger_tag": "B-Legislation",
>       "argument_tag": "O"
>     },
>     {
>       "token": "hành",
>       "trigger_tag": "I-Legislation",
>       "argument_tag": "O"
>     }
>   ]
> }
> 📊 [Kafka Routing] In: P0@O3 ➡️ Out: P0@O3 (284.0ms)
> --------------------------------------------------
> ```

### Bước 4: Mở Terminal 2 — Khởi chạy Producer
Mở một terminal mới song song và chạy Producer:
```bash
python streaming/producer.py
```
> **Log màn hình sẽ hiển thị**:
> - Đọc tự động các bài báo từ `dataset.json`.
> - Gửi lần lượt từng bài báo theo chu kỳ ngẫu nhiên từ 2 đến 3 giây.
> - In chi tiết ID bài báo, Partition, Offset gửi thành công.

### Bước 5: Kiểm tra kết quả trực quan trên Kafka UI
Mở trình duyệt web và truy cập địa chỉ:
👉 **[http://localhost:8080](http://localhost:8080)**

Tại giao diện Kafka UI:
1. **Topics**:
   - `political_news_raw`: Xem các bài báo thô liên tục được đẩy vào từ Producer.
   - `political_news_extracted`: Xem các bản ghi sự kiện đã được Consumer phân tích và trích xuất.
2. **Consumers**:
   - Quan sát group `news_event_extraction_group` với tiến độ xử lý message lag = 0.
3. **Messages**:
   - Nhấp vào từng tin nhắn trong `political_news_extracted` để xem trực tiếp cấu trúc JSON chi tiết.

---

## 📋 Cấu Trúc Dữ Liệu (Payload Schema)

### 1. Dữ liệu đầu vào: `political_news_raw`
```json
{
  "article_id": "art_20240906_0001",
  "url": "https://baochinhphu.vn/...",
  "title": "Quốc hội họp phiên toàn thể thông qua Nghị quyết mới",
  "summary": "Chiều 20/6, tại Hội trường Ba Đình, Quốc hội đã biểu quyết thông qua Nghị quyết...",
  "publish_date": "20/06/2024 16:40",
  "published_at": "2026-09-10T12:00:00.000000"
}
```

### 2. Dữ liệu đầu ra: `political_news_extracted`
```json
{
  "article_id": "art_20240906_0001",
  "source_url": "https://baochinhphu.vn/...",
  "title": "Quốc hội họp phiên toàn thể thông qua Nghị quyết mới",
  "original_summary": "Chiều 20/6, tại Hội trường Ba Đình, Quốc hội đã biểu quyết thông qua Nghị quyết...",
  "publish_date": "20/06/2024 16:40",
  "processed_at": "2026-09-10T12:00:02.150000",
  "processing_time_ms": 128.5,
  "extracted_event": {
    "event_type": "Legislation",
    "triggers": [
      {
        "phrase": "thông qua",
        "type": "Legislation"
      }
    ],
    "arguments": {
      "subject": [
        "Quốc hội"
      ],
      "location": [
        "Hội trường Ba Đình"
      ],
      "time": [
        "Chiều 20/6"
      ]
    },
    "entities": [
      {
        "token": "thông",
        "trigger_tag": "B-Legislation",
        "argument_tag": "O"
      },
      {
        "token": "qua",
        "trigger_tag": "I-Legislation",
        "argument_tag": "O"
      }
    ]
  }
}
```

---

## ⚙️ Các Tùy Chọn Tham Số Nâng Cao

### Tùy chọn Producer (`streaming/producer.py`)
```bash
# Đổi tần suất gửi (ví dụ: mỗi 1 đến 1.5 giây)
python streaming/producer.py --min-interval 1.0 --max-interval 1.5

# Chỉ gửi 20 bài báo rồi dừng
python streaming/producer.py --limit 20

# Chỉ định file dữ liệu khác
python streaming/producer.py --data-path custom_news.json
```

### Tùy chọn Consumer (`streaming/consumer.py`)
```bash
# Đọc lại từ đầu toàn bộ các bài báo đã có trong topic
python streaming/consumer.py --from-beginning

# Giới hạn xử lý 10 bài báo rồi dừng
python streaming/consumer.py --max-messages 10
```

---

## 🛑 Dừng Hệ Thống (Graceful Shutdown)
- Trong terminal Producer hoặc Consumer, nhấn tổ hợp phím **`Ctrl + C`**. 
- Hệ thống sẽ kích hoạt **Graceful Shutdown**: tự động xả hết các tin còn tồn trong buffer, cam kết offset cuối cùng và đóng kết nối an toàn.
- Khi không sử dụng nữa, tắt hạ tầng Docker:
  ```bash
  docker compose down
  ```
  *(Để xóa cả volume dữ liệu cũ: `docker compose down -v`)*
