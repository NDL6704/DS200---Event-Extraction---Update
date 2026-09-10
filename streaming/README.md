# Module Streaming Thoi Gian Thuc - Event Extraction

Module này hiện thực hóa pipeline xử lý luồng (stream processing) thời gian thực cho bài toán **Trích xuất Sự kiện Tin tức Chính trị Việt Nam (Vietnamese Political Events Extraction)** sử dụng **Apache Kafka**, **Docker Compose**, và mô hình **Joint Learning (XLM-RoBERTa OneIE-based)**.

---

## 1. Kien Truc He Thong (Pipeline Architecture)

```
[ data/dataset.json ]
        │
        ▼
┌───────────────────────┐
│ streaming/producer.py │  ──(Moi 2-3s push 1 bai bao, UTF-8)──►  [ Kafka: political_news_raw ]
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
[ Terminal Consumer Log ] ◄──────(Ban ghi su kien JSON)─────────  [ Kafka: political_news_extracted ]
```

---

## 2. Cau Truc Thu Muc Lien Quan

```
DS200/
├── data/
│   └── dataset.json                # 5.814 bai bao mau dau vao cho producer
├── models/
│   ├── model.py                    # Kien truc OneIE JointModel (XLM-RoBERTa + 3 heads)
│   ├── checkpoints/
│   │   └── best_model.pt           # Checkpoint trong so mo hinh da huan luyen
│   └── encoders/                   # 3 bo ma hoa nhan OneIE:
│       ├── trigger_encoder.pkl     # 26 nhan BIO cho Trigger
│       ├── arg_encoder.pkl         # 7 nhan BIO cho Argument
│       └── event_encoder.pkl       # 14 nhan Event Type
├── streaming/
│   ├── producer.py                 # Doc data/dataset.json & phat vao 'political_news_raw'
│   ├── extractor.py                # Engine suy luan OneIE ket noi models/
│   ├── consumer.py                 # Lang nghe tin tuc, trich xuat & day sang 'political_news_extracted'
│   ├── __init__.py
│   └── README.md                   # Huong dan van hanh streaming
├── docker-compose.yml              # Dung Kafka broker (Port 9092) va Zookeeper (Port 2181)
└── requirements.txt                # Danh sach dependencies chung
```

---

## 3. Huong Dan Van Hanh Tung Buoc (Step-by-Step)

### Buoc 1: Cai dat thu vien moi truong
Tu thu muc goc du an, cai dat cac dependency:
```bash
pip install -r requirements.txt
```

### Buoc 2: Khoi dong Kafka Cluster voi Docker
Mo terminal tai thu muc goc du an (`DS200`) va thuc hien:
```bash
docker compose up -d
```
Kiem tra trang thai cac container dang chay:
```bash
docker compose ps
```
He thong se chay 2 container nhe va toi uu tai nguyen:
- `kafka_zookeeper` (Port `2181`)
- `kafka_broker` (Port `9092`)

### Buoc 3: Mo Terminal 1 — Khoi chay Consumer
Chay Consumer de san sang don nhan du lieu va thuc hien suy luan mo hinh:
```bash
python streaming/consumer.py
```
> Log man hinh se hien thi truc tiep theo dung dinh dang Event Extraction va cap nhat lien tuc theo thoi gian thuc:
> ```json
> --- Processed: Ban hanh vi tri viec lam cua dai bieu Quoc hoi... ---
> {
>   "event_type": "Legislation",
>   "entities": [
>     {
>       "token": "Uy",
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
>       "token": "hanh",
>       "trigger_tag": "I-Legislation",
>       "argument_tag": "O"
>     }
>   ]
> }
> [Kafka Routing] In: P0@O3 -> Out: P0@O3 (284.0ms)
> --------------------------------------------------
> ```

### Buoc 4: Mo Terminal 2 — Khoi chay Producer
Mo mot terminal moi song song va chay Producer:
```bash
python streaming/producer.py
```
> Log man hinh se hien thi:
> - Doc tu dong cac bai bao tu `dataset.json`.
> - Gui lan luot tung bai bao theo chu ky ngau nhien tu 2 den 3 giay.
> - In chi tiet ID bai bao, Partition, Offset gui thanh cong.

---

## 4. Cau Truc Du Lieu (Payload Schema)

### 1. Du lieu dau vao: `political_news_raw`
```json
{
  "article_id": "art_20240906_0001",
  "url": "https://baochinhphu.vn/...",
  "title": "Quoc hoi hop phien toan the thong qua Nghi quyet moi",
  "summary": "Chieu 20/6, tai Hoi truong Ba Dinh, Quoc hoi da bieu quyet thong qua Nghi quyet...",
  "publish_date": "20/06/2024 16:40",
  "published_at": "2026-09-10T12:00:00.000000"
}
```

### 2. Du lieu dau ra: `political_news_extracted`
```json
{
  "article_id": "art_20240906_0001",
  "source_url": "https://baochinhphu.vn/...",
  "title": "Quoc hoi hop phien toan the thong qua Nghi quyet moi",
  "original_summary": "Chieu 20/6, tai Hoi truong Ba Dinh, Quoc hoi da bieu quyet thong qua Nghi quyet...",
  "publish_date": "20/06/2024 16:40",
  "processed_at": "2026-09-10T12:00:02.150000",
  "processing_time_ms": 128.5,
  "extracted_event": {
    "event_type": "Legislation",
    "triggers": [
      {
        "phrase": "thong qua",
        "type": "Legislation"
      }
    ],
    "arguments": {
      "subject": [
        "Quoc hoi"
      ],
      "location": [
        "Hoi truong Ba Dinh"
      ],
      "time": [
        "Chieu 20/6"
      ]
    },
    "entities": [
      {
        "token": "thong",
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

## 5. Cac Tuy Chon Tham So Nang Cao

### Tuy chon Producer (`streaming/producer.py`)
```bash
# Doi tan suat gui (vi du: moi 1 den 1.5 giay)
python streaming/producer.py --min-interval 1.0 --max-interval 1.5

# Chi gui 20 bai bao roi dung
python streaming/producer.py --limit 20

# Chi dinh file du lieu khac
python streaming/producer.py --data-path custom_news.json
```

### Tuy chon Consumer (`streaming/consumer.py`)
```bash
# Doc lai tu dau toan bo cac bai bao da co trong topic
python streaming/consumer.py --from-beginning

# Gioi han xu ly 10 bai bao roi dung
python streaming/consumer.py --max-messages 10
```

---

## 6. Dung He Thong (Graceful Shutdown)
- Trong terminal Producer hoac Consumer, nhan to hop phim **`Ctrl + C`**. 
- He thong se kich hoat **Graceful Shutdown**: tu dong xa het cac tin con ton trong buffer, cam ket offset cuoi cung va dong ket noi an toan.
- Khi khong su dung nua, tat ha tang Docker:
  ```bash
  docker compose down
  ```
  *(De xoa ca volume du lieu cu: `docker compose down -v`)*
