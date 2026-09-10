# Events Extraction from Vietnamese Political News

Dự án nghiên cứu và triển khai hệ thống Trích xuất Thông tin Sự kiện từ Tin tức Chính trị Việt Nam (thu thập từ baochinhphu.vn), bao gồm:
1. **Event Trigger Tagging**: Gán thẻ BIO từ kích hoạt sự kiện.
2. **Argument Detection**: Nhận diện thành phần tham gia sự kiện (Subject, Location, Time).
3. **Event Type Classification**: Phân loại 14 loại sự kiện chính trị ở mức câu.
4. **Real-time Kafka Streaming Pipeline**: Phát và trích xuất sự kiện thời gian thực qua Apache Kafka kết hợp mô hình OneIE Joint Learning.

---

## 1. Cấu Trúc Mô Hình Sử Dụng (Model Architecture)

Hệ thống áp dụng kiến trúc **Joint Learning (OneIE-based)** nhằm giải quyết đồng thời 3 bài toán trích xuất trên một mạng nơ-ron duy nhất, chia sẻ không gian biểu diễn ngữ nghĩa để tối ưu hóa hiệu năng và hạn chế hiện tượng lan truyền sai số (Error Propagation) của các mô hình dạng tuần tự (Pipeline).

```
                             [ Input Text / Tokens ]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   XLMRobertaTokenizerFast   │
                         │    (Subword Tokenization)   │
                         └─────────────────────────────┘
                                        │ (input_ids, attention_mask)
                                        ▼
                         ┌─────────────────────────────┐
                         │   XLM-RoBERTa (Backbone)    │
                         │      hidden_size = 768      │
                         └─────────────────────────────┘
                                   │         │
                 ┌─────────────────┘         └──────────────────┐
                 ▼ (Sequence Output: H)                         ▼ (CLS Token: h_CLS)
      ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
      │ Trigger Classifier  │   │ Argument Classifier │   │  Event Classifier   │
      │ Linear(768 -> 384)  │   │ Linear(768 -> 384)  │   │ Linear(768 -> 384)  │
      │ ReLU + Dropout(0.1) │   │ ReLU + Dropout(0.1) │   │ ReLU + Dropout(0.1) │
      │ Linear(384 -> 26)   │   │ Linear(384 -> 7)    │   │ Linear(384 -> 14)   │
      └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
                 │                         │                         │
                 ▼                         ▼                         ▼
         Trigger BIO Tags          Argument BIO Tags            Event Type
           (26 classes)               (7 classes)              (14 classes)
```

### Chi tiết các thành phần mô hình

1. **Backbone ngôn ngữ**:
   - Sử dụng pre-trained **`xlm-roberta-base`** làm bộ trích xuất đặc trưng ngữ cảnh đa ngôn ngữ mạnh mẽ cho tiếng Việt.
   - Đầu ra tầng ẩn cuối cùng (last hidden state) có kích thước: $\mathbf{H} \in \mathbb{R}^{B \times L \times 768}$, với $B$ là batch size, $L$ là chiều dài chuỗi token (`max_length = 128`).
   - Vector ngữ cảnh toàn câu được lấy từ token đại diện $\mathbf{h}_{\text{CLS}} = \mathbf{H}[:, 0, :] \in \mathbb{R}^{B \times 768}$.

2. **Trigger Classifier (Token-level BIO Tagging)**:
   - Kiến trúc: `Dropout(0.1) -> Linear(768, 384) -> ReLU() -> Dropout(0.1) -> Linear(384, 26)`
   - Nhận đầu vào là toàn bộ biểu diễn chuỗi $\mathbf{H}$.
   - Dự đoán 26 nhãn BIO cho Trigger (gồm nhãn `B-` và `I-` cho 13 loại sự kiện cùng nhãn `O`).

3. **Argument Classifier (Token-level BIO Tagging)**:
   - Kiến trúc: `Dropout(0.1) -> Linear(768, 384) -> ReLU() -> Dropout(0.1) -> Linear(384, 7)`
   - Nhận đầu vào là toàn bộ biểu diễn chuỗi $\mathbf{H}$.
   - Dự đoán 7 nhãn Argument (`B-Arg-Subject`, `I-Arg-Subject`, `B-Arg-Time`, `I-Arg-Time`, `B-Arg-Location`, `I-Arg-Location`, `O`).

4. **Event Type Classifier (Sequence-level Classification)**:
   - Kiến trúc: `Dropout(0.1) -> Linear(768, 384) -> ReLU() -> Dropout(0.1) -> Linear(384, 14)`
   - Nhận đầu vào là vector đại diện toàn câu `h_CLS` (CLS token representation).
   - Phân loại câu vào một trong 14 nhóm sự kiện chính trị (Celebration, Commemoration, Condolence, Cooperation, Diplomatic Reception, Government Formation, Inspection, Legislation, Meeting, Proposal, Publication, Recognition, Statement, Visiting).

5. **Hàm mất mát liên kết (Weighted Multi-Task Loss)**:
   - Huấn luyện kết hợp 3 nhánh mục tiêu thông qua hàm mất mát tổng hợp:

   ```math
   \mathcal{L}_{\text{total}} = 0.4 \times \mathcal{L}_{\text{trigger}} + 0.3 \times \mathcal{L}_{\text{argument}} + 0.3 \times \mathcal{L}_{\text{event}}
   ```

   - **Xử lý mất cân bằng lớp (Class Imbalance)**: Do nhãn `O` chiếm đa số (~95% lượng token), hàm mất mát sử dụng kỹ thuật tính trọng số nghịch đảo căn bậc hai tần suất (`calculate_class_weights`):

   ```math
   w_c = \frac{1}{\sqrt{N_c}} \times \frac{C}{\sum_{j=1}^{C} \frac{1}{\sqrt{N_j}}}
   ```

   giúp mô hình học hiệu quả các thẻ Trigger và Argument có tần suất xuất hiện thấp.

6. **Cơ chế căn chỉnh Subword (Subword-to-Word Alignment)**:
   - Sử dụng `word_ids()` của `XLMRobertaTokenizerFast` để ánh xạ chính xác kết quả dự đoán của các subword về từng từ token gốc, loại bỏ sai lệch độ dài khi xuất dữ liệu JSON.

---

## 2. Cấu Trúc Thư Mục Dự Án

```
DS200/
├── data/                                 # Du lieu phuc vu pipeline va huan luyen
│   ├── dataset.json                      # 5.814 bai bao mau cho Kafka Producer
│   └── results.csv                       # Tap du lieu gan nhan day du cho huan luyen
│
├── models/                               # Kien truc OneIE, trong so va bo ma hoa
│   ├── __init__.py
│   ├── model.py                          # Lop JointModel (XLM-RoBERTa + 3 classifier heads)
│   ├── checkpoints/
│   │   └── best_model.pt                 # Trong so tot nhat da huan luyen cua OneIE
│   └── encoders/
│       ├── trigger_encoder.pkl           # 26 lop nhan BIO trigger
│       ├── arg_encoder.pkl               # 7 lop nhan BIO argument
│       └── event_encoder.pkl             # 14 loai su kien
│
├── streaming/                            # Module Streaming thoi gian thuc qua Kafka
│   ├── __init__.py
│   ├── producer.py                       # Doc data/dataset.json, push vao political_news_raw
│   ├── extractor.py                      # Engine suy luan OneIE ket noi models/
│   ├── consumer.py                       # Tieu thu tu political_news_raw, xuat sang political_news_extracted
│   └── README.md                         # Huong dan chi tiet van hanh streaming
│
├── training/                             # Module huan luyen & danh gia mo hinh OneIE
│   ├── __init__.py
│   ├── train_pipeline.py                 # Huan luyen OneIE giai quyet mat can bang lop
│   └── Pipeline_Joint_Learning.ipynb     # Notebook nghien cuu & huan luyen goc
│
├── archive/                              # Luu tru lich su nghien cuu, du lieu cu va tai lieu
│   ├── notebooks/                        # Notebook crawl & gan nhan thu nghiem cu
│   │   ├── autolabel.ipynb
│   │   ├── baochinhphu_crawler.ipynb
│   │   ├── vnexpress_crawler.ipynb
│   │   ├── token_tags.ipynb
│   │   ├── json_to_csv.ipynb
│   │   ├── Demo_streaming.ipynb
│   │   └── Demo_best_model.ipynb
│   ├── legacy_data/                      # Du lieu crawl tho & bang tinh trung gian cu
│   └── docs/                             # Bao cao PDF & slides do an
│
├── docker-compose.yml                    # Cum Kafka Broker (Port 9092) va Zookeeper (Port 2181)
├── requirements.txt                      # Dependencies cho toan bo du an
└── README.md
```

---

## 3. Hướng Dẫn Vận Hành Nhanh

### Bước 1: Cài đặt thư viện phụ thuộc
```powershell
pip install -r requirements.txt
```

### Bước 2: Khởi động hạ tầng Kafka bằng Docker
```powershell
docker compose up -d
```
- **Kafka Broker**: `localhost:9092`
- **Zookeeper**: `localhost:2181`

### Bước 3: Huấn luyện mô hình (Tùy chọn)
```powershell
python training/train_pipeline.py
```
Trọng số tối ưu sẽ tự động được lưu vào `models/checkpoints/best_model.pt`.

### Bước 4: Chạy Streaming thời gian thực
Mở 2 cửa sổ terminal:

- **Terminal 1: Khởi động Consumer (Bộ trích xuất OneIE)**
  ```powershell
  python streaming/consumer.py
  ```
  *(Lắng nghe topic `political_news_raw`, chạy suy luận qua mô hình OneIE và đẩy kết quả sang `political_news_extracted`).*

- **Terminal 2: Khởi động Producer (Phát bài báo vào Kafka)**
  ```powershell
  python streaming/producer.py
  ```
  *(Đọc bài báo từ `data/dataset.json` và bắn đều đặn mỗi 2–3 giây vào Kafka).*
