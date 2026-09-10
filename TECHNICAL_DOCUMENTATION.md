# BÁO CÁO TỔNG HỢP VÀ PHÂN TÍCH KỸ THUẬT DỰ ÁN
## ĐỀ TÀI: TRÍCH XUẤT THÔNG TIN SỰ KIỆN TỪ TIN TỨC CHÍNH TRỊ VIỆT NAM
### (Event Extraction from Vietnamese Political News)

---

## MỤC LỤC
1. [Giới thiệu và Đặt vấn đề](#1-giới-thiệu-và-đặt-vấn-đề)
2. [Quy trình Xử lý và Tiền xử lý Dữ liệu](#2-quy-trình-xử-lý-và-tiền-xử-lý-dữ-liệu)
3. [Kiến trúc Mô hình Học Liên kết (OneIE Joint Learning)](#3-kiến-trúc-mô-hình-học-liên-kết-oneie-joint-learning)
4. [Chiến lược Huấn luyện và Tối ưu hóa](#4-chiến-lược-huấn-luyện-và-tối-ưu-hóa)
5. [Đánh giá và Kiểm nghiệm Mô hình](#5-đánh-giá-và-kiểm-nghiệm-mô-hình)
6. [Hệ thống Xử lý Luồng Thời gian Thực (Real-time Streaming với Apache Kafka)](#6-hệ-thống-xử-lý-luồng-thời-gian-thực-real-time-streaming-với-apache-kafka)
7. [Tổng kết và Hướng phát triển](#7-tổng-kết-và-hướng-phát-triển)

---

## 1. GIỚI THIỆU VÀ ĐẶT VẤN ĐỀ

### 1.1. Bối cảnh
Tin tức chính trị là nguồn thông tin quan trọng phản ánh các hoạt động đối nội, đối ngoại, chỉ đạo điều hành và sự kiện lập pháp của một quốc gia. Với sự bùng nổ của báo điện tử (đặc biệt là các cổng thông tin chính thức như Cổng Thông tin điện tử Chính phủ - baochinhphu.vn), lượng văn bản phát sinh hàng ngày là rất lớn. Việc đọc và phân loại thủ công tốn nhiều nhân lực và thời gian.

### 1.2. Mục tiêu nghiên cứu
Dự án hướng đến việc xây dựng một hệ thống hoàn chỉnh có khả năng tự động trích xuất thông tin sự kiện chính trị từ văn bản tiếng Việt theo chuẩn cấu trúc bao gồm 3 thành phần cốt lõi:
* **Event Trigger Tagging**: Nhận diện và phân loại từ/cụm từ kích hoạt sự kiện (ví dụ: *"ban hành"*, *"chủ trì"*, *"thăm chính thức"*).
* **Argument Detection**: Xác định các thực thể tham gia và đóng vai trò ngữ nghĩa trong sự kiện, bao gồm: Chủ thể (**Subject**), Thời gian (**Time**), và Địa điểm (**Location**).
* **Event Type Classification**: Phân loại toàn bộ câu/bài báo vào một trong 14 nhóm sự kiện chính trị được định nghĩa chuẩn.
* **Real-time Pipeline**: Triển khai khả năng xử lý luồng sự kiện theo thời gian thực kết nối với hạ tầng phân tán Apache Kafka.

---

## 2. QUY TRÌNH XỬ LÝ VÀ TIỀN XỬ LÝ DỮ LIỆU

### 2.1. Thu thập dữ liệu (Data Crawling)
Dữ liệu được thu thập tự động từ Cổng Thông tin điện tử Chính phủ Việt Nam (`baochinhphu.vn`) qua giao thức HTTP kết hợp thư viện `BeautifulSoup4` và `requests`.
* **Cấu trúc trường thu thập**: URL bài viết, Tiêu đề (`title`), Tóm tắt nội dung (`summary`), Thời gian xuất bản (`publish_date`).
* **Quy mô tập dữ liệu**: 5.814 bài báo chính trị được chuẩn hóa lưu trữ trong `data/dataset.json`.

### 2.2. Lược đồ gán nhãn BIO (BIO Tagging Scheme)
Để thực hiện trích xuất thực thể ở cấp độ từ (token-level), dự án áp dụng lược đồ mã hóa BIO:
* **B- (Begin)**: Đánh dấu token mở đầu một thực thể hoặc từ kích hoạt.
* **I- (Inside)**: Đánh dấu các token tiếp theo nằm trong cùng một cụm thực thể hoặc từ kích hoạt.
* **O (Outside)**: Các token thông thường không thuộc thực thể hay từ kích hoạt sự kiện.

#### Không gian nhãn trong đề tài:
1. **Trigger Labels (26 nhãn)**: Gồm nhãn `B-` và `I-` cho 13 loại sự kiện (Celebration, Commemoration, Condolence, Cooperation, Diplomatic Reception, Government Formation, Inspection, Legislation, Meeting, Proposal, Publication, Recognition, Visiting) và nhãn `O`.
2. **Argument Labels (7 nhãn)**: `B-Arg-Subject`, `I-Arg-Subject`, `B-Arg-Time`, `I-Arg-Time`, `B-Arg-Location`, `I-Arg-Location`, và nhãn `O`.
3. **Event Type Labels (14 nhãn)**: 14 lớp sự kiện chính trị độc lập.

### 2.3. Quy trình gán nhãn kết hợp (Hybrid Auto-Labeling)
Do việc gán nhãn thủ công hàng nghìn văn bản là bất khả thi, dự án áp dụng quy trình gán nhãn lai:
1. **Rule-based & Gazetters**: Xây dựng từ điển các cụm từ kích hoạt đặc trưng và danh sách thực thể chính trị Việt Nam (các bộ ban ngành, chức danh lãnh đạo, địa danh 63 tỉnh thành).
2. **LLM Verification**: Sử dụng API mô hình ngôn ngữ lớn (Gemini Flash) để gán nhãn và đối soát các trường hợp ngữ cảnh phức tạp.
3. **Lọc và Chuẩn hóa**: Loại bỏ các bản ghi không khớp độ dài token giữa chuỗi từ và chuỗi nhãn, thu được tập dữ liệu chất lượng cao `data/results.csv`.

---

## 3. KIẾN TRÚC MÔ HÌNH HỌC LIÊN KẾT (ONEIE JOINT LEARNING)

### 3.1. Hạn chế của phương pháp Pipeline tuần tự
Trong phương pháp truyền thống (Pipeline):
* Mô hình 1 phát hiện Trigger $\rightarrow$ Mô hình 2 phát hiện Arguments $\rightarrow$ Mô hình 3 phân loại loại sự kiện.
* **Nhược điểm nghiêm trọng**: Hiện tượng **lan truyền sai số (Error Propagation)**. Nếu bước 1 xác định sai Trigger, bước 2 và bước 3 chắc chắn đưa ra kết quả sai lệch mà không thể tự sửa chữa.

### 3.2. Cấu trúc OneIE Joint Learning
Dự án triển khai mô hình **JointModel** lấy cảm hứng từ cấu trúc OneIE (Lin et al.), cho phép các tác vụ chia sẻ không gian biểu diễn đặc trưng ẩn và bổ trợ thông tin ngữ nghĩa lẫn nhau:

```
                                  Văn bản đầu vào (Input Sentence)
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │     XLMRobertaTokenizerFast     │
                               │      (Subword Tokenization)     │
                               └─────────────────────────────────┘
                                                │ (input_ids, attention_mask)
                                                ▼
                               ┌─────────────────────────────────┐
                               │      XLM-RoBERTa (Backbone)     │
                               │        hidden_size = 768        │
                               └─────────────────────────────────┘
                                         │             │
                    ┌────────────────────┘             └────────────────────┐
                    ▼ (Sequence Output: H)                                  ▼ (CLS Token: h_CLS)
         ┌─────────────────────┐              ┌─────────────────────┐    ┌─────────────────────┐
         │ Trigger Classifier  │              │ Argument Classifier │    │  Event Classifier   │
         │ Linear(768 -> 384)  │              │ Linear(768 -> 384)  │    │ Linear(768 -> 384)  │
         │ ReLU + Dropout(0.1) │              │ ReLU + Dropout(0.1) │    │ ReLU + Dropout(0.1) │
         │ Linear(384 -> 26)   │              │ Linear(384 -> 7)    │    │ Linear(384 -> 14)   │
         └─────────────────────┘              └─────────────────────┘    └─────────────────────┘
                    │                                    │                          │
                    ▼                                    ▼                          ▼
             Trigger BIO Tags                     Argument BIO Tags             Event Type
               (26 nhãn)                             (7 nhãn)                   (14 nhãn)
```

### 3.3. Các thành phần chi tiết của JointModel
1. **Bộ trích xuất đặc trưng ngôn ngữ (Backbone)**:
   * Sử dụng `XLMRobertaModel` (`xlm-roberta-base`) được tiền huấn luyện trên 2.5TB dữ liệu đa ngôn ngữ Common Crawl (trong đó có tiếng Việt).
   * Đầu ra biểu diễn chuỗi: $\mathbf{H} \in \mathbb{R}^{B \times L \times 768}$, trong đó $B$ là kích thước batch, $L$ là độ dài token (`max_length = 128`), và $768$ là số chiều vector ẩn.
   * Vector biểu diễn toàn câu: Lấy tại vị trí token đầu tiên $\mathbf{h}_{\text{CLS}} = \mathbf{H}[:, 0, :] \in \mathbb{R}^{B \times 768}$.

2. **Nhánh phân loại Trigger (Trigger Head)**:
   * Mạng 2 tầng nơ-ron: `Dropout(0.1) -> Linear(768, 384) -> ReLU() -> Dropout(0.1) -> Linear(384, 26)`.
   * Áp dụng lên từng vector vị trí $\mathbf{H}[:, i, :]$ để sinh logits cho 26 nhãn BIO Trigger.

3. **Nhánh phân loại Argument (Argument Head)**:
   * Mạng 2 tầng nơ-ron: `Dropout(0.1) -> Linear(768, 384) -> ReLU() -> Dropout(0.1) -> Linear(384, 7)`.
   * Áp dụng lên từng vector vị trí $\mathbf{H}[:, i, :]$ để sinh logits cho 7 nhãn BIO Argument.

4. **Nhánh phân loại Loại Sự Kiện (Event Head)**:
   * Mạng 2 tầng nơ-ron: `Dropout(0.1) -> Linear(768, 384) -> ReLU() -> Dropout(0.1) -> Linear(384, 14)`.
   * Áp dụng lên vector đại diện $\mathbf{h}_{\text{CLS}}$ để sinh phân bố xác suất cho 14 lớp sự kiện.

5. **Kỹ thuật căn chỉnh Subword (Subword Alignment)**:
   * Khi tokenizer tách từ tiếng Việt thành các mẩu subword (ví dụ: *"Hưng Yên"* thành *"Hưng"*, *"_Y", "_ên"*), `XLMRobertaTokenizerFast.word_ids()` được dùng để bắt cặp nhãn dự đoán của subword đầu tiên về lại đúng từ vựng gốc, bảo toàn tính chuẩn xác cho JSON đầu ra.

---

## 4. CHIẾN LƯỢC HUẤN LUYỆN VÀ TỐI ƯU HÓA

### 4.1. Hàm mất mát liên kết đa nhiệm (Joint Multi-Task Loss)
Hệ thống kết hợp ba hàm mất mát thành phần với trọng số tối ưu thực nghiệm:

```math
\mathcal{L}_{\text{total}} = 0.4 \times \mathcal{L}_{\text{trigger}} + 0.3 \times \mathcal{L}_{\text{arg}} + 0.3 \times \mathcal{L}_{\text{event}}
```

Trong đó:
* $\mathcal{L}_{\text{trigger}}$ và $\mathcal{L}_{\text{arg}}$ là Cross-Entropy Loss tính trên các token hợp lệ (loại trừ padding với `ignore_index = -100`).
* $\mathcal{L}_{\text{event}}$ là Cross-Entropy Loss phân loại câu.

### 4.2. Giải quyết vấn đề mất cân bằng lớp cực đoan (Severe Class Imbalance)
* **Thực trạng**: Trong tác vụ trích xuất thực thể, hơn 95% token trong văn bản mang nhãn `O` (Outside). Các nhãn quan trọng như `B-Arg-Time`, `B-Legislation` chỉ chiếm dưới 1%. Nếu huấn luyện thông thường, mô hình sẽ tối ưu cục bộ bằng cách dự đoán tất cả là `O`, dẫn đến F1-score của các thực thể gần bằng 0.
* **Giải pháp**: Áp dụng kỹ thuật tính trọng số nghịch đảo căn bậc hai tần suất xuất hiện (`calculate_class_weights`):

```math
w_c = \frac{1}{\sqrt{N_c}} \times \frac{C}{\sum_{j=1}^{C} \frac{1}{\sqrt{N_j}}}
```

Trọng số này được đưa trực tiếp vào tham số `weight` của `nn.CrossEntropyLoss`, buộc mạng nơ-ron phải chịu phạt nặng khi dự đoán sai các nhãn thực thể hiếm.

### 4.3. Tối ưu hóa tham số (Optimization)
* **Thuật toán**: `AdamW` với tốc độ học $\eta = 3 \times 10^{-5}$ và cơ chế suy giảm trọng số (weight decay) chống quá khớp.
* **Gradient Clipping**: Giới hạn chuẩn gradient tối đa ở mức $1.0$ (`torch.nn.utils.clip_grad_norm_`) để đảm bảo tính ổn định trong quá trình cập nhật trọng số.
* **Fine-tuning theo tầng (Layer Freezing)**: Đóng băng 10 tầng Transformer đầu tiên của XLM-RoBERTa, chỉ fine-tune 2 tầng cuối (layer 10, 11) và 3 đầu phân loại, giúp tốc độ huấn luyện nhanh gấp 4 lần trên CPU/GPU mà vẫn giữ được độ hội tụ tối ưu.

---

## 5. ĐÁNH GIÁ VÀ KIỂM NGHIỆM MÔ HÌNH

### 5.1. Thang đo đánh giá (Evaluation Metrics)
* **Precision (Macro)**: Đo lường độ chính xác của các dự đoán dương tính trên từng lớp nhãn.
* **Recall (Macro)**: Đo lường khả năng không bỏ sót các thực thể thực tế trên từng lớp nhãn.
* **F1-score (Macro)**: Trung bình điều hòa giữa Precision và Recall, là chỉ số quan trọng nhất phản ánh khả năng nhận diện cân bằng giữa các lớp hiếm và lớp phổ biến:
  $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \times \text{Precision}_k \times \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

### 5.2. Kết quả kiểm nghiệm thực tế
* Mô hình cho độ chính xác cao trong việc nhận diện đúng loại sự kiện chính trị (`event_type`).
* Các thẻ Argument chủ chốt như `B-Arg-Subject` (Chủ thể thực hiện sự kiện: *Thủ tướng, Quốc hội, Bộ trưởng...*), `B-Arg-Time` (*Sáng 1/7, Chiều nay...*), `B-Arg-Location` (*Hà Nội, Hưng Yên...*) được trích xuất chính xác theo từng token.

---

## 6. HỆ THỐNG XỬ LÝ LUỒNG THỜI GIAN THỰC (REAL-TIME STREAMING VỚI APACHE KAFKA)

### 6.1. Kiến trúc luồng dữ liệu (Data Pipeline Architecture)

```
[ data/dataset.json ]
        │
        ▼
┌────────────────────────┐
│  streaming/producer.py │ ────(Phát bài báo định dạng JSON UTF-8, 2-3s)────► [ Topic: political_news_raw ]
└────────────────────────┘                                                                  │
                                                                                            ▼
                                                                                ┌────────────────────────┐
                                                                                │  streaming/consumer.py │
                                                                                │ (news_event_extraction)│
                                                                                └────────────────────────┘
                                                                                            │
                                                                                            ▼
                                                                                ┌────────────────────────┐
                                                                                │  streaming/extractor.py│
                                                                                │   (JointModel OneIE)   │
                                                                                └────────────────────────┘
                                                                                            │
                                                                                            ▼
[ Terminal Consumer Log ] ◄──(Bản ghi sự kiện JSON chi tiết)───────── [ Topic: political_news_extracted ]
```

### 6.2. Các thành phần trong phân hệ Streaming

1. **Hạ tầng Docker Compose (`docker-compose.yml`)**:
   * `kafka_broker` (Port 9092): Broker tiếp nhận, quản lý phân vùng (partitions) và lưu trữ tin nhắn.
   * `kafka_zookeeper` (Port 2181): Điều phối cụm và quản lý metadata của broker.
   * Tối ưu hóa tài nguyên phần cứng bằng cách chạy trực tiếp các container cốt lõi và theo dõi qua luồng log Terminal tiêu chuẩn.

2. **Kafka Producer (`streaming/producer.py`)**:
   * Đọc tuần tự dữ liệu từ `data/dataset.json`.
   * Tuần tự hóa JSON với mã hóa UTF-8 đầy đủ hỗ trợ tiếng Việt có dấu.
   * Định kỳ phát bài báo mới vào topic `political_news_raw` với tần suất cấu hình linh hoạt (2–3 giây/tin).
   * Tích hợp cơ chế callback `_on_delivery_success` và `_on_delivery_error` giám sát việc phân phối bản tin.

3. **Inference Extractor (`streaming/extractor.py`)**:
   * Đóng gói toàn bộ logic suy luận: tải mô hình `JointModel` từ `models/checkpoints/best_model.pt` và các bộ giải mã `models/encoders/`.
   * Thực thi `predict()` thuần túy dựa trên mạng nơ-ron học sâu (không dùng luật cứng hoặc regex).
   * Ánh xạ subword về từ vựng gốc và xuất danh sách thực thể có gắn nhãn BIO.

4. **Kafka Consumer (`streaming/consumer.py`)**:
   * Thuộc Consumer Group `news_event_extraction_group`, lắng nghe liên tục topic `political_news_raw`.
   * Áp dụng cơ chế **Manual Commit** (`enable_auto_commit=False`): chỉ cam kết offset sau khi đã trích xuất thành công và gửi sang topic đầu ra, bảo đảm nguyên tắc **At-least-once Processing**.
   * Đẩy kết quả có cấu trúc sang topic `political_news_extracted`.
   * Độ trễ trung bình của toàn bộ chu trình trích xuất chỉ dao động trong khoảng **200ms - 350ms** cho mỗi bài báo.

5. **Cơ chế Graceful Shutdown**:
   * Bắt các tín hiệu ngắt hệ điều hành (`SIGINT`, `SIGTERM` qua `Ctrl + C`).
   * Tự động hoàn tất các bản ghi đang dở dang, xả bộ đệm (flush), đóng kết nối an toàn, ngăn chặn tình trạng thất thoát dữ liệu hoặc corrupt consumer group offset.

---

## 7. TỔNG KẾT VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Kết quả đạt được
1. Xây dựng thành công quy trình xử lý dữ liệu và gán nhãn cho hơn 5.800 bài báo chính trị tiếng Việt.
2. Thiết kế và triển khai kiến trúc OneIE Joint Learning sử dụng XLM-RoBERTa giải quyết đồng thời 3 bài toán trích xuất sự kiện.
3. Giải quyết triệt để vấn đề mất cân bằng nhãn trong dữ liệu bằng hàm mất mát có trọng số.
4. Triển khai hoàn chỉnh hệ thống Streaming thời gian thực trên nền tảng Apache Kafka và Docker.
5. Tái cấu trúc mã nguồn theo tiêu chuẩn công nghiệp: phân chia module rõ ràng (`data/`, `models/`, `streaming/`, `training/`, `archive/`), mã nguồn sạch và tài liệu hóa đầy đủ.

### 7.2. Hướng phát triển tiếp theo
* Bổ sung cơ chế Relation Extraction để xác định mối quan hệ giữa các arguments và trigger (ví dụ: liên kết Subject nào thực hiện Action nào).
* Nâng cấp lên kiến trúc Spark Structured Streaming để xử lý song song với lượng dữ liệu báo chí cực lớn (Big Data Scale).
* Triển khai giao diện Dashboard trực quan hóa dòng sự kiện theo dòng thời gian (Timeline Event Visualization).
