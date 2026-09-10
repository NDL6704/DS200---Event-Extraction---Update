"""
producer.py
Producer mo phong nguon du lieu stream bai bao chinh tri Viet Nam:
- Doc du lieu bai bao tu data/dataset.json
- Tuan tu hoa JSON duoi dinh dang UTF-8 chuan xac
- Day vao Kafka topic 'political_news_raw' theo chu ky cau hinh duoc (mac dinh 2-3s)
- Ghi log day du: partition, offset, article_id, timestamp
- Ho tro Graceful Shutdown khi nhan Ctrl + C
"""

import os
import sys
import json
import time
import random
import signal
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Dam bao ma hoa UTF-8 cho stdout/stderr tren Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    print("[ERROR] Thu vien 'kafka' chua duoc cai dat. Hay chay: pip install kafka-python-ng")
    sys.exit(1)


class NewsArticleProducer:
    """
    Lop phat du lieu bai bao chinh tri vao Kafka topic
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "political_news_raw",
        data_path: Optional[str] = None,
        min_interval: float = 2.0,
        max_interval: float = 3.0,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.is_running = True
        self.total_sent = 0

        # Khoi tao Kafka Producer voi serializer UTF-8
        print(f"[INFO] Dang ket noi toi Kafka Broker tai {self.bootstrap_servers}...")
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1,
            )
            print(f"[SUCCESS] Da ket noi thanh cong toi Kafka topic '{self.topic}'.")
        except Exception as e:
            print(f"[ERROR] Khong the ket noi toi Kafka: {e}")
            print("[HINT] Vui long dam bao Kafka dang chay qua Docker: 'docker compose up -d'")
            sys.exit(1)

        # Tai du lieu bai bao mau
        self.articles = self._load_data(data_path)
        print(f"[INFO] Da nap thanh cong {len(self.articles)} bai bao mau vao bo dem.")

        # Dang ky xu ly tin hieu ngat an toan
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _load_data(self, data_path: Optional[str]) -> List[Dict[str, Any]]:
        """Tai danh sach bai bao tu file dataset.json hoac cac file co san"""
        candidates = []
        if data_path:
            candidates.append(data_path)

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(curr_dir, ".."))
        candidates.extend([
            os.path.join(root_dir, "data", "dataset.json"),
            os.path.join(root_dir, "data", "results.csv"),
            os.path.join(root_dir, "dataset.json"),
            os.path.join(root_dir, "results.csv"),
        ])

        for path in candidates:
            if os.path.exists(path):
                print(f"[INFO] Dang doc du lieu tu: {path}...")
                if path.endswith(".json"):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            return data
                    except Exception as e:
                        print(f"[WARNING] Loi doc file JSON {path}: {e}")
                elif path.endswith(".csv"):
                    try:
                        import pandas as pd
                        df = pd.read_csv(path)
                        records = df.to_dict(orient="records")
                        if len(records) > 0:
                            return records
                    except Exception as e:
                        print(f"[WARNING] Loi doc file CSV {path}: {e}")

        # Du lieu du phong neu khong tim thay file
        print("[WARNING] Khong tim thay file du lieu goc, su dung danh sach bai bao mau mac dinh.")
        return [
            {
                "url": "https://baochinhphu.vn/ban-hanh-nghi-quyet-moi-1.htm",
                "title": "Quoc hoi thong qua Nghi quyet ve ke hoach phat trien kinh te - xa hoi",
                "summary": "Chieu 20/6, tai Hoi truong Quoc hoi, cac dai bieu da tien hanh bieu quyet thong qua Nghi quyet moi.",
                "publish_date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            },
            {
                "url": "https://baochinhphu.vn/thu-tuong-chu-tri-phien-hop-2.htm",
                "title": "Thu tuong Pham Minh Chinh chu tri Phien hop Chinh phu thuong ky",
                "summary": "Sang nay, Thu tuong Chinh phu chu tri cuoc hop nham ra soat cac nhiem vu trong tam.",
                "publish_date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            },
            {
                "url": "https://baochinhphu.vn/ngoai-giao-song-phuong-3.htm",
                "title": "Viet Nam va doi tac thuc day quan he hop tac ngoai giao toan dien",
                "summary": "Hai ben nhat tri tang cuong trao doi doan cap cao va mo rong thuong mai song phuong.",
                "publish_date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            }
        ]

    def _handle_shutdown(self, signum, frame):
        """Xu ly Graceful Shutdown khi nhan Ctrl+C"""
        print("\n\n[INFO] Nhan tin hieu dung (SIGINT/SIGTERM). Dang tien hanh tat an toan...")
        self.is_running = False

    def _on_delivery_success(self, metadata, article_id, title):
        """Callback khi gui message thanh cong"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title_disp = (title[:50] + "...") if len(title) > 50 else title
        print(
            f"[{now}] [SENT SUCCESS] ID: {article_id:<18} | "
            f"Topic: {metadata.topic} | Partition: {metadata.partition} | Offset: {metadata.offset}\n"
            f"                     Title: {title_disp}"
        )

    def _on_delivery_error(self, exc, article_id):
        """Callback khi gui message that bai"""
        print(f"[ERROR] Gui message that bai ID: {article_id} - Chi tiet: {exc}")

    def run(self, limit: Optional[int] = None):
        """Bat dau vong lap phat bai bao vao Kafka topic"""
        print("\n" + "=" * 70)
        print(f"[START] PRODUCER: Topic '{self.topic}' (Chu ky: {self.min_interval:.1f}s - {self.max_interval:.1f}s)")
        print("[INFO] Nhan Ctrl + C de dung phat du lieu bat ky luc nao.")
        print("=" * 70 + "\n")

        idx = 0
        total_articles = len(self.articles)

        while self.is_running:
            if limit and self.total_sent >= limit:
                print(f"\n[INFO] Da dat gioi han gui {limit} bai bao. Ket thuc qua trinh.")
                break

            article = self.articles[idx % total_articles]
            idx += 1
            self.total_sent += 1

            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            article_id = f"art_{timestamp_str}_{self.total_sent:04d}"

            payload = {
                "article_id": article_id,
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "summary": article.get("summary", ""),
                "publish_date": article.get("publish_date", ""),
                "crawled_at": datetime.now().isoformat(),
                "sequence_number": self.total_sent,
            }

            # Giu nguyen annotation neu co san trong du lieu mau
            for extra_field in ["event_type", "trigger", "arg_subject", "arg_time", "arg_location"]:
                if extra_field in article and article[extra_field]:
                    payload[extra_field] = article[extra_field]

            try:
                future = self.producer.send(
                    topic=self.topic,
                    key=article_id,
                    value=payload
                )
                future.add_callback(self._on_delivery_success, article_id, payload["title"])
                future.add_errback(self._on_delivery_error, article_id)
            except Exception as e:
                print(f"[ERROR] Loi ngoai le khi gui ID {article_id}: {e}")

            if not self.is_running:
                break

            sleep_time = random.uniform(self.min_interval, self.max_interval)
            time.sleep(sleep_time)

        self.close()

    def close(self):
        """Dong ket noi producer an toan"""
        print("[INFO] Dang day not cac ban tin con lai trong bo dem (flush)...")
        try:
            self.producer.flush(timeout=5)
            self.producer.close(timeout=5)
            print("[INFO] Da dong ket noi Kafka Producer an toan.")
        except Exception as e:
            print(f"[WARNING] Loi khi dong Producer: {e}")

        print("\n" + "=" * 70)
        print(f"[REPORT] Tong so bai bao da gui: {self.total_sent}")
        print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Kafka News Article Streaming Producer")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Dia chi Kafka Broker (mac dinh: localhost:9092)"
    )
    parser.add_argument(
        "--topic",
        default="political_news_raw",
        help="Kafka topic dau vao (mac dinh: political_news_raw)"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Duong dan den file du lieu mau (dataset.json hoac results.csv)"
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=2.0,
        help="Khoang thoi gian nghi toi thieu giua cac tin (giay, mac dinh: 2.0)"
    )
    parser.add_argument(
        "--max-interval",
        type=float,
        default=3.0,
        help="Khoang thoi gian nghi toi da giua cac tin (giay, mac dinh: 3.0)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Chi gui het danh sach bai bao 1 lan duy nhat"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Gioi han so luong bai bao can gui"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    producer = NewsArticleProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        data_path=args.data_path,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
    )

    limit = args.limit
    if args.no_loop and limit is None:
        limit = len(producer.articles)

    producer.run(limit=limit)
