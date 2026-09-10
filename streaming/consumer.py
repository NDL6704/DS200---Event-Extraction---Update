"""
consumer.py
Consumer lang nghe bai bao tu Kafka topic 'political_news_raw' (Consumer Group: 'news_event_extraction_group'),
ket noi truc tiep mo hinh OneIE Event Extraction de trich xuat Trigger, Argument, Event Type,
va day ket qua co cau truc sang topic 'political_news_extracted'.
"""

import os
import sys
import json
import time
import signal
import argparse
from datetime import datetime
from typing import Optional

# Reconfigure stdout/stderr for UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Thiet lap duong dan import module cuc bo
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    print("[ERROR] Thu vien 'kafka' chua duoc cai dat. Hay chay: pip install kafka-python-ng")
    sys.exit(1)

from extractor import EventExtractor


class NewsEventConsumer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        raw_topic: str = "political_news_raw",
        extracted_topic: str = "political_news_extracted",
        group_id: str = "news_event_extraction_group",
        model_path: Optional[str] = None,
        encoder_dir: Optional[str] = None,
        auto_offset_reset: str = "latest",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.raw_topic = raw_topic
        self.extracted_topic = extracted_topic
        self.group_id = group_id
        self.is_running = True
        self.total_processed = 0

        print("\n" + "=" * 75)
        print("[INIT] KHOI TAO HE THONG EVENT EXTRACTION CONSUMER")
        print("=" * 75)

        # 1. Khoi tao mo hinh EventExtractor
        print("[INFO] Dang tai mo hinh suy luan OneIE Event Extraction...")
        self.extractor = EventExtractor(
            model_path=model_path,
            encoder_dir=encoder_dir,
        )

        # 2. Khoi tao Kafka Consumer
        print(f"[INFO] Dang ket noi Consumer toi topic '{self.raw_topic}' (Group: '{self.group_id}')...")
        try:
            self.consumer = KafkaConsumer(
                self.raw_topic,
                bootstrap_servers=self.bootstrap_servers.split(","),
                group_id=self.group_id,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=False,  # Cam ket thu cong sau khi xu ly xong ban tin
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                consumer_timeout_ms=1000,   # Quet non-blocking theo chu ky 1 giay
            )
            print("[SUCCESS] Kafka Consumer da san sang.")
        except Exception as e:
            print(f"[ERROR] Khong the khoi tao Kafka Consumer: {e}")
            print("[HINT] Vui long dam bao Kafka dang chay qua Docker: 'docker compose up -d'")
            sys.exit(1)

        # 3. Khoi tao Kafka Producer (de day ket qua trich xuat sang topic moi)
        print(f"[INFO] Dang ket noi Producer toi topic dau ra '{self.extracted_topic}'...")
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
            )
            print("[SUCCESS] Kafka Output Producer da san sang.")
        except Exception as e:
            print(f"[ERROR] Khong the khoi tao Output Producer: {e}")
            sys.exit(1)

        # Dang ky xu ly tin hieu ngat an toan
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Xu ly Graceful Shutdown khi nhan Ctrl+C"""
        print("\n\n[INFO] Nhan tin hieu dung (SIGINT/SIGTERM). Dang hoan tat va dong consumer...")
        self.is_running = False

    def process_message(self, message):
        """Xu ly mot message nhan duoc tu Kafka"""
        article_data = message.value
        if not isinstance(article_data, dict):
            print(f"[WARNING] Du lieu khong hop le (khong phai JSON object): {article_data}")
            return

        article_id = article_data.get("article_id") or message.key or f"art_unknown_{self.total_processed}"
        title = article_data.get("title", "")
        summary = article_data.get("summary", "")
        url = article_data.get("url", "")
        publish_date = article_data.get("publish_date", "")

        # Uu tien trich xuat tu summary, neu summary rong thi lay title
        text_to_extract = summary.strip() if summary else title.strip()

        start_time = time.time()
        # Chay suy luan qua mo hinh OneIE EventExtractor
        extracted_result = self.extractor.predict(text_to_extract, article_data=article_data)
        duration_ms = (time.time() - start_time) * 1000

        # Cau truc payload dau ra chuan
        output_record = {
            "article_id": article_id,
            "source_url": url,
            "title": title,
            "original_summary": summary,
            "publish_date": publish_date,
            "processed_at": datetime.now().isoformat(),
            "processing_time_ms": round(duration_ms, 2),
            "extracted_event": extracted_result,
        }

        # Gui ket qua sang topic extracted
        future = self.producer.send(
            topic=self.extracted_topic,
            key=article_id,
            value=output_record
        )
        metadata = future.get(timeout=10)

        # Cam ket offset ban tin vua xu ly thanh cong
        self.consumer.commit()
        self.total_processed += 1

        # Hien thi log dung dinh dang yeu cau
        self._print_log(
            in_partition=message.partition,
            in_offset=message.offset,
            out_partition=metadata.partition,
            out_offset=metadata.offset,
            article_id=article_id,
            title=title,
            duration_ms=duration_ms,
            extracted=extracted_result,
        )

    def _print_log(self, in_partition, in_offset, out_partition, out_offset, article_id, title, duration_ms, extracted):
        title_disp = (title[:50] + "...") if len(title) > 50 else title
        event_type = extracted.get("event_type", "None")
        entities = extracted.get("entities", [])

        # Dinh dang chuan theo notebook va yeu cau
        print(f"\n--- Processed: {title_disp} ---")
        output_event_data = {
            "event_type": event_type,
            "entities": entities
        }
        print(json.dumps(output_event_data, ensure_ascii=False, indent=2))
        print(f"[ROUTING] In: P{in_partition}@O{in_offset} -> Out: P{out_partition}@O{out_offset} ({duration_ms:.1f}ms)")
        print("-" * 50)

    def run(self, max_messages: Optional[int] = None):
        """Bat dau lang nghe tin tuc lien tuc"""
        print(f"\n[START] CONSUMER BAT DAU LANG NGHE TOPIC '{self.raw_topic}'...")
        print("[INFO] Nhan Ctrl + C de dung Consumer bat ky luc nao.\n")

        while self.is_running:
            if max_messages and self.total_processed >= max_messages:
                print(f"\n[INFO] Da dat gioi han xu ly {max_messages} ban tin. Ket thuc qua trinh.")
                break

            try:
                # Polling tin nhan tu topic
                message_batch = self.consumer.poll(timeout_ms=1000)
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        if not self.is_running:
                            break
                        self.process_message(message)
                        if max_messages and self.total_processed >= max_messages:
                            break
                    if not self.is_running or (max_messages and self.total_processed >= max_messages):
                        break
            except Exception as e:
                if self.is_running:
                    print(f"[ERROR] Loi trong vong lap Consumer: {e}")
                time.sleep(1.0)

        self.close()

    def close(self):
        """Dong ket noi consumer va producer an toan"""
        print("[INFO] Dang giai phong tai nguyen va dong cac ket noi...")
        try:
            self.consumer.close()
            self.producer.flush(timeout=5)
            self.producer.close(timeout=5)
            print("[INFO] Da dong Consumer va Producer an toan.")
        except Exception as e:
            print(f"[WARNING] Loi khi dong ket noi: {e}")

        print("\n" + "=" * 75)
        print(f"[REPORT] Da trich xuat thanh cong {self.total_processed} bai bao sang topic '{self.extracted_topic}'")
        print("=" * 75 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Kafka News Article Event Extraction Consumer")
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Dia chi Kafka Broker (mac dinh: localhost:9092)"
    )
    parser.add_argument(
        "--raw-topic",
        default="political_news_raw",
        help="Topic dau vao chua bai bao tho (mac dinh: political_news_raw)"
    )
    parser.add_argument(
        "--extracted-topic",
        default="political_news_extracted",
        help="Topic dau ra chua ket qua trich xuat (mac dinh: political_news_extracted)"
    )
    parser.add_argument(
        "--group-id",
        default="news_event_extraction_group",
        help="Kafka Consumer Group ID (mac dinh: news_event_extraction_group)"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Duong dan checkpoint best_model.pt (mac dinh tu dong tim trong models/checkpoints)"
    )
    parser.add_argument(
        "--encoder-dir",
        default=None,
        help="Duong dan thu muc chua encoders (mac dinh tu dong tim trong models/encoders)"
    )
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Doc lai tu ban tin dau tien trong topic (offset earliest)"
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="So luong tin toi da can xu ly truoc khi dung"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    offset_reset = "earliest" if args.from_beginning else "latest"

    consumer = NewsEventConsumer(
        bootstrap_servers=args.bootstrap_servers,
        raw_topic=args.raw_topic,
        extracted_topic=args.extracted_topic,
        group_id=args.group_id,
        model_path=args.model_path,
        encoder_dir=args.encoder_dir,
        auto_offset_reset=offset_reset,
    )

    consumer.run(max_messages=args.max_messages)
