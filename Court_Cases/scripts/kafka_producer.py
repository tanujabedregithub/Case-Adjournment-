import csv
import json
import os
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import time

while True:
    try:
        admin = KafkaAdminClient(
            bootstrap_servers="kafka:29092",
            client_id="topic_creator"
        )
        break
    except Exception:
        print("Waiting for Kafka...")
        time.sleep(5)

try:
    admin.create_topics([
        NewTopic(
            name="court_cases_raw",
            num_partitions=1,
            replication_factor=1
        )
    ])
    print("Topic created.")
except TopicAlreadyExistsError:
    print("Topic already exists.")
except Exception as e:
    print(f"Topic creation skipped: {e}")

admin.close()
time.sleep(3)

producer = KafkaProducer(
    bootstrap_servers=['kafka:29092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    
    # WHY: acks=1 means only wait for Kafka leader
    # to confirm, not all replicas. Much faster.
    acks=1,
    
    # WHY: batch_size groups multiple messages
    # together and sends them in one network call
    # instead of one call per message
    batch_size=65536,           # 64KB per batch
    
    # WHY: linger_ms waits 10ms to collect more
    # messages before sending — fills batches better
    linger_ms=10,
    
    # WHY: compression reduces data size by 60-70%
    # less data = faster network transfer to Kafka
    compression_type='gzip',
    
    
    
    
    retries=2,
    max_block_ms=60000,
)

TOPIC = 'court_cases_raw'
BASE_PATH = '/opt/airflow/data/csv/cases'

for year in range(2010, 2012):
    file_path = f"{BASE_PATH}/cases_{year}.csv"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"Starting {year}...")
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            payload = {
                "ddl_case_id"      : row.get("ddl_case_id"),
                "state_code"       : row.get("state_code"),
                "dist_code"        : row.get("dist_code"),
                "court_no"         : row.get("court_no"),
                "judge_position"   : row.get("judge_position"),
                "type_name"        : row.get("type_name"),
                "purpose_name"     : row.get("purpose_name"),
                "disp_name"        : row.get("disp_name"),
                "date_of_filing"   : row.get("date_of_filing"),
                "date_of_decision" : row.get("date_of_decision"),
                "date_first_list"  : row.get("date_first_list"),
                "date_last_list"   : row.get("date_last_list"),
                "date_next_list"   : row.get("date_next_list"),
                "female_petitioner": row.get("female_petitioner"),
                "female_defendant" : row.get("female_defendant")
            }

            # WHY: No future.get() here — fire and forget
            # Producer sends and moves immediately to next row
            producer.send(TOPIC, value=payload)
            count += 1

            if count % 500000 == 0:
                producer.flush()

        producer.flush()
        print(f"Year {year} done: {count:,} total")

producer.close()
print("All years complete.")