from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "pushkar",
}

with DAG(
    dag_id="court_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1
) as dag:
    run_producer = BashOperator(
    task_id="run_kafka_producer",
    bash_command="python3 -u /opt/airflow/scripts/kafka_producer.py",
)

run_spark = BashOperator(
    task_id="run_spark_streaming",
    bash_command="""
    docker exec pipeline_spark_master \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --driver-memory 2g \
    --executor-memory 4g \
    --total-executor-cores 2 \
    --conf spark.jars.ivy=/tmp/.ivy2 \
    --conf spark.network.timeout=600s \
    --conf spark.executor.heartbeatInterval=60s \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.6.0 \
    /opt/spark/work-dir/scripts/spark_streaming.py
    """,
)

run_aggregation = BashOperator(
    task_id="run_spark_aggregation",
    bash_command="""
    docker exec pipeline_spark_master \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --driver-memory 1g \
    --executor-memory 2g \
    --total-executor-cores 1 \
    --conf spark.network.timeout=600s \
    --conf spark.executor.heartbeatInterval=60s \
    --conf spark.jars.ivy=/tmp/.ivy2 \
    --packages org.postgresql:postgresql:42.6.0 \
    /opt/spark/work-dir/scripts/spark_aggregations.py
    """,
)

run_train_model = BashOperator(
    task_id="run_train_model",
    bash_command="python3 -u /opt/airflow/scripts/ml/train_model.py",
)

run_predict_lineup = BashOperator(
    task_id="run_predict_lineup",
    bash_command="python3 -u /opt/airflow/scripts/ml/predict_lineup.py",
)

run_producer >> run_spark >> run_aggregation >> run_train_model >> run_predict_lineup

