
FROM apache/airflow:2.7.1-python3.10

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends docker.io libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install kafka-python \
    "numpy<2" \
    pandas sqlalchemy psycopg2-binary \
    scikit-learn lightgbm xgboost imbalanced-learn \
    shap matplotlib
