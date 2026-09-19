FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

USER 10001
CMD ["python", "-c", "from src.ekf_core import ExtendedKalmanFilterBMS; print('BMS Estimator Initialized.')"]