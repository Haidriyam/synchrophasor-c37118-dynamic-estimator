FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY estimator/ estimator/

USER 10001
CMD ["python", "-c", "from estimator.c37118_parser import C37118FrameParser; print('PMU Telemetry Engine Online.')"]