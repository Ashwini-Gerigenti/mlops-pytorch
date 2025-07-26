FROM python:3.10-slim

WORKDIR /app

COPY train.py predict.py ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN python train.py

CMD ["python", "predict.py"]
