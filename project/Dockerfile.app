FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install fastapi uvicorn transformers torch

COPY . .

EXPOSE 8005

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
