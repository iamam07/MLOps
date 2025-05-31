FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install fastapi uvicorn transformers torch

COPY . .
COPY best_model.pt /app/checkpoints/best_model.pt

EXPOSE 8005

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
