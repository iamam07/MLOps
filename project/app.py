from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from pathlib import Path
from model import CrossEncoderRegressionModel

import os
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/mnt/model/best_model.pt"))

TOKENIZER_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

app = FastAPI()

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el tokenizador y el modelo al iniciar la API
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = CrossEncoderRegressionModel(model_name=TOKENIZER_NAME)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)  # Cargar directamente el state_dict
except FileNotFoundError:
    raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}")
model.to(device)
model.eval()

# Definir el esquema de entrada (dos oraciones)
class Sentences(BaseModel):
    sentence1: str
    sentence2: str

@app.post("/predict")
def predict_similarity(sentences: Sentences):
    try:
        # Tokenizar las oraciones juntas
        encoding = tokenizer(
            sentences.sentence1,  # Corregir: usar sentences.sentence1
            sentences.sentence2,  # Corregir: usar sentences.sentence2
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Mover tensores al dispositivo
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Calcular similitud
        with torch.no_grad():
            similarity = model(input_ids, attention_mask)
            similarity = similarity.item() * 5.0  # Escalar a [0, 5]
        
        return {"similarity": float(similarity)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8005)