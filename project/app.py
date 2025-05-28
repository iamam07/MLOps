from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from pathlib import Path
from model import CrossAttentionModel

MODEL_PATH = Path("./checkpoints/best_model_cross.pt")
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L12-v2"

app = FastAPI()

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el tokenizador y el modelo al iniciar la API
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = CrossAttentionModel(model_name=TOKENIZER_NAME)  # Especificar model_name
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
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
        # Tokenizar las oraciones
        encoded1 = tokenizer(sentences.sentence1, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        encoded2 = tokenizer(sentences.sentence2, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # Extraer tensores y mover al dispositivo
        input_ids1 = encoded1["input_ids"].to(device)
        attention_mask1 = encoded1["attention_mask"].to(device)
        input_ids2 = encoded2["input_ids"].to(device)
        attention_mask2 = encoded2["attention_mask"].to(device)
        
        # Calcular similitud
        with torch.no_grad():
            similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            # Manejar tanto salida escalar como tensor
            if torch.is_tensor(similarity):
                similarity = similarity.item()
        
        return {"similarity": float(similarity)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8005)