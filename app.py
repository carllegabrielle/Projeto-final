# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import cv2
import os

# 1) FastAPI sem docs
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Mount static directory for HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2) Carrega o modelo
MODEL_PATH = "models/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 3) Função de pré-processamento aprimorada
def preprocess_image(data: bytes) -> np.ndarray:
    # 3.1) Lê bytes e converte para grayscale OpenCV
    buf = np.frombuffer(data, np.uint8)
    gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    # 3.2) Reduz ruído com blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 3.3) Otsu threshold (binário inverso: traço branco, fundo preto)
    _, bin_img = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 3.4) Morfologia de fechamento para preencher “buracos” no traço
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3.5) Encontra contornos e extrai o maior
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
    else:
        h, w = closed.shape
        x = y = 0

    # 3.6) Recorta e centraliza
    digit = closed[y:y+h, x:x+w]
    size = max(w,h)
    square = np.zeros((size,size), dtype=np.uint8)
    dx = (size - w)//2
    dy = (size - h)//2
    square[dy:dy+h, dx:dx+w] = digit

    # 3.7) Redimensiona para 64×64
    resized = cv2.resize(square, (64,64), interpolation=cv2.INTER_AREA)

    # 3.8) Normaliza e inverte (traço=1, fundo=0)
    arr = resized.astype(np.float32)/255.0
    arr = 1.0 - arr

    # 3.9) Formato final para a rede: (1,64,64,1)
    return arr.reshape(1,64,64,1)


# 4) Rota raiz
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join("static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# 4.1) Rota de desenho
@app.get("/draw", response_class=HTMLResponse)
async def draw():
    html_path = os.path.join("static", "draw.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# 5) Rota de predição
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Envie um arquivo de imagem.")
    data = await file.read()
    try:
        x = preprocess_image(data)
    except Exception:
        raise HTTPException(400, "Erro ao processar imagem.")
    preds = model.predict(x)
    digit = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return JSONResponse({"digit": digit, "confidence": confidence})

# 6) Run
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
