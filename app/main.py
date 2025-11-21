from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os
import base64

from .model_handler import ModelHandler
from .utils import allowed_file

# Inicializar FastAPI
app = FastAPI(
    title="IABunnies Face Recognition API",
    description="API para reconocimiento facial con TensorFlow/Keras",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar modelo
model_handler = ModelHandler(
    model_path="model/modelo_final.h5",
    classes_path="model/classes.npy",
    metadata_path="model/metadata.json"
)

# =============== MODELOS PYDANTIC ===============
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str]

class PredictBase64Request(BaseModel):
    image_base64: str
    threshold: float = 0.70

# =============== ENDPOINTS ===============

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "IABunnies Face Recognition API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model_handler.is_loaded(),
        "python_version": "3.8",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST - multipart + query)",
            "predict_simple": "/predict/simple (POST - solo multipart)",
            "predict_base64": "/predict/base64 (POST - JSON base64)",
            "info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model_handler.is_loaded(),
        "classes": model_handler.get_classes()
    }

@app.get("/model/info")
async def model_info():
    """Información del modelo"""
    try:
        info = model_handler.get_model_info()
        return {
            "status": "success",
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_face(
    file: UploadFile = File(...),
    threshold: float = Query(0.70, ge=0.0, le=1.0, description="Umbral de confianza")
):
    """
    Predecir persona en imagen (threshold en query param)
    
    **Uso:**
    ```bash
    curl -X POST "http://localhost:8000/predict?threshold=0.70" \
      -F "file=@imagen.jpg"
    ```
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó archivo")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Formato no permitido. Use: JPG, JPEG, PNG, BMP"
        )
    
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        
        result = model_handler.predict(contents, threshold=threshold)
        
        return {
            "status": "success",
            "data": result
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/predict/simple")
async def predict_face_simple(file: UploadFile = File(...)):
    """
    Predicción simple - Solo archivo (threshold = 0.70 por defecto)
    
    **Uso en ATAC:**
    POST /predict/simple
    Body (multipart/form-data): file=[archivo]
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó archivo")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Formato no permitido. Use: JPG, JPEG, PNG, BMP"
        )
    
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        
        result = model_handler.predict(contents, threshold=0.70)
        
        return {
            "status": "success",
            "data": result
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/predict/base64")
async def predict_face_base64(request: PredictBase64Request):
    """
    Predicción con imagen en base64 (para clientes con problemas de multipart)
    
    **Body (JSON):**
    ```json
    {
      "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "threshold": 0.70
    }
    ```
    """
    try:
        # Extraer base64 (quitar prefijo si existe)
        if ',' in request.image_base64:
            image_base64 = request.image_base64.split(',')[1]
        else:
            image_base64 = request.image_base64
        
        # Decodificar
        image_bytes = base64.b64decode(image_base64)
        
        # Predecir
        result = model_handler.predict(image_bytes, threshold=request.threshold)
        
        return {
            "status": "success",
            "data": result
        }
    
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Base64 inválido")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    threshold: float = Query(0.70, ge=0.0, le=1.0)
):
    """Predecir múltiples imágenes"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Máximo 10 imágenes")
    
    results = []
    
    for file in files:
        if not file.filename:
            results.append({
                "filename": "unknown",
                "status": "error",
                "message": "Sin nombre"
            })
            continue
        
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Formato no permitido"
            })
            continue
        
        try:
            contents = await file.read()
            prediction = model_handler.predict(contents, threshold=threshold)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "data": prediction
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {
        "status": "success",
        "total": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results
    }

# =============== MANEJO DE ERRORES ===============

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "Endpoint no encontrado",
            "path": str(request.url)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Error interno del servidor"
        }
    )

# =============== STARTUP ===============

@app.on_event("startup")
async def startup_event():
    print("="*60)
    print(" IA Face Python 3.8")
    print("="*60)
    
    if model_handler.is_loaded():
        info = model_handler.get_model_info()
        print(f" Modelo cargado :) ")
        print(f" Clases: {info['classes']}")
        print(f" Precisión: {info['accuracy']:.2f}%")
    else:
        print(" Error: Modelo no cargado :( ")
    
    print("="*60)

@app.on_event("shutdown")
async def shutdown_event():
    print(" Cerrando API...")

# =============== RUN ===============

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True
    )
