import numpy as np
import cv2
import json
from tensorflow import keras
from io import BytesIO
from PIL import Image

class ModelHandler:
    """Manejador del modelo de reconocimiento facial"""
    
    def __init__(self, model_path: str, classes_path: str, metadata_path: str):
        self.model_path = model_path
        self.classes_path = classes_path
        self.metadata_path = metadata_path
        self.model = None
        self.classes = None
        self.metadata = None
        self.img_size = (150, 150)
        
        # Cargar detector de rostros
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Cargar modelo al inicializar
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo, clases y metadatos"""
        try:
            # Cargar modelo
            print(f" Cargando modelo: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print(" Modelo cargado :) ")
            
            # Cargar clases
            print(f" Cargando clases: {self.classes_path}")
            self.classes = np.load(self.classes_path, allow_pickle=True).tolist()
            print(f" Clases: {self.classes}")
            
            # Cargar metadatos
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(" Metadatos cargados")
            except Exception as e:
                print(f"  No se pudieron cargar metadatos: {e}")
                self.metadata = {
                    'personas': self.classes,
                    'img_size': list(self.img_size),
                    'architecture': 'Unknown',
                    'final_val_accuracy': 0.0
                }
            
        except Exception as e:
            print(f" Error al cargar modelo: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado"""
        return self.model is not None and self.classes is not None
    
    def get_classes(self):
        """Obtiene lista de clases"""
        return self.classes if self.classes else []
    
    def get_model_info(self) -> dict:
        """Obtener información del modelo"""
        return {
            "classes": self.classes,
            "num_classes": len(self.classes) if self.classes else 0,
            "img_size": list(self.img_size),
            "architecture": self.metadata.get("architecture", "Unknown"),
            "accuracy": self.metadata.get("final_val_accuracy", 0.0) * 100
        }
    
    def _preprocess_image(self, image_bytes: bytes):
        """
        Preprocesar imagen desde bytes
        
        Returns:
            tuple: (imagen procesada, rostro detectado)
        """
        try:
            # Convertir bytes a imagen PIL
            image = Image.open(BytesIO(image_bytes))
            
            # Convertir a array numpy (grayscale)
            img_array = np.array(image.convert('L'))
            
        except Exception as e:
            raise ValueError(f"Error al leer imagen: {str(e)}")
        
        # Detectar rostro
        faces = self.face_cascade.detectMultiScale(
            img_array, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Si se detecta rostro, recortar con margen
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            margin = int(max(w, h) * 0.3)
            
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_array.shape[1], x + w + margin)
            y2 = min(img_array.shape[0], y + h + margin)
            
            rostro = img_array[y1:y2, x1:x2]
            face_detected = True
        else:
            rostro = img_array
            face_detected = False
        
        # Redimensionar a tamaño del modelo
        img_resized = cv2.resize(rostro, self.img_size, interpolation=cv2.INTER_CUBIC)
        
        # Agregar dimensiones (batch, height, width, channels)
        img_processed = np.expand_dims(img_resized, axis=-1)  # Canal
        img_processed = np.expand_dims(img_processed, axis=0)  # Batch
        
        return img_processed, face_detected
    
    def predict(self, image_bytes: bytes, threshold: float = 0.70) -> dict:
        """
        Hacer predicción sobre una imagen
        
        Args:
            image_bytes: Bytes de la imagen
            threshold: Umbral de confianza (0.0 - 1.0)
        
        Returns:
            Diccionario con resultado de predicción
        """
        if not self.is_loaded():
            raise ValueError("Modelo no cargado")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold debe estar entre 0.0 y 1.0")
        
        # Preprocesar imagen
        try:
            img_processed, face_detected = self._preprocess_image(image_bytes)
        except Exception as e:
            raise ValueError(f"Error al procesar imagen: {str(e)}")
        
        # Hacer predicción
        prediction = self.model.predict(img_processed, verbose=0)[0]
        
        # Para clasificación binaria (2 clases)
        if len(prediction) == 1:
            # Sigmoid output
            prob_clase_1 = float(prediction[0])
            prob_clase_0 = 1 - prob_clase_1
            confidence = max(prob_clase_0, prob_clase_1)
            predicted_idx = 0 if prob_clase_0 > prob_clase_1 else 1
            probabilities = {
                self.classes[0]: prob_clase_0,
                self.classes[1]: prob_clase_1
            }
        else:
            # Softmax output (para más de 2 clases)
            predicted_idx = int(np.argmax(prediction))
            confidence = float(prediction[predicted_idx])
            probabilities = {
                self.classes[i]: float(prediction[i])
                for i in range(len(self.classes))
            }
        
        # Aplicar umbral
        if confidence < threshold:
            predicted_name = "DESCONOCIDO"
            is_unknown = True
        else:
            predicted_name = self.classes[predicted_idx]
            is_unknown = False
        
        # Construir respuesta
        return {
            "predicted_person": predicted_name,
            "confidence": round(confidence * 100, 2),
            "is_unknown": is_unknown,
            "threshold": threshold * 100,
            "face_detected": face_detected,
            "probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()},
            "all_classes": self.classes
        }
