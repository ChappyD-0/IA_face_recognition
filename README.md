# IA Face Recognition API

API REST para reconocimiento facial usando TensorFlow/Keras.

## Ejecutar Localmente

### Con Docker

```bash
# Construir imagen
docker build -t iabunnies-api .

# Ejecutar contenedor
docker run -p 8000:8000 iabunnies-api
