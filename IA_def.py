from imageai.Detection import ObjectDetection
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Configuración para ImageAI
detector = ObjectDetection()
model_path_yolo = "./yolov3.pt"  # Ruta al modelo YOLOv3
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path_yolo)
detector.loadModel()

# Configuración para Keras
model_path_keras = "./keras_model.h5"
labels_path = "./labels.txt"

# Cargar modelo Keras y etiquetas
try:
    model_keras = load_model(model_path_keras, compile=False)
    class_names = [line.strip() for line in open(labels_path, "r", encoding="utf-8").readlines()]
except Exception as e:
    raise ValueError(f"Error al cargar el modelo o las etiquetas: {e}")

def detect_objects(image_path):
    """
    Detecta objetos en la imagen utilizando ImageAI.
    Retorna una lista de objetos detectados y la imagen procesada.
    """
    output_path = "./detected_image.jpg"
    detections = detector.detectObjectsFromImage(
        input_image=image_path,
        output_image_path=output_path,
        minimum_percentage_probability=50
    )
    detected_objects = [detection['name'] for detection in detections]
    return detected_objects, output_path

def classify_image(image_path):
    """
    Clasifica una imagen utilizando el modelo Keras.
    Retorna el nombre de la clase, el puntaje de confianza y todas las predicciones.
    """
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Realizar la predicción
    prediction = model_keras.predict(data)
    all_predictions = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])
    
    return class_name, confidence_score, all_predictions

def validate_labels():
    """
    Verifica que las etiquetas en labels.txt sean únicas y estén correctamente definidas.
    """
    if len(class_names) != len(set(class_names)):
        raise ValueError("Existen etiquetas duplicadas en labels.txt.")
    if any(not label for label in class_names):
        raise ValueError("Se encontró una etiqueta vacía en labels.txt.")
