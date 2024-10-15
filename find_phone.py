import torch
import sys
from PIL import Image

# Cargar el modelo YOLOv5 entrenado
def find_phone(image_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local')

    # Realizar la inferencia en la imagen proporcionada
    results = model(image_path)

    # Obtener las dimensiones de la imagen
    img = Image.open(image_path)
    image_width, image_height = img.size

    # Obtener las coordenadas normalizadas del centro del bounding box
    df = results.pandas().xyxy[0]
    
    if not df.empty:
        # Obtener el centro del bounding box
        row_pred = df.iloc[0]  # Si hay varias detecciones, tomamos la primera
        x_center = (row_pred['xmin'] + row_pred['xmax']) / 2
        y_center = (row_pred['ymin'] + row_pred['ymax']) / 2
        
        # Normalizar las coordenadas
        x_normalized = x_center / image_width
        y_normalized = y_center / image_height

        # Imprimir las coordenadas normalizadas
        print(f"{x_normalized:.4f} {y_normalized:.4f}")
    else:
        print("No se detectó el teléfono en la imagen.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python find_phone.py <ruta_imagen>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    find_phone(image_path)
    