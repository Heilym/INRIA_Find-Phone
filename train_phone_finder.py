import os
import torch
import yaml
import shutil
import sys

width = 0.0816   # Asignar un valor predeterminado al ancho del celular
height = 0.1226  # Asignar un valor predeterminado al alto del celular

# Crear la estructura de carpetas que YOLOv5 espera
def create_yolo_dataset_structure(data_dir):
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    
    # Crear carpetas
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # Leer el archivo labels.txt para generar los archivos de etiquetas
    label_file = os.path.join(data_dir, 'labels.txt')
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # Mover las imágenes a la carpeta 'images' y crear los archivos de etiquetas en 'labels'
    for line in lines:
        image_file, x_center, y_center = line.split()
        # Mover imagen a images/
        image_path = os.path.join(data_dir, image_file)
        shutil.move(image_path, os.path.join(image_dir, image_file))
        
        # Crear archivo de etiqueta en formato YOLO
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
        with open(label_path, 'w') as label_out:
            label_out.write(f'0 {x_center} {y_center} {width} {height}\n')

# Crear el archivo data.yaml para YOLOv5
def create_data_yaml(data_dir):
    data_yaml = {
        'train': os.path.join(data_dir, 'images'),
        'nc': 1,  # Número de clases
        'names': ['phone']  # Nombre de la clase
    }
    with open(os.path.join(data_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
        
# Entrenar el modelo YOLOv5 en el conjunto de datos proporcionado.
def train_yolo(data_dir):
    create_yolo_dataset_structure(data_dir)
    create_data_yaml(data_dir)

    # Cargar YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Entrenar el modelo
    model.train(data=os.path.join(data_dir, 'data.yaml'), epochs=20, imgsz=640)

    # Guardar los pesos entrenados
    model.save(os.path.join(data_dir, 'best.pt'))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python train_phone_finder.py <ruta_carpeta_datos>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    train_yolo(data_dir)
    