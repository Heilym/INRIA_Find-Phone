import os
import torch
import yaml
import shutil
import sys
import subprocess

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
    
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"No se encontró el archivo labels.txt en {label_file}")
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # Mover las imágenes a la carpeta 'images' y crear los archivos de etiquetas en 'labels'
    for line in lines:
        image_file, x_center, y_center = line.split()
        # Mover imagen a images/
        image_path = os.path.join(data_dir, image_file)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen {image_path}")
        
        shutil.move(image_path, os.path.join(image_dir, image_file))
        
        # Crear archivo de etiqueta en formato YOLO
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
        with open(label_path, 'w') as label_out:
            label_out.write(f'0 {x_center.strip()} {y_center.strip()} {width} {height}\n') 

# Crear el archivo data.yaml para YOLOv5:
def create_data_yaml(data_dir):
    data_yaml = {
        'train': '../'+os.path.join(data_dir, 'images'),
        'val': '../'+os.path.join(data_dir, 'images'),
        'nc': 1,  # Número de clases (en este caso, solo teléfono)
        'names': ['phone']  # Nombre de la clase
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
            
# Entrenar el modelo YOLOv5 en el conjunto de datos proporcionado
def train_yolo(data_dir):
    create_yolo_dataset_structure(data_dir)
    create_data_yaml(data_dir)

    # Ejecutar el script train.py de YOLOv5 usando subprocess
    yolov5_dir = './yolov5'  # Ruta a la carpeta clonada de YOLOv5

    command = [
        'python', os.path.join(yolov5_dir, 'train.py'),
        '--img', '640',  # Tamaño de la imagen
        '--batch', '16',  # Tamaño del batch
        '--epochs', '50',  # Número de épocas
        '--data', 'data.yaml',  # Ruta al archivo data.yaml
        '--cfg', os.path.join(yolov5_dir, 'models', 'yolov5s.yaml'),  # Ruta al archivo de configuración del modelo
        '--weights', 'yolov5s',  # Modelo preentrenado
        '--project', data_dir,  # Carpeta donde se guardarán los resultados
        '--name', 'finetuned'  # Nombre del experimento
    ]

    # Ejecutar el comando para entrenar el modelo
    subprocess.run(command, check=True)

if __name__ == '__main__':
    # Asegurarse de que la carpeta de datos ha sido proporcionada como argumento
    if len(sys.argv) != 2:
        print("Uso: python train_phone_finder.py <ruta_carpeta_datos>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    train_yolo(data_dir)
