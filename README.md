# INRIA_Find-Phone

### Dependencias:
Python 3.9.5
#### Instalar las dependencias:
pip install -r requirements.txt
#### Clonar el repositorio de YOLOv5
git clone https://github.com/ultralytics/yolov5.git

### Pasos para entrenar el modelo:
Colocar el archivo labels.txt y las imágenes en una carpeta (por ejemplo, ~/find_phone_data).
#### Ejecutar el script de entrenamiento:
python train_phone_finder.py ~/find_phone_data

### Pasos para realizar la predicción:
Después de entrenar el modelo, usar el script de predicción para detectar el teléfono en una imagen de prueba.
#### Ejecutar el script de predicción:
python find_phone.py ~/test_images/51.jpg
