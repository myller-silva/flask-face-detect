import cv2
import os
import numpy as np

# Caminho para o diretório com as imagens de treinamento
training_data_dir = 'training_data'

# Cria o reconhecedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar dados de treinamento
def load_training_data(directory):
    faces = []
    ids = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_id = int(filename.split('.')[0])  # Remove a extensão e converte para inteiro
            faces.append(img)
            ids.append(face_id)
    return faces, np.array(ids, dtype=np.int32)  # Certifique-se de que os IDs sejam inteiros

# Carrega as imagens de treinamento
faces, ids = load_training_data(training_data_dir)

# Treina o reconhecedor com as imagens
recognizer.train(faces, ids)

# Salva o modelo treinado
model_path = 'face_recognizer.yml'
recognizer.save(model_path)

print(f"Modelo salvo em {model_path}")