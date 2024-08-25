import cv2
import os

# Verifica se o módulo 'face' está disponível
if not hasattr(cv2, 'face'):
    raise ImportError("O módulo cv2.face não está disponível. Certifique-se de ter instalado opencv-contrib-python.")

# Caminho para o modelo LBPH
model_path = 'face_recognizer.yml'

# Verifica se o modelo existe e pode ser lido
if not os.path.exists(model_path):
    raise IOError(f'Model does not exist: {model_path}')

# Carrega o modelo LBPH de reconhecimento facial
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
