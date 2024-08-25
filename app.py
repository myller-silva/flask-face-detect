from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)

# Caminho relativo para o arquivo Haar Cascade
cascade_path = 'haarcascade_frontalface_default.xml'

# Verifica se o arquivo existe e pode ser lido
if not os.path.exists(cascade_path):
    raise IOError(f'File does not exist: {cascade_path}')
if not os.access(cascade_path, os.R_OK):
    raise IOError(f'File is not readable: {cascade_path}')

# Carrega o modelo de detecção facial
face_cascade = cv2.CascadeClassifier(cascade_path)

# Carrega o modelo de reconhecimento facial
recognizer_path = 'face_recognizer.yml'
if not os.path.exists(recognizer_path):
    raise IOError(f'Model does not exist: {recognizer_path}')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(recognizer_path)

@app.route('/')
def index():
    return render_template('index.html', image_data=None)

@app.route('/upload')
def upload_page():
    return render_template('upload.html')
@app.route('/process_image', methods=['POST'])
def process_image():
    # Recebe a imagem do formulário de upload
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta rostos
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    recognition_results = []

    for (x, y, w, h) in faces:
        # Reconhece o rosto
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Adiciona resultado de reconhecimento
        recognition_results.append({
            'label': int(id_),  # Converte numpy.int32 para int
            'confidence': float(confidence),  # Converte numpy.float32 para float
            'position': [int(x), int(y), int(w), int(h)]  # Converte numpy.int32 para int
        })
        
        # Desenha retângulos ao redor dos rostos detectados com cores diferentes
        color = (0, 255, 0) if int(id_) == 1 else (255, 0, 0)  # Exemplo de cores diferentes
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    
    # Converte a imagem para base64
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    img_data = f"data:image/jpeg;base64,{img_base64}"
    
    # Extraindo IDs dos rostos identificados
    ids = [result['label'] for result in recognition_results]

    return render_template('index.html', image_data=img_data, ids=ids)

