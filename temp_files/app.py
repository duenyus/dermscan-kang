from flask import Flask, request, jsonify, render_template, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid
import json
from models.dermscan_model import DermScanModel

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 허용된 파일 확장자 확인
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 모델 로드
model = DermScanModel()

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 및 분석 API
@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # 안전한 파일명으로 변경
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # 이미지 전처리
            processed_image = model.preprocess_image(file_path)
            
            # 병변 설명 생성
            description = model.generate_description(processed_image)
            
            # 진단명 예측
            diagnoses = model.predict(processed_image)
            
            # 결과 반환
            result = {
                'image_url': url_for('static', filename=f'uploads/{unique_filename}', _external=True),
                'description': description,
                'diagnoses': diagnoses
            }
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
