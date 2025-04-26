from flask import Flask, request, jsonify, render_template, url_for, redirect, send_from_directory
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid
import json
import logging
import re
import sqlite3
from datetime import datetime

from models.swin_model import SwinModel
from models.vi_model import ViModel

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# SQLite 데이터베이스 설정
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dermscan.db')

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 데이터베이스 초기화 함수
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT NOT NULL,
        model TEXT NOT NULL,
        ip_address TEXT NOT NULL,
        score INTEGER NOT NULL,
        diagnoses TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT NOT NULL,
        model TEXT NOT NULL,
        ip_address TEXT NOT NULL,
        diagnoses TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()
    logger.info("SQLite 데이터베이스가 초기화되었습니다.")

# 데이터베이스 초기화 실행
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 모델 초기화
try:
    logger.info("모델 초기화 시작...")
    models = {}

    try:
        vi_model = ViModel()
        models['vi'] = vi_model
        logger.info("vi 모델이 로드되었습니다.")
    except Exception as e:
        logger.error(f"ViModel 로드 실패: {str(e)}")

    try:
        swin_model = SwinModel()
        models['swin'] = swin_model
        logger.info("swin 모델이 로드되었습니다.")
    except Exception as e:
        logger.error(f"SwinModel 로드 실패: {str(e)}")

    default_model = 'vi' if 'vi' in models else 'swin'

except Exception as e:
    logger.error(f"모델 초기화 오류: {str(e)}")
    models = {}
    default_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # 이미지 분석 로직
            model = models.get(default_model)
            if not model:
                return jsonify({'error': 'No model available'}), 500

            # 이미지 전처리 및 분석
            model = models.get(default_model)
            if not model:
                return jsonify({'error': 'No model available'}), 500

            # 이미지 전처리
            inputs = model.preprocess_image(file_path)
            if inputs is None:
                return jsonify({'error': 'Image preprocessing failed'}), 500

            # 진단 결과 및 설명 생성
            diagnoses = model.predict(inputs)
            description = model.generate_description(inputs)

            result = {
                'image_url': url_for('serve_image', filename=unique_filename, _external=True),
                'description': description,
                'diagnoses': diagnoses
            }

            return jsonify(result)

        except Exception as e:
            logger.error(f"이미지 분석 오류: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting DermScan backend server on port 5000...")
    app.run(host='0.0.0.0', port=5000)