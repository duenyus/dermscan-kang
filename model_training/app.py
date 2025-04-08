import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import base64
from PIL import Image
import io
import tensorflow as tf

# 예측기 모듈 경로 추가
sys.path.append('/home/ubuntu/dermscan/model_training/scripts')
from dermscan_predictor import DermScanPredictor

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청 허용

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 경로 설정
MODEL_DIR = '/home/ubuntu/dermscan/model_training/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 더미 모델 생성 함수
def create_dummy_model():
    # 이미지 크기 설정
    img_size = (224, 224)
    
    # 기본 EfficientNet 모델 생성
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=img_size + (3,)
    )
    
    # 모델 구조 생성
    inputs = tf.keras.layers.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 모델 저장
    dummy_model_path = os.path.join(MODEL_DIR, "dummy_model.h5")
    model.save(dummy_model_path)
    
    return dummy_model_path

# 더미 모델 생성
dummy_model_path = os.path.join(MODEL_DIR, "dummy_model.h5")
if not os.path.exists(dummy_model_path):
    print("더미 모델 파일이 없습니다. 생성합니다...")
    dummy_model_path = create_dummy_model()
    print(f"더미 모델 생성 완료: {dummy_model_path}")

# 클래스 매핑 파일 생성
class_mapping_path = os.path.join(MODEL_DIR, "class_mapping.json")
if not os.path.exists(class_mapping_path):
    print("클래스 매핑 파일이 없습니다. 생성합니다...")
    class_mapping = {
        "0": {"name": "Melanoma", "description": "악성 흑색종"},
        "1": {"name": "Nevus", "description": "양성 모반"},
        "2": {"name": "Seborrheic Keratosis", "description": "지루성 각화증"}
    }
    with open(class_mapping_path, 'w') as f:
        json.dump(class_mapping, f)
    print(f"클래스 매핑 파일 생성 완료: {class_mapping_path}")

# 예측기 초기화
predictor = DermScanPredictor(dummy_model_path, class_mapping_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        # 요청에서 이미지 데이터 추출
        if 'image' in request.files:
            # 파일 업로드 방식
            file = request.files['image']
            img_bytes = file.read()
            
            # 이미지 저장 (선택 사항)
            filename = f"upload_{len(os.listdir(UPLOAD_FOLDER))}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
            
            # 예측
            result = predictor.predict(img_bytes)
            
            # 파일 경로 추가
            result['image_path'] = f"/static/uploads/{filename}"
            
        elif 'imageData' in request.json:
            # Base64 인코딩 방식
            image_data = request.json['imageData']
            
            # Base64 데이터에서 헤더 제거
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # 이미지 저장 (선택 사항)
            img_bytes = base64.b64decode(image_data)
            filename = f"upload_{len(os.listdir(UPLOAD_FOLDER))}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
            
            # 예측
            result = predictor.predict_from_base64(image_data)
            
            # 파일 경로 추가
            result['image_path'] = f"/static/uploads/{filename}"
            
        else:
            return jsonify({
                'success': False,
                'error': '이미지 데이터가 없습니다.'
            }), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    """API 테스트 엔드포인트"""
    return jsonify({
        'success': True,
        'message': 'API가 정상적으로 작동 중입니다.',
        'model_path': dummy_model_path,
        'class_mapping_path': class_mapping_path
    })

if __name__ == '__main__':
    # 서버 실행
    app.run(host='0.0.0.0', port=8080, debug=True)
