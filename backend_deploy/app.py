import os
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # CORS 설정 강화

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 허용된 파일 확장자 확인
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # 이미지가 RGBA인 경우 RGB로 변환
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 병변 설명 생성 함수
def generate_description(image_array):
    # 실제 구현에서는 이미지 특성 추출 및 분석 필요
    # 현재는 임시 설명 생성
    descriptions = [
        "경계가 불규칙하고 비대칭적인 갈색 병변으로, 다양한 색조를 보이며 직경은 약 8mm입니다.",
        "경계가 명확하고 대칭적인 붉은색 병변으로, 표면이 매끄럽고 직경은 약 5mm입니다.",
        "불규칙한 경계를 가진 검은색 반점으로, 중앙부에 색소 변화가 있으며 직경은 약 10mm입니다.",
        "경계가 명확한 붉은색 구진으로, 표면이 약간 융기되어 있고 직경은 약 7mm입니다.",
        "불규칙한 형태의 갈색 반점으로, 여러 색조가 혼합되어 있으며 직경은 약 12mm입니다.",
        "대칭적인 형태의 검은색 병변으로, 경계가 명확하고 직경은 약 6mm입니다.",
        "불규칙한 경계와 비대칭적 형태를 가진 다색성 병변으로, 직경은 약 9mm입니다."
    ]
    
    # 실제 모델에서는 이미지 특성에 따라 설명 생성
    # 현재는 임의 선택
    return np.random.choice(descriptions)

# 진단명 예측 함수
def predict_diagnoses(image_array):
    # 임시 진단명 (실제 구현 시 교체)
    diagnoses = [
        "멜라노마(Melanoma)",
        "기저세포암(Basal Cell Carcinoma)",
        "편평세포암(Squamous Cell Carcinoma)",
        "지루성 각화증(Seborrheic Keratosis)",
        "양성 모반(Benign Nevus)",
        "혈관종(Hemangioma)",
        "피부섬유종(Dermatofibroma)"
    ]
    
    # 임시로 랜덤하게 2개 선택 (실제 구현 시 교체)
    predictions = np.random.dirichlet(np.ones(7), size=1)[0]
    
    # 상위 2개 진단명 선택
    top_indices = predictions.argsort()[-2:][::-1]
    top_probabilities = predictions[top_indices]
    
    # 결과 정규화
    normalized_probs = top_probabilities / np.sum(top_probabilities)
    
    result = [
        {"diagnosis": diagnoses[top_indices[0]], "probability": float(normalized_probs[0])},
        {"diagnosis": diagnoses[top_indices[1]], "probability": float(normalized_probs[1])}
    ]
    
    return result

# 정적 파일 제공
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 테스트 엔드포인트
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "ok", "message": "API is working"}), 200

# 이미지 업로드 및 분석 API
@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze_image():
    # CORS preflight 요청 처리
    if request.method == 'OPTIONS':
        return '', 204
        
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
            processed_image = preprocess_image(file_path)
            
            # 병변 설명 생성
            description = generate_description(processed_image)
            
            # 진단명 예측
            diagnoses = predict_diagnoses(processed_image)
            
            # 결과 반환
            result = {
                'image_url': f"/static/uploads/{unique_filename}",
                'description': description,
                'diagnoses': diagnoses
            }
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    print("Starting DermScan backend server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=True)
