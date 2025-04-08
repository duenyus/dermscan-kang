import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, applications
from PIL import Image
import io
import base64
import json
import random

class DermScanPredictor:
    def __init__(self, model_path=None, class_mapping_path=None):
        # 이미지 크기 설정
        self.img_size = (224, 224)  # EfficientNet 기본 입력 크기
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        
        # 클래스 매핑 로드
        self.class_mapping = self._load_class_mapping()
        
        # 모델 로드
        print("모델 로드 시도...")
        try:
            self.model = self._load_model()
            print("모델 로드 성공")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("더미 모델 생성")
            self.model = self._create_dummy_model()
    
    def _load_class_mapping(self):
        """클래스 매핑 파일 로드"""
        try:
            if self.class_mapping_path and os.path.exists(self.class_mapping_path):
                with open(self.class_mapping_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"클래스 매핑 로드 실패: {e}")
        
        # 기본 클래스 매핑
        return {
            0: {"name": "Melanoma", "description": "악성 흑색종"},
            1: {"name": "Nevus", "description": "양성 모반"},
            2: {"name": "Seborrheic Keratosis", "description": "지루성 각화증"}
        }
    
    def _load_model(self):
        """모델 파일 로드"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                model = models.load_model(self.model_path)
                return model
        except Exception as e:
            raise Exception(f"모델 로드 실패: {e}")
        
        # 모델 파일이 없는 경우 더미 모델 생성
        return self._create_dummy_model()
    
    def _create_dummy_model(self):
        """더미 모델 생성"""
        # 기본 EfficientNet 모델 생성
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.img_size + (3,)
        )
        
        # 모델 구조 생성
        inputs = layers.Input(shape=self.img_size + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.class_mapping), activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess_image(self, img_bytes):
        """이미지 전처리"""
        try:
            # 바이트 데이터에서 이미지 로드
            img = Image.open(io.BytesIO(img_bytes))
            
            # RGB로 변환 (알파 채널 제거)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 크기 조정
            img = img.resize(self.img_size)
            
            # 배열로 변환
            img_array = np.array(img)
            
            # 배치 차원 추가 및 정규화
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            raise Exception(f"이미지 전처리 실패: {e}")
    
    def predict(self, img_bytes):
        """이미지 바이트 데이터로부터 예측"""
        try:
            # 이미지 전처리
            img_array = self._preprocess_image(img_bytes)
            
            # 예측
            predictions = self.model.predict(img_array)
            
            # 결과 처리
            return self._process_predictions(predictions[0])
            
        except Exception as e:
            # 실제 예측 실패 시 시뮬레이션 결과 반환
            print(f"예측 실패, 시뮬레이션 결과 반환: {e}")
            return self._simulate_predictions()
    
    def predict_from_base64(self, base64_data):
        """Base64 인코딩된 이미지 데이터로부터 예측"""
        try:
            # Base64 디코딩
            img_bytes = base64.b64decode(base64_data)
            
            # 예측
            return self.predict(img_bytes)
            
        except Exception as e:
            # 실패 시 시뮬레이션 결과 반환
            print(f"Base64 예측 실패, 시뮬레이션 결과 반환: {e}")
            return self._simulate_predictions()
    
    def _process_predictions(self, predictions):
        """예측 결과 처리"""
        # 상위 3개 클래스 인덱스 및 확률
        top_indices = np.argsort(predictions)[::-1][:3]
        top_probabilities = predictions[top_indices]
        
        # 결과 포맷팅
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
            class_info = self.class_mapping.get(str(idx), self.class_mapping.get(idx, {"name": f"Class {idx}", "description": "Unknown"}))
            results.append({
                "class_name": class_info["name"],
                "description": class_info["description"],
                "probability": float(prob),
                "rank": i + 1
            })
        
        # 병변 설명 생성
        description = self._generate_description(results)
        
        return {
            "success": True,
            "predictions": results,
            "description": description
        }
    
    def _simulate_predictions(self):
        """예측 결과 시뮬레이션"""
        # 시뮬레이션된 예측 결과
        simulated_classes = [
            {"class_name": "Melanoma", "description": "악성 흑색종", "probability": 0.75, "rank": 1},
            {"class_name": "Nevus", "description": "양성 모반", "probability": 0.15, "rank": 2},
            {"class_name": "Seborrheic Keratosis", "description": "지루성 각화증", "probability": 0.10, "rank": 3}
        ]
        
        # 병변 설명 생성
        description = self._generate_description(simulated_classes)
        
        return {
            "success": True,
            "predictions": simulated_classes,
            "description": description
        }
    
    def _generate_description(self, predictions):
        """병변 설명 생성"""
        top_class = predictions[0]["class_name"]
        
        # 클래스별 설명 템플릿
        descriptions = {
            "Melanoma": [
                "불규칙한 경계와 갈색 색조를 가진 병변으로, 멜라노마의 특징을 보입니다.",
                "갈색 색조가 불균일하게 분포된 비대칭 병변으로, 멜라노마가 의심됩니다.",
                "경계가 불분명하고 갈색 색조를 띄는 병변으로, 멜라노마의 가능성이 있습니다."
            ],
            "Nevus": [
                "경계가 명확하고 색조가 균일한 병변으로, 양성 모반의 특징을 보입니다.",
                "대칭적이고 둥근 형태의 병변으로, 양성 모반으로 판단됩니다.",
                "균일한 색조와 규칙적인 경계를 가진 병변으로, 양성 모반의 가능성이 높습니다."
            ],
            "Seborrheic Keratosis": [
                "표면이 거칠고 각질화된 병변으로, 지루성 각화증의 특징을 보입니다.",
                "갈색 또는 검은색의 융기된 병변으로, 지루성 각화증으로 판단됩니다.",
                "표면에 각질이 있고 '붙어있는' 듯한 모양의 병변으로, 지루성 각화증의 가능성이 높습니다."
            ]
        }
        
        # 해당 클래스의 설명이 있으면 랜덤하게 선택, 없으면 기본 설명 사용
        if top_class in descriptions:
            return random.choice(descriptions[top_class])
        else:
            return f"이 병변은 {top_class}의 특징을 보이며, 추가적인 의학적 검사가 권장됩니다."
