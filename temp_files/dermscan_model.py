import tensorflow as tf
import numpy as np
from PIL import Image
import os

class DermScanModel:
    def __init__(self):
        self.model = self._create_model()
        self.class_names = [
            "멜라노마(Melanoma)",
            "기저세포암(Basal Cell Carcinoma)",
            "편평세포암(Squamous Cell Carcinoma)",
            "지루성 각화증(Seborrheic Keratosis)",
            "양성 모반(Benign Nevus)",
            "혈관종(Hemangioma)",
            "피부섬유종(Dermatofibroma)"
        ]
        
    def _create_model(self):
        """
        EfficientNet 기반 모델 생성
        실제 프로덕션에서는 사전 학습된 가중치를 로드해야 함
        """
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # 전이학습을 위해 기본 모델 동결
        base_model.trainable = False
        
        # 새로운 분류 레이어 추가
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_path):
        """
        이미지 전처리 함수
        """
        img = Image.open(image_path)
        img = img.convert('RGB')  # 이미지가 RGBA인 경우 RGB로 변환
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def generate_description(self, image_array, features=None):
        """
        병변 설명 생성 함수
        실제 구현에서는 이미지 특성을 분석하여 설명 생성
        """
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
    
    def predict(self, image_array):
        """
        이미지 분석 및 진단명 예측
        """
        # 모델 예측 (실제 학습된 모델이 없으므로 임의 결과 생성)
        # 실제 구현에서는 self.model.predict(image_array) 사용
        
        # 임의 예측 결과 생성 (실제 구현 시 교체)
        predictions = np.random.dirichlet(np.ones(7), size=1)[0]
        
        # 상위 2개 진단명 선택
        top_indices = predictions.argsort()[-2:][::-1]
        top_probabilities = predictions[top_indices]
        
        # 결과 정규화
        normalized_probs = top_probabilities / np.sum(top_probabilities)
        
        result = [
            {"diagnosis": self.class_names[top_indices[0]], "probability": float(normalized_probs[0])},
            {"diagnosis": self.class_names[top_indices[1]], "probability": float(normalized_probs[1])}
        ]
        
        return result
