import logging

import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dermscan_model.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class SwinModel:
    def __init__(self, model_name="NeuronZero/SkinCancerClassifier"):
        """
        Swin Transformer 기반 피부 병변 분류 모델 초기화
        """
        self.model_name = model_name
        self.current_image_path = None

        # 클래스명 정의 (영문 코드)
        self.class_codes = {
            "AK": "악성 각화증(Actinic Keratosis)",
            "BCC": "기저세포암(Basal Cell Carcinoma)",
            "BKL": "지루성 각화증(Seborrheic Keratosis)",
            "DF": "피부섬유종(Dermatofibroma)",
            "MEL": "멜라노마(Melanoma)",
            "NV": "양성 모반(Benign Nevus)",
            "SCC": "편평세포암(Squamous Cell Carcinoma)",
            "VASC": "혈관종(Vascular Lesion)"
        }

        # 한글 클래스명 리스트 (UI 표시용)
        self.class_names = [
            "악성 각화증(Actinic Keratosis)",
            "기저세포암(Basal Cell Carcinoma)",
            "지루성 각화증(Seborrheic Keratosis)",
            "피부섬유종(Dermatofibroma)",
            "멜라노마(Melanoma)",
            "양성 모반(Benign Nevus)",
            "편평세포암(Squamous Cell Carcinoma)",
            "혈관종(Vascular Lesion)"
        ]

        try:
            logger.info(f"Swin 모델 로드 시도: {model_name}")

            # 프로세서 및 모델 로드
            self.processor = AutoImageProcessor.from_pretrained(model_name, init_empty_weights=True)
            logger.info(f"Swin 모델 로드 시도: AutoImageProcessor")
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            logger.info(f"Swin 모델 로드 시도: AutoModelForImageClassification")
            # 모델 구성 정보 로깅
            logger.info("모델 구성 정보:")
            logger.info(f"  label2id: {self.model.config.label2id}")
            logger.info(f"  id2label: {self.model.config.id2label}")

            # 클래스 매핑 생성
            self.id_to_korean = self._create_class_mapping()

        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise

    def _create_class_mapping(self):
        """
        모델의 클래스 ID를 한글 이름에 매핑
        """
        mapping = {}

        # 모델의 모든 레이블 출력
        logger.info("모델의 전체 레이블:")
        for idx, label in self.model.config.id2label.items():
            logger.info(f"  {idx}: {label}")
            # 영문 코드를 한글 이름으로 매핑
            mapping[int(idx)] = self.class_codes.get(label, f"알 수 없음({label})")

        return mapping

    def preprocess_image(self, image_path):
        """
        이미지 전처리 함수

        :param image_path: 전처리할 이미지 경로
        :return: 전처리된 이미지 입력
        """
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')  # RGBA인 경우 RGB로 변환

            # 이미지 경로 저장
            self.current_image_path = image_path

            # 모델용 이미지 전처리
            inputs = self.processor(images=img, return_tensors="pt")
            logger.info(f"이미지 전처리 완료: {list(inputs.keys())}")
            return inputs

        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {str(e)}")
            return None

    def generate_description(self, image_input):
        """
        병변 설명 생성 함수
        """
        try:
            # 모델 예측
            outputs = self.model(**image_input)
            logits = outputs.logits

            # PyTorch 텐서를 numpy 배열로 변환
            np_logits = logits.detach().cpu().numpy()
            probabilities = tf.nn.softmax(np_logits, axis=-1).numpy()[0]
            # 최상위 예측 클래스
            top_idx = np.argmax(probabilities)
            top_prob = probabilities[top_idx]
            logger.info(self.model.config.id2label)
            logger.info(str(top_idx))
            top_class_code = self.model.config.id2label[top_idx]
            top_class_korean = self.id_to_korean[top_idx]
            # 질환별 설명 사전
            descriptions_map = {
                "기저세포암(Basal Cell Carcinoma)": {
                    "description": "주로 햇빛에 노출된 부위에 발생하는 피부 병변으로, 붉은색 또는 분홍빛을 띄며 약간 융기되어 있습니다. 크기는 보통 5-8mm 정도이며, 표면이 유리알 같은 광택을 보일 수 있습니다.",
                    "risk": "가장 흔한 피부암이지만 천천히 자라며 전이 위험이 상대적으로 낮습니다."
                },
                "편평세포암(Squamous Cell Carcinoma)": {
                    "description": "거친 피부 표면과 붉은색 또는 짙은 분홍색의 병변으로 특징지어집니다. 주로 자외선 노출이 많은 부위에 발생하며, 크기는 보통 6-9mm 정도입니다.",
                    "risk": "조기에 발견되면 치료 가능성이 높은 피부암입니다."
                },
                "지루성 각화증(Seborrheic Keratosis)": {
                    "description": "노인성 반점 또는 노인 사마귀라고도 불리며, 갈색 또는 검은색의 약간 융기된 병변입니다. 표면이 거칠고 마치 붙어있는 것처럼 보이며, 크기는 3-7mm 정도입니다.",
                    "risk": "대부분 양성이며 암으로 진행될 가능성은 낮습니다."
                },
                "양성 모반(Benign Nevus)": {
                    "description": "흔히 점 또는 모반이라고 불리는 갈색 반점으로, 대칭적이고 경계가 명확합니다. 크기는 보통 3-6mm 정도이며, 색상과 모양이 균일합니다.",
                    "risk": "대부분 양성이지만 정기적인 관찰이 필요합니다."
                },
                "멜라노마(Melanoma)": {
                    "description": "불규칙한 경계와 다양한 색조를 가진 비대칭적인 피부 병변입니다. 색상이 불균일하고 경계가 명확하지 않을 수 있으며, 크기는 보통 6-10mm 정도입니다.",
                    "risk": "악성 종양으로, 조기 발견과 전문의 진단이 매우 중요합니다."
                },
                "악성 각화증(Actinic Keratosis)": {
                    "description": "피부의 표면에 거칠고 붉은색 또는 갈색의 각질화된 부위로 나타납니다. 주로 오랫동안 햇빛에 노출된 부위에 발생하며, 크기는 3-6mm 정도입니다.",
                    "risk": "일부는 시간이 지남에 따라 편평세포암으로 발전할 수 있어 정기적인 관찰이 중요합니다."
                },
                "피부섬유종(Dermatofibroma)": {
                    "description": "단단하고 작은 갈색 또는 진한 갈색의 피부 병변으로, 대칭적이며 경계가 명확합니다. 크기는 보통 3-5mm 정도입니다.",
                    "risk": "양성 종양으로 암으로 진행될 가능성은 매우 낮습니다."
                },
                "혈관종(Vascular Lesion)": {
                    "description": "붉은색 또는 보라색을 띄는 혈관성 병변으로, 대칭적이고 표면이 매끄럽습니다. 크기는 보통 4-7mm 정도이며, 특히 영유아기에 흔히 발생합니다.",
                    "risk": "대부분 양성이며 대개 자연적으로 사라집니다."
                }
            }

            # 상위 3개 진단 가져오기
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            top_3_probs = probabilities[top_3_idx]
            top_3_classes = [self.id_to_korean[idx] for idx in top_3_idx]

            # 가장 높은 확률의 진단으로 설명 생성
            description_info = descriptions_map.get(top_3_classes[0], descriptions_map["양성 모반(Benign Nevus)"])

            # 신뢰도에 따른 문구 추가
            confidence_phrase = (
                f"높은 신뢰도로 {top_3_classes[0]}(으)로 추정됩니다. " if top_3_probs[0] > 0.7 else
                f"잠정적으로 {top_3_classes[0]}(으)로 추정됩니다. "
            )

            # 최종 설명 구성
            final_description = (
                f"{confidence_phrase}{description_info['description']} "
                f"{description_info['risk']} "
                "정확한 진단을 위해서는 피부과 전문의와의 상담을 권장합니다."
            )

            # 로깅
            logger.info(f"생성된 병변 설명: {final_description}")

            return final_description

        except Exception as e:
            logger.error(f"설명 생성 중 오류 발생: {str(e)}")
            return "피부 병변이 관찰되었으며, 전문의와의 상담이 필요합니다."

    def predict(self, image_input):
        """
        이미지 분석 및 진단명 예측
        """
        try:
            # 모델 예측
            outputs = self.model(**image_input)
            logits = outputs.logits

            # PyTorch 텐서를 numpy 배열로 변환
            np_logits = logits.detach().cpu().numpy()
            probabilities = tf.nn.softmax(np_logits, axis=-1).numpy()[0]

            # 로짓 값 직접 처리
            def process_logits(probs, threshold=0.1):
                # 최대값 강조
                max_prob = np.max(probs)
                emphasized_probs = np.where(
                    probs >= max_prob * threshold,
                    probs,
                    0
                )

                # 재정규화
                total = np.sum(emphasized_probs)
                if total > 0:
                    emphasized_probs /= total

                return emphasized_probs

            # 로짓 처리
            processed_probs = process_logits(probabilities)

            # 결과 생성
            result = []
            for i, prob in enumerate(processed_probs):
                if prob > 0:
                    class_name = self.id_to_korean[i]
                    result.append({
                        "diagnosis": class_name,
                        "probability": float(prob)
                    })

            # 확률 기준 내림차순 정렬
            result.sort(key=lambda x: x["probability"], reverse=True)

            # 로깅
            logger.info("예측 결과 상세:")
            for item in result:
                logger.info(f"  {item['diagnosis']}: {item['probability'] * 100:.2f}%")

            return result

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            return []