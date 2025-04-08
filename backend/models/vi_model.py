import numpy as np
from PIL import Image
import tensorflow as tf
import logging
from transformers import (
    TFAutoModelForImageClassification,
    AutoImageProcessor,
    AutoConfig
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


class ViModel:
    def __init__(self, model_name="Anwarkh1/Skin_Cancer-Image_Classification"):
        """
        ViT 기반 피부 병변 분류 모델 초기화
        """
        self.model_name = model_name
        self.current_image_path = None

        # 클래스명 정의
        self.class_names = [
            "멜라노마(Melanoma)",
            "기저세포암(Basal Cell Carcinoma)",
            "편평세포암(Squamous Cell Carcinoma)",
            "지루성 각화증(Seborrheic Keratosis)",
            "양성 모반(Benign Nevus)",
            "혈관종(Vascular Lesion)",
            "피부섬유종(Dermatofibroma)"
        ]

        # 직접 매핑 정의
        self.direct_mapping = {
            "benign_keratosis-like_lesions": "지루성 각화증(Seborrheic Keratosis)",
            "basal_cell_carcinoma": "기저세포암(Basal Cell Carcinoma)",
            "actinic_keratoses": "편평세포암(Squamous Cell Carcinoma)",
            "vascular_lesions": "혈관종(Vascular Lesion)",
            "melanocytic_Nevi": "양성 모반(Benign Nevus)",
            "melanoma": "멜라노마(Melanoma)",
            "dermatofibroma": "피부섬유종(Dermatofibroma)"
        }

        try:
            logger.info(f"모델 로드 시도: {model_name}")

            # 모델 구성 명시적 설정
            config = AutoConfig.from_pretrained(model_name)
            config.id2label = {
                0: "benign_keratosis-like_lesions",
                1: "basal_cell_carcinoma",
                2: "actinic_keratoses",
                3: "vascular_lesions",
                4: "melanocytic_Nevi",
                5: "melanoma",
                6: "dermatofibroma"
            }
            config.label2id = {label: idx for idx, label in config.id2label.items()}

            # 프로세서 로드
            self.processor = AutoImageProcessor.from_pretrained(model_name)

            # 명시적 구성으로 모델 로드
            self.model = TFAutoModelForImageClassification.from_pretrained(
                model_name,
                config=config
            )

            # 로깅
            logger.info("모델 구성 정보:")
            logger.info(f"  label2id: {self.model.config.label2id}")
            logger.info(f"  id2label: {self.model.config.id2label}")

            # 클래스 매핑 생성
            self.hf_to_our_mapping = self._create_class_mapping()

        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise

    def _create_class_mapping(self):
        """
        Hugging Face 모델의 클래스를 우리 클래스에 매핑
        """
        mapping = {}

        # 모델의 모든 레이블 출력
        logger.info("모델의 전체 레이블:")
        for idx, hf_label in self.model.config.id2label.items():
            logger.info(f"  {idx}: {hf_label}")

        for idx, hf_label in self.model.config.id2label.items():
            # 직접 매핑 우선 적용
            mapped_class = self.direct_mapping.get(hf_label, f"기타({hf_label})")
            mapping[idx] = mapped_class

        # 매핑 결과 로깅
        logger.info("클래스 매핑 결과:")
        for idx, our_class in mapping.items():
            logger.info(f"  {self.model.config.id2label[idx]} -> {our_class}")

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

            # 이미지 경로 저장 (크기 측정에 사용)
            self.current_image_path = image_path

            # Hugging Face 모델용 이미지 전처리
            inputs = self.processor(images=img, return_tensors="tf")
            logger.info(f"이미지 전처리 완료: {list(inputs.keys())}")
            return inputs

        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {str(e)}")
            # 오류 발생 시 더미 이미지 반환
            return None

    def generate_description(self, image_input):
        """
        병변 설명 생성 함수
        """
        try:
            # 모델 예측
            outputs = self.model(**image_input)
            logits = outputs.logits

            # 로짓 처리
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]

            # 최상위 예측 클래스
            top_idx = np.argmax(probabilities)
            top_prob = probabilities[top_idx]
            top_class = self.model.config.id2label[top_idx]

            # 클래스 매핑
            mapped_class = self.hf_to_our_mapping.get(top_idx, f"기타({top_class})")

            # 질환별 설명 사전
            descriptions_map = {
                "멜라노마(Melanoma)": {
                    "description": "불규칙한 경계와 다양한 색조를 가진 비대칭적인 피부 병변입니다. 색상이 불균일하고 경계가 명확하지 않을 수 있으며, 크기는 보통 6-10mm 정도입니다.",
                    "risk": "악성 종양으로, 조기 발견과 전문의 진단이 매우 중요합니다."
                },
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
                "혈관종(Vascular Lesion)": {
                    "description": "붉은색 또는 보라색을 띄는 혈관성 병변으로, 대칭적이고 표면이 매끄럽습니다. 크기는 보통 4-7mm 정도이며, 특히 영유아기에 흔히 발생합니다.",
                    "risk": "대부분 양성이며 대개 자연적으로 사라집니다."
                },
                "피부섬유종(Dermatofibroma)": {
                    "description": "단단하고 작은 갈색 또는 진한 갈색의 피부 병변으로, 대칭적이며 경계가 명확합니다. 크기는 보통 3-5mm 정도입니다.",
                    "risk": "양성 종양으로 암으로 진행될 가능성은 매우 낮습니다."
                }
            }

            # 기본 설명 선택 (기본값으로 양성 모반 사용)
            description_info = descriptions_map.get(mapped_class, descriptions_map["양성 모반(Benign Nevus)"])

            # 신뢰도에 따른 문구 추가
            confidence_phrase = (
                f"높은 신뢰도로 {mapped_class}(으)로 추정됩니다. " if top_prob > 0.7 else
                f"잠정적으로 {mapped_class}(으)로 추정됩니다. "
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

            # 로짓 값 직접 처리
            def process_logits(logits, threshold=0.1):
                # 로짓에 소프트맥스 적용
                probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]

                # 최대값 강조
                max_prob = np.max(probabilities)
                emphasized_probs = np.where(
                    probabilities >= max_prob * threshold,
                    probabilities,
                    0
                )

                # 재정규화
                total = np.sum(emphasized_probs)
                if total > 0:
                    emphasized_probs /= total

                return emphasized_probs

            # 로짓 처리
            probabilities = process_logits(logits)

            # 결과 생성
            result = []
            for i, prob in enumerate(probabilities):
                if prob > 0:
                    class_name = self.hf_to_our_mapping.get(i, f"기타({self.model.config.id2label[i]})")
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