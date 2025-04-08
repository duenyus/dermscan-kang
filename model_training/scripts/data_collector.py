import os
import requests
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataCollector:
    def __init__(self, data_dir):
        """
        데이터 수집 및 전처리를 위한 클래스
        
        Args:
            data_dir (str): 데이터를 저장할 디렉토리 경로
        """
        self.data_dir = data_dir
        self.ham10000_dir = os.path.join(data_dir, 'HAM10000')
        self.isic_dir = os.path.join(data_dir, 'ISIC')
        
        # 필요한 디렉토리 생성
        os.makedirs(self.ham10000_dir, exist_ok=True)
        os.makedirs(self.isic_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
    
    def download_ham10000(self):
        """HAM10000 데이터셋 다운로드"""
        print("HAM10000 데이터셋 다운로드 중...")
        
        # 이미지 데이터 다운로드
        image_url = "https://dataverse.harvard.edu/api/access/datafile/3172570"
        image_path = os.path.join(self.ham10000_dir, "HAM10000_images.zip")
        
        if not os.path.exists(image_path):
            print("이미지 파일 다운로드 중...")
            self._download_file(image_url, image_path)
            
            # 압축 해제
            with zipfile.ZipFile(image_path, 'r') as zip_ref:
                zip_ref.extractall(self.ham10000_dir)
            print("이미지 압축 해제 완료")
        else:
            print("이미지 파일이 이미 존재합니다.")
        
        # 메타데이터 다운로드
        metadata_url = "https://dataverse.harvard.edu/api/access/datafile/3172571"
        metadata_path = os.path.join(self.ham10000_dir, "HAM10000_metadata.csv")
        
        if not os.path.exists(metadata_path):
            print("메타데이터 파일 다운로드 중...")
            self._download_file(metadata_url, metadata_path)
            print("메타데이터 다운로드 완료")
        else:
            print("메타데이터 파일이 이미 존재합니다.")
        
        return self.ham10000_dir
    
    def download_isic_subset(self, limit=1000):
        """
        ISIC Archive에서 일부 데이터 다운로드 (API 제한으로 인해 일부만 다운로드)
        
        Args:
            limit (int): 다운로드할 이미지 수
        """
        print(f"ISIC Archive에서 {limit}개의 이미지 다운로드 중...")
        
        # ISIC API를 통해 이미지 목록 가져오기
        api_url = f"https://api.isic-archive.com/api/v1/image?limit={limit}&sort=name&sortdir=1&detail=true"
        
        try:
            response = requests.get(api_url)
            images = response.json()
            
            # 이미지 다운로드
            for i, image in enumerate(tqdm(images[:limit], desc="ISIC 이미지 다운로드")):
                image_id = image['_id']
                image_url = f"https://api.isic-archive.com/api/v1/image/{image_id}/download"
                image_path = os.path.join(self.isic_dir, f"{image_id}.jpg")
                
                if not os.path.exists(image_path):
                    self._download_file(image_url, image_path)
                
                # 메타데이터 저장
                if i % 50 == 0:  # 50개마다 저장
                    metadata_df = pd.DataFrame(images[:i+1])
                    metadata_df.to_csv(os.path.join(self.isic_dir, "isic_metadata.csv"), index=False)
            
            # 최종 메타데이터 저장
            metadata_df = pd.DataFrame(images)
            metadata_df.to_csv(os.path.join(self.isic_dir, "isic_metadata.csv"), index=False)
            
            print(f"ISIC 데이터 다운로드 완료: {len(images)}개 이미지")
            return self.isic_dir
            
        except Exception as e:
            print(f"ISIC 데이터 다운로드 중 오류 발생: {e}")
            print("샘플 데이터를 생성합니다.")
            self._create_sample_data()
            return self.isic_dir
    
    def _create_sample_data(self, num_samples=100):
        """
        API 접근이 어려울 경우 샘플 데이터 생성
        
        Args:
            num_samples (int): 생성할 샘플 수
        """
        print(f"{num_samples}개의 샘플 데이터 생성 중...")
        
        # 샘플 이미지 생성 (단색 이미지)
        os.makedirs(os.path.join(self.isic_dir, "samples"), exist_ok=True)
        
        # 진단 클래스 정의
        diagnoses = [
            "melanoma", 
            "nevus", 
            "seborrheic_keratosis", 
            "basal_cell_carcinoma", 
            "actinic_keratosis"
        ]
        
        # 메타데이터를 위한 리스트
        metadata = []
        
        # 각 클래스별로 이미지 생성
        for diagnosis in diagnoses:
            for i in range(num_samples // len(diagnoses)):
                # 이미지 ID 생성
                image_id = f"sample_{diagnosis}_{i}"
                image_path = os.path.join(self.isic_dir, "samples", f"{image_id}.jpg")
                
                # 단색 이미지 생성 (각 진단별로 다른 색상)
                if diagnosis == "melanoma":
                    color = (139, 69, 19)  # 갈색
                elif diagnosis == "nevus":
                    color = (210, 180, 140)  # 베이지색
                elif diagnosis == "seborrheic_keratosis":
                    color = (169, 169, 169)  # 회색
                elif diagnosis == "basal_cell_carcinoma":
                    color = (255, 182, 193)  # 분홍색
                else:  # actinic_keratosis
                    color = (240, 128, 128)  # 연한 빨간색
                
                # 약간의 노이즈 추가
                noise = np.random.randint(-20, 20, 3)
                color = tuple(max(0, min(255, c + n)) for c, n in zip(color, noise))
                
                # 이미지 생성 및 저장
                img = Image.new('RGB', (224, 224), color)
                img.save(image_path)
                
                # 메타데이터 추가
                metadata.append({
                    "_id": image_id,
                    "diagnosis": diagnosis,
                    "age": np.random.randint(20, 80),
                    "sex": np.random.choice(["male", "female"]),
                    "localization": np.random.choice(["back", "scalp", "face", "chest", "upper extremity", "lower extremity"]),
                    "is_sample": True
                })
        
        # 메타데이터 저장
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(self.isic_dir, "isic_metadata.csv"), index=False)
        
        print(f"샘플 데이터 생성 완료: {len(metadata)}개 이미지")
    
    def preprocess_ham10000(self):
        """HAM10000 데이터셋 전처리"""
        print("HAM10000 데이터셋 전처리 중...")
        
        # 메타데이터 로드
        metadata_path = os.path.join(self.ham10000_dir, "HAM10000_metadata.csv")
        if not os.path.exists(metadata_path):
            print("메타데이터 파일이 없습니다. 먼저 데이터를 다운로드하세요.")
            return None
        
        df = pd.read_csv(metadata_path)
        
        # 이미지 디렉토리 확인
        image_dir = os.path.join(self.ham10000_dir, "HAM10000_images")
        if not os.path.exists(image_dir):
            image_dir = os.path.join(self.ham10000_dir, "HAM10000_images_part_1")
        
        if not os.path.exists(image_dir):
            print("이미지 디렉토리를 찾을 수 없습니다. 먼저 데이터를 다운로드하세요.")
            return None
        
        # 클래스 매핑 (진단명 -> 숫자)
        diagnosis_mapping = {
            'akiec': 0,  # Actinic keratoses
            'bcc': 1,    # Basal cell carcinoma
            'bkl': 2,    # Benign keratosis-like lesions
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic nevi
            'vasc': 6    # Vascular lesions
        }
        
        # 클래스 이름 매핑
        diagnosis_names = {
            0: 'Actinic keratoses',
            1: 'Basal cell carcinoma',
            2: 'Benign keratosis-like lesions',
            3: 'Dermatofibroma',
            4: 'Melanoma',
            5: 'Melanocytic nevi',
            6: 'Vascular lesions'
        }
        
        # 데이터프레임에 숫자 라벨 추가
        df['label'] = df['dx'].map(diagnosis_mapping)
        
        # 클래스 분포 확인
        print("클래스 분포:")
        for dx, count in df['dx'].value_counts().items():
            label = diagnosis_mapping[dx]
            name = diagnosis_names[label]
            print(f"{dx} ({name}): {count}개 ({count/len(df)*100:.1f}%)")
        
        # 전처리된 데이터 저장
        processed_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # 메타데이터 저장
        df.to_csv(os.path.join(processed_dir, 'ham10000_processed.csv'), index=False)
        
        # 클래스 매핑 정보 저장
        with open(os.path.join(processed_dir, 'class_mapping.txt'), 'w') as f:
            for code, label in diagnosis_mapping.items():
                name = diagnosis_names[label]
                f.write(f"{code},{label},{name}\n")
        
        print("HAM10000 데이터셋 전처리 완료")
        return df
    
    def preprocess_isic(self):
        """ISIC 데이터셋 전처리"""
        print("ISIC 데이터셋 전처리 중...")
        
        # 메타데이터 로드
        metadata_path = os.path.join(self.isic_dir, "isic_metadata.csv")
        if not os.path.exists(metadata_path):
            print("메타데이터 파일이 없습니다. 먼저 데이터를 다운로드하세요.")
            return None
        
        df = pd.read_csv(metadata_path)
        
        # 샘플 데이터인 경우
        if 'is_sample' in df.columns and df['is_sample'].any():
            print("샘플 데이터 전처리 중...")
            
            # 이미 전처리된 형태로 생성되었으므로 간단히 처리
            # 클래스 매핑 (진단명 -> 숫자)
            diagnosis_mapping = {
                'melanoma': 0,
                'nevus': 1,
                'seborrheic_keratosis': 2,
                'basal_cell_carcinoma': 3,
                'actinic_keratosis': 4
            }
            
            # 클래스 이름 매핑
            diagnosis_names = {
                0: 'Melanoma',
                1: 'Nevus',
                2: 'Seborrheic keratosis',
                3: 'Basal cell carcinoma',
                4: 'Actinic keratosis'
            }
            
            # 데이터프레임에 숫자 라벨 추가
            df['label'] = df['diagnosis'].map(diagnosis_mapping)
            
            # 이미지 경로 추가
            df['image_path'] = df['_id'].apply(lambda x: os.path.join(self.isic_dir, "samples", f"{x}.jpg"))
            
        else:
            # 실제 ISIC 데이터 처리 (API 응답 구조에 따라 조정 필요)
            print("실제 ISIC 데이터 전처리 중...")
            
            # 여기서는 ISIC 데이터의 구조를 정확히 알 수 없으므로 가정하에 처리
            # 실제 구현 시 API 응답 구조에 맞게 조정 필요
            if 'meta' in df.columns and 'clinical' in df.columns:
                # 진단 정보 추출 (예시)
                df['diagnosis'] = df.apply(
                    lambda row: self._extract_diagnosis_from_meta(row), 
                    axis=1
                )
            
            # 클래스 매핑 (진단명 -> 숫자)
            diagnosis_mapping = {
                'melanoma': 0,
                'nevus': 1,
                'seborrheic_keratosis': 2,
                'basal_cell_carcinoma': 3,
                'actinic_keratosis': 4,
                'unknown': 5
            }
            
            # 클래스 이름 매핑
            diagnosis_names = {
                0: 'Melanoma',
                1: 'Nevus',
                2: 'Seborrheic keratosis',
                3: 'Basal cell carcinoma',
                4: 'Actinic keratosis',
                5: 'Unknown'
            }
            
            # 데이터프레임에 숫자 라벨 추가 (없는 경우 unknown으로 처리)
            if 'diagnosis' not in df.columns:
                df['diagnosis'] = 'unknown'
            
            df['label'] = df['diagnosis'].map(diagnosis_mapping)
            
            # 이미지 경로 추가
            df['image_path'] = df['_id'].apply(lambda x: os.path.join(self.isic_dir, f"{x}.jpg"))
        
        # 클래스 분포 확인
        print("클래스 분포:")
        for dx, count in df['diagnosis'].value_counts().items():
            label = diagnosis_mapping.get(dx, 5)  # 없는 경우 unknown(5)으로 처리
            name = diagnosis_names[label]
            print(f"{dx} ({name}): {count}개 ({count/len(df)*100:.1f}%)")
        
        # 전처리된 데이터 저장
        processed_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # 메타데이터 저장
        df.to_csv(os.path.join(processed_dir, 'isic_processed.csv'), index=False)
        
        # 클래스 매핑 정보 저장
        with open(os.path.join(processed_dir, 'isic_class_mapping.txt'), 'w') as f:
            for code, label in diagnosis_mapping.items():
                name = diagnosis_names[label]
                f.write(f"{code},{label},{name}\n")
        
        print("ISIC 데이터셋 전처리 완료")
        return df
    
    def _extract_diagnosis_from_meta(self, row):
        """
        ISIC 메타데이터에서 진단 정보 추출 (예시 함수)
        실제 구현 시 API 응답 구조에 맞게 조정 필요
        """
        try:
            # 예시: 메타데이터 구조에 따라 조정 필요
            if isinstance(row['meta'], dict) and 'clinical' in row['meta']:
                diagnosis = row['meta']['clinical'].get('diagnosis', '')
                if 'melanoma' in diagnosis.lower():
                    return 'melanoma'
                elif 'nevus' in diagnosis.lower() or 'nevi' in diagnosis.lower():
                    return 'nevus'
                elif 'keratosis' in diagnosis.lower() and 'seborrheic' in diagnosis.lower():
                    return 'seborrheic_keratosis'
                elif 'basal cell' in diagnosis.lower() or 'bcc' in diagnosis.lower():
                    return 'basal_cell_carcinoma'
                elif 'actinic' in diagnosis.lower() or 'solar' in diagnosis.lower():
                    return 'actinic_keratosis'
            return 'unknown'
        except:
            return 'unknown'
    
    def prepare_dataset(self, img_size=(224, 224), batch_size=32, val_split=0.2, test_split=0.1):
        """
        학습, 검증, 테스트 데이터셋 준비
        
        Args:
            img_size (tuple): 이미지 크기 (width, height)
            batch_size (int): 배치 크기
            val_split (float): 검증 데이터 비율
            test_split (float): 테스트 데이터 비율
            
        Returns:
            tuple: (train_ds, val_ds, test_ds, class_names, num_classes)
        """
        print(f"데이터셋 준비 중 (이미지 크기: {img_size}, 배치 크기: {batch_size})...")
        
        processed_dir = os.path.join(self.data_dir, 'processed')
        
        # HAM10000 데이터 로드
        ham_csv = os.path.join(processed_dir, 'ham10000_processed.csv')
        if os.path.exists(ham_csv):
            ham_df = pd.read_csv(ham_csv)
            print(f"HAM10000 데이터 로드: {len(ham_df)}개 이미지")
        else:
            ham_df = pd.DataFrame()
            print("HAM10000 데이터를 찾을 수 없습니다.")
        
        # ISIC 데이터 로드
        isic_csv = os.path.join(processed_dir, 'isic_processed.csv')
        if os.path.exists(isic_csv):
            isic_df = pd.read_csv(isic_csv)
            print(f"ISIC 데이터 로드: {len(isic_df)}개 이미지")
        else:
            isic_df = pd.DataFrame()
            print("ISIC 데이터를 찾을 수 없습니다.")
        
        # 데이터가 없는 경우 샘플 데이터 생성
        if len(ham_df) == 0 and len(isic_df) == 0:
            print("데이터를 찾을 수 없습니다. 샘플 데이터를 생성합니다.")
            self._create_sample_data(num_samples=500)
            isic_df = self.preprocess_isic()
        
        # 데이터셋 준비
        if len(isic_df) > 0:
            # ISIC 데이터 사용
            df = isic_df
            image_column = 'image_path'
            label_column = 'label'
            
            # 클래스 정보 로드
            class_mapping_path = os.path.join(processed_dir, 'isic_class_mapping.txt')
            class_names = {}
            with open(class_mapping_path, 'r') as f:
                for line in f:
                    code, label, name = line.strip().split(',')
                    class_names[int(label)] = name
            
            num_classes = len(class_names)
            
        elif len(ham_df) > 0:
            # HAM10000 데이터 사용
            df = ham_df
            image_column = 'image_id'  # HAM10000의 이미지 ID 컬럼
            label_column = 'label'
            
            # 이미지 경로 추가
            image_dir = os.path.join(self.ham10000_dir, "HAM10000_images")
            if not os.path.exists(image_dir):
                image_dir = os.path.join(self.ham10000_dir, "HAM10000_images_part_1")
            
            df['image_path'] = df[image_column].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
            image_column = 'image_path'
            
            # 클래스 정보 로드
            class_mapping_path = os.path.join(processed_dir, 'class_mapping.txt')
            class_names = {}
            with open(class_mapping_path, 'r') as f:
                for line in f:
                    code, label, name = line.strip().split(',')
                    class_names[int(label)] = name
            
            num_classes = len(class_names)
        else:
            print("사용 가능한 데이터가 없습니다.")
            return None, None, None, None, 0
        
        # 데이터 분할 (학습, 검증, 테스트)
        train_df, temp_df = train_test_split(df, test_size=val_split+test_split, stratify=df[label_column], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_split/(val_split+test_split), stratify=temp_df[label_column], random_state=42)
        
        print(f"데이터 분할: 학습 {len(train_df)}개, 검증 {len(val_df)}개, 테스트 {len(test_df)}개")
        
        # 데이터셋 생성 함수
        def create_dataset(dataframe, is_training=False):
            # 이미지 경로와 라벨 추출
            images = dataframe[image_column].tolist()
            labels = dataframe[label_column].tolist()
            
            # 데이터셋 생성
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            
            # 이미지 로드 및 전처리 함수
            def process_path(image_path, label):
                # 이미지 로드
                img = tf.io.read_file(image_path)
                img = tf.image.decode_jpeg(img, channels=3)
                
                # 이미지 크기 조정
                img = tf.image.resize(img, img_size)
                
                # 픽셀값 정규화 [0, 1]
                img = tf.cast(img, tf.float32) / 255.0
                
                # 원-핫 인코딩
                label = tf.one_hot(label, num_classes)
                
                return img, label
            
            # 데이터셋 매핑
            dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            
            # 학습 데이터셋인 경우 셔플
            if is_training:
                dataset = dataset.shuffle(buffer_size=len(dataframe))
            
            # 배치 설정 및 프리페치
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        # 데이터셋 생성
        train_ds = create_dataset(train_df, is_training=True)
        val_ds = create_dataset(val_df)
        test_ds = create_dataset(test_df)
        
        # 데이터셋 정보 저장
        dataset_info = {
            'num_train': len(train_df),
            'num_val': len(val_df),
            'num_test': len(test_df),
            'num_classes': num_classes,
            'class_names': class_names,
            'img_size': img_size,
            'batch_size': batch_size
        }
        
        # JSON으로 저장
        import json
        with open(os.path.join(processed_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print("데이터셋 준비 완료")
        return train_ds, val_ds, test_ds, class_names, num_classes
    
    def _download_file(self, url, destination):
        """
        파일 다운로드 유틸리티 함수
        
        Args:
            url (str): 다운로드 URL
            destination (str): 저장 경로
        """
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
                    
        except Exception as e:
            print(f"다운로드 중 오류 발생: {e}")
            if os.path.exists(destination):
                os.remove(destination)
            raise e

# 사용 예시
if __name__ == "__main__":
    data_dir = "/home/ubuntu/dermscan/model_training/data"
    collector = DataCollector(data_dir)
    
    # 데이터 다운로드
    # collector.download_ham10000()
    collector.download_isic_subset(limit=100)  # 테스트를 위해 적은 수만 다운로드
    
    # 데이터 전처리
    # collector.preprocess_ham10000()
    collector.preprocess_isic()
    
    # 데이터셋 준비
    train_ds, val_ds, test_ds, class_names, num_classes = collector.prepare_dataset()
    
    print(f"클래스 정보: {class_names}")
    print(f"클래스 수: {num_classes}")
