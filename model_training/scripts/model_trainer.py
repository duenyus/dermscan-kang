import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# 데이터 수집기 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_collector import DataCollector

class ModelTrainer:
    def __init__(self, data_dir, models_dir, logs_dir):
        """
        모델 학습을 위한 클래스
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            models_dir (str): 모델 저장 디렉토리 경로
            logs_dir (str): 로그 저장 디렉토리 경로
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        
        # 필요한 디렉토리 생성
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # 데이터 수집기 초기화
        self.data_collector = DataCollector(data_dir)
        
        # 현재 시간을 이용한 실행 ID 생성
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 및 학습 관련 속성
        self.model = None
        self.history = None
        self.class_names = None
        self.num_classes = 0
    
    def prepare_data(self, img_size=(224, 224), batch_size=32):
        """
        데이터 준비
        
        Args:
            img_size (tuple): 이미지 크기 (width, height)
            batch_size (int): 배치 크기
            
        Returns:
            tuple: (train_ds, val_ds, test_ds)
        """
        print("데이터 준비 중...")
        
        # 데이터셋 정보 파일 확인
        dataset_info_path = os.path.join(self.data_dir, 'processed', 'dataset_info.json')
        
        if os.path.exists(dataset_info_path):
            # 기존 데이터셋 정보 로드
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
            
            self.class_names = dataset_info['class_names']
            self.num_classes = dataset_info['num_classes']
            
            print(f"기존 데이터셋 정보 로드: {self.num_classes}개 클래스, {dataset_info['num_train']}개 학습 데이터")
        
        # 데이터셋 준비
        train_ds, val_ds, test_ds, class_names, num_classes = self.data_collector.prepare_dataset(
            img_size=img_size, 
            batch_size=batch_size
        )
        
        # 클래스 정보 업데이트
        if class_names:
            self.class_names = class_names
        if num_classes > 0:
            self.num_classes = num_classes
        
        print(f"데이터 준비 완료: {self.num_classes}개 클래스")
        return train_ds, val_ds, test_ds
    
    def build_model(self, model_name='efficientnet', img_size=(224, 224)):
        """
        모델 구축
        
        Args:
            model_name (str): 사용할 모델 이름 ('efficientnet', 'resnet', 'mobilenet')
            img_size (tuple): 이미지 크기 (width, height)
            
        Returns:
            tf.keras.Model: 구축된 모델
        """
        print(f"{model_name} 모델 구축 중...")
        
        # 입력 형태 정의
        input_shape = img_size + (3,)  # (height, width, channels)
        
        # 기본 모델 선택
        if model_name == 'efficientnet':
            base_model = applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape
            )
        elif model_name == 'resnet':
            base_model = applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape
            )
        elif model_name == 'mobilenet':
            base_model = applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=input_shape
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        # 기본 모델 동결 (학습 초기에는 사전 학습된 가중치 유지)
        base_model.trainable = False
        
        # 모델 구축
        inputs = tf.keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        
        # 모델 컴파일
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 모델 요약 출력
        self.model.summary()
        
        print(f"{model_name} 모델 구축 완료")
        return self.model
    
    def train(self, train_ds, val_ds, epochs=20, fine_tune_at=10, unfreeze_layers=100):
        """
        모델 학습
        
        Args:
            train_ds (tf.data.Dataset): 학습 데이터셋
            val_ds (tf.data.Dataset): 검증 데이터셋
            epochs (int): 전체 에폭 수
            fine_tune_at (int): 미세 조정 시작 에폭
            unfreeze_layers (int): 미세 조정 시 해제할 레이어 수
            
        Returns:
            tf.keras.callbacks.History: 학습 히스토리
        """
        if self.model is None:
            raise ValueError("모델이 구축되지 않았습니다. build_model()을 먼저 호출하세요.")
        
        print(f"모델 학습 시작 (전체 {epochs}에폭, 미세 조정 {fine_tune_at}에폭부터)")
        
        # 모델 저장 경로
        model_path = os.path.join(self.models_dir, f"model_{self.run_id}")
        os.makedirs(model_path, exist_ok=True)
        
        # 체크포인트 콜백
        checkpoint_path = os.path.join(model_path, "best_model.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        # 조기 종료 콜백
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
        
        # 학습률 감소 콜백
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
        
        # 텐서보드 콜백
        tensorboard_path = os.path.join(self.logs_dir, self.run_id)
        tensorboard = TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=1,
            write_graph=True
        )
        
        # 콜백 리스트
        callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard]
        
        # 1단계: 전이 학습 (기본 모델 동결)
        print("1단계: 전이 학습 (기본 모델 동결)")
        history1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_at,
            callbacks=callbacks
        )
        
        # 2단계: 미세 조정 (일부 레이어 해제)
        if fine_tune_at < epochs and unfreeze_layers > 0:
            print(f"2단계: 미세 조정 (상위 {unfreeze_layers}개 레이어 해제)")
            
            # 기본 모델의 일부 레이어 해제
            base_model = self.model.layers[1]  # 기본 모델은 일반적으로 두 번째 레이어
            base_model.trainable = True
            
            # 상위 레이어만 학습 가능하도록 설정
            for layer in base_model.layers[:-unfreeze_layers]:
                layer.trainable = False
            
            # 더 낮은 학습률로 재컴파일
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 미세 조정 학습
            history2 = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                initial_epoch=fine_tune_at,
                callbacks=callbacks
            )
            
            # 히스토리 병합
            self.history = {
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
                'loss': history1.history['loss'] + history2.history['loss'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss']
            }
        else:
            self.history = history1.history
        
        # 최종 모델 저장
        final_model_path = os.path.join(model_path, "final_model.h5")
        self.model.save(final_model_path)
        
        # 모델 정보 저장
        model_info = {
            'run_id': self.run_id,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'img_size': img_size,
            'epochs': epochs,
            'fine_tune_at': fine_tune_at,
            'unfreeze_layers': unfreeze_layers,
            'final_accuracy': float(self.history['val_accuracy'][-1]),
            'best_accuracy': float(max(self.history['val_accuracy'])),
            'final_loss': float(self.history['val_loss'][-1]),
            'best_loss': float(min(self.history['val_loss']))
        }
        
        with open(os.path.join(model_path, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"모델 학습 완료: 최종 정확도 {model_info['final_accuracy']:.4f}, 최고 정확도 {model_info['best_accuracy']:.4f}")
        return self.history
    
    def plot_training_history(self, save_path=None):
        """
        학습 히스토리 시각화
        
        Args:
            save_path (str, optional): 그래프 저장 경로
        """
        if self.history is None:
            raise ValueError("학습 히스토리가 없습니다. train()을 먼저 호출하세요.")
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 정확도 그래프
        ax1.plot(self.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True)
        
        # 손실 그래프
        ax2.plot(self.history['loss'], label='Train Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"학습 히스토리 그래프 저장: {save_path}")
        
        plt.show()
    
    def load_model(self, model_path):
        """
        저장된 모델 로드
        
        Args:
            model_path (str): 모델 파일 경로
            
        Returns:
            tf.keras.Model: 로드된 모델
        """
        print(f"모델 로드 중: {model_path}")
        
        # 모델 로드
        self.model = models.load_model(model_path)
        
        # 모델 정보 로드
        model_dir = os.path.dirname(model_path)
        model_info_path = os.path.join(model_dir, "model_info.json")
        
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            self.class_names = model_info['class_names']
            self.num_classes = model_info['num_classes']
            
            print(f"모델 정보 로드: {self.num_classes}개 클래스, 정확도 {model_info['best_accuracy']:.4f}")
        else:
            print("모델 정보 파일을 찾을 수 없습니다.")
        
        return self.model

# 학습 파라미터
img_size = (224, 224)
batch_size = 32
epochs = 20
fine_tune_at = 10
unfreeze_layers = 100

# 메인 실행 코드
if __name__ == "__main__":
    # 경로 설정
    base_dir = "/home/ubuntu/dermscan/model_training"
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    
    # 모델 트레이너 초기화
    trainer = ModelTrainer(data_dir, models_dir, logs_dir)
    
    # 데이터 준비
    train_ds, val_ds, test_ds = trainer.prepare_data(img_size=img_size, batch_size=batch_size)
    
    # 모델 구축
    model = trainer.build_model(model_name='efficientnet', img_size=img_size)
    
    # 모델 학습
    history = trainer.train(train_ds, val_ds, epochs=epochs, fine_tune_at=fine_tune_at, unfreeze_layers=unfreeze_layers)
    
    # 학습 히스토리 시각화
    plot_path = os.path.join(models_dir, f"training_history_{trainer.run_id}.png")
    trainer.plot_training_history(save_path=plot_path)
