import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
import pandas as pd
from datetime import datetime
import sys

# 데이터 수집기와 모델 트레이너 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_collector import DataCollector
from model_trainer import ModelTrainer

class ModelEvaluator:
    def __init__(self, data_dir, models_dir, evaluation_dir):
        """
        모델 평가를 위한 클래스
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            models_dir (str): 모델 저장 디렉토리 경로
            evaluation_dir (str): 평가 결과 저장 디렉토리 경로
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.evaluation_dir = evaluation_dir
        
        # 필요한 디렉토리 생성
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # 데이터 수집기 초기화
        self.data_collector = DataCollector(data_dir)
        
        # 현재 시간을 이용한 평가 ID 생성
        self.eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 및 평가 관련 속성
        self.model = None
        self.class_names = None
        self.num_classes = 0
        self.test_ds = None
        self.y_true = []
        self.y_pred = []
        self.y_prob = []
    
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
            
            # 클래스 정보 로드
            if isinstance(model_info['class_names'], dict):
                self.class_names = model_info['class_names']
            else:
                # 리스트인 경우 딕셔너리로 변환
                self.class_names = {i: name for i, name in enumerate(model_info['class_names'])}
            
            self.num_classes = model_info['num_classes']
            
            print(f"모델 정보 로드: {self.num_classes}개 클래스, 정확도 {model_info.get('best_accuracy', 'N/A')}")
        else:
            print("모델 정보 파일을 찾을 수 없습니다.")
        
        return self.model
    
    def prepare_test_data(self, img_size=(224, 224), batch_size=32):
        """
        테스트 데이터 준비
        
        Args:
            img_size (tuple): 이미지 크기 (width, height)
            batch_size (int): 배치 크기
            
        Returns:
            tf.data.Dataset: 테스트 데이터셋
        """
        print("테스트 데이터 준비 중...")
        
        # 데이터셋 준비
        _, _, test_ds, class_names, num_classes = self.data_collector.prepare_dataset(
            img_size=img_size, 
            batch_size=batch_size
        )
        
        # 클래스 정보 업데이트
        if class_names and not self.class_names:
            self.class_names = class_names
        if num_classes > 0 and self.num_classes == 0:
            self.num_classes = num_classes
        
        self.test_ds = test_ds
        print(f"테스트 데이터 준비 완료")
        return test_ds
    
    def evaluate(self, test_ds=None):
        """
        모델 평가
        
        Args:
            test_ds (tf.data.Dataset, optional): 테스트 데이터셋
            
        Returns:
            dict: 평가 결과
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        if test_ds is None:
            if self.test_ds is None:
                raise ValueError("테스트 데이터셋이 준비되지 않았습니다. prepare_test_data()를 먼저 호출하세요.")
            test_ds = self.test_ds
        
        print("모델 평가 중...")
        
        # 모델 평가
        evaluation = self.model.evaluate(test_ds)
        
        # 평가 결과 저장
        eval_results = {
            'loss': float(evaluation[0]),
            'accuracy': float(evaluation[1])
        }
        
        print(f"평가 결과: 손실 {eval_results['loss']:.4f}, 정확도 {eval_results['accuracy']:.4f}")
        
        # 예측 및 실제 라벨 수집
        self.y_true = []
        self.y_pred = []
        self.y_prob = []
        
        for images, labels in test_ds:
            # 예측
            predictions = self.model.predict(images)
            
            # 원-핫 인코딩된 라벨을 클래스 인덱스로 변환
            true_labels = np.argmax(labels.numpy(), axis=1)
            pred_labels = np.argmax(predictions, axis=1)
            
            # 결과 저장
            self.y_true.extend(true_labels)
            self.y_pred.extend(pred_labels)
            self.y_prob.extend(predictions)
        
        # 리스트를 numpy 배열로 변환
        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(self.y_pred)
        self.y_prob = np.array(self.y_prob)
        
        # 혼동 행렬 계산
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # 분류 보고서 생성
        cr = classification_report(self.y_true, self.y_pred, output_dict=True)
        
        # 평가 결과 저장 디렉토리
        eval_dir = os.path.join(self.evaluation_dir, f"eval_{self.eval_id}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 평가 결과 저장
        eval_results.update({
            'eval_id': self.eval_id,
            'confusion_matrix': cm.tolist(),
            'classification_report': cr
        })
        
        with open(os.path.join(eval_dir, "evaluation_results.json"), 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        print(f"평가 결과 저장: {os.path.join(eval_dir, 'evaluation_results.json')}")
        return eval_results
    
    def plot_confusion_matrix(self, save_path=None):
        """
        혼동 행렬 시각화
        
        Args:
            save_path (str, optional): 그래프 저장 경로
        """
        if len(self.y_true) == 0 or len(self.y_pred) == 0:
            raise ValueError("평가 데이터가 없습니다. evaluate()를 먼저 호출하세요.")
        
        # 혼동 행렬 계산
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # 클래스 이름 준비
        if self.class_names:
            class_names = [self.class_names.get(str(i), f"Class {i}") for i in range(self.num_classes)]
        else:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        # 그래프 생성
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"혼동 행렬 그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """
        ROC 곡선 시각화
        
        Args:
            save_path (str, optional): 그래프 저장 경로
        """
        if len(self.y_true) == 0 or len(self.y_prob) == 0:
            raise ValueError("평가 데이터가 없습니다. evaluate()를 먼저 호출하세요.")
        
        # 클래스 이름 준비
        if self.class_names:
            class_names = [self.class_names.get(str(i), f"Class {i}") for i in range(self.num_classes)]
        else:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        # 그래프 생성
        plt.figure(figsize=(10, 8))
        
        # 각 클래스별 ROC 곡선
        for i in range(self.num_classes):
            # 이진 분류 문제로 변환 (one-vs-rest)
            y_true_binary = (self.y_true == i).astype(int)
            y_score = self.y_prob[:, i]
            
            # ROC 곡선 계산
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            # ROC 곡선 그리기
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        # 그래프 설정
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"ROC 곡선 그래프 저장: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """
        정밀도-재현율 곡선 시각화
        
        Args:
            save_path (str, optional): 그래프 저장 경로
        """
        if len(self.y_true) == 0 or len(self.y_prob) == 0:
            raise ValueError("평가 데이터가 없습니다. evaluate()를 먼저 호출하세요.")
        
        # 클래스 이름 준비
        if self.class_names:
            class_names = [self.class_names.get(str(i), f"Class {i}") for i in range(self.num_classes)]
        else:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        # 그래프 생성
        plt.figure(figsize=(10, 8))
        
        # 각 클래스별 정밀도-재현율 곡선
        for i in range(self.num_classes):
            # 이진 분류 문제로 변환 (one-vs-rest)
            y_true_binary = (self.y_true == i).astype(int)
            y_score = self.y_prob[:, i]
            
            # 정밀도-재현율 곡선 계산
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            pr_auc = auc(recall, precision)
            
            # 정밀도-재현율 곡선 그리기
            plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AUC = {pr_auc:.2f})')
        
        # 그래프 설정
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"정밀도-재현율 곡선 그래프 저장: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self):
        """
        종합 평가 보고서 생성
        
        Returns:
            str: 보고서 저장 경로
        """
        if len(self.y_true) == 0 or len(self.y_pred) == 0:
            raise ValueError("평가 데이터가 없습니다. evaluate()를 먼저 호출하세요.")
        
        # 평가 결과 저장 디렉토리
        eval_dir = os.path.join(self.evaluation_dir, f"eval_{self.eval_id}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 혼동 행렬 그래프 저장
        cm_path = os.path.join(eval_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(save_path=cm_path)
        
        # ROC 곡선 그래프 저장
        roc_path = os.path.join(eval_dir, "roc_curve.png")
        self.plot_roc_curve(save_path=roc_path)
        
        # 정밀도-재현율 곡선 그래프 저장
        pr_path = os.path.join(eval_dir, "precision_recall_curve.png")
        self.plot_precision_recall_curve(save_path=pr_path)
        
        # 분류 보고서 생성
        cr = classification_report(self.y_true, self.y_pred, output_dict=True)
        
        # 클래스별 성능 지표
        class_metrics = []
        for i in range(self.num_classes):
            class_name = self.class_names.get(str(i), f"Class {i}") if self.class_names else f"Class {i}"
            metrics = cr.get(str(i), {})
            class_metrics.append({
                'class': class_name,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1-score': metrics.get('f1-score', 0),
                'support': metrics.get('support', 0)
            })
        
        # 전체 성능 지표
        overall_metrics = {
            'accuracy': cr.get('accuracy', 0),
            'macro_avg': cr.get('macro avg', {}),
            'weighted_avg': cr.get('weighted avg', {})
        }
        
        # HTML 보고서 생성
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-value {{ font-weight: bold; }}
                .container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .chart {{ width: 48%; margin-bottom: 20px; }}
                .chart img {{ width: 100%; border: 1px solid #ddd; }}
                @media (max-width: 768px) {{ .chart {{ width: 100%; }} }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <p><strong>Evaluation ID:</strong> {self.eval_id}</p>
            <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Overall Performance</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td class="metric-value">{overall_metrics['accuracy']:.4f}</td>
                </tr>
                <tr>
                    <td>Macro Avg Precision</td>
                    <td class="metric-value">{overall_metrics['macro_avg'].get('precision', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Macro Avg Recall</td>
                    <td class="metric-value">{overall_metrics['macro_avg'].get('recall', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Macro Avg F1-Score</td>
                    <td class="metric-value">{overall_metrics['macro_avg'].get('f1-score', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Weighted Avg Precision</td>
                    <td class="metric-value">{overall_metrics['weighted_avg'].get('precision', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Weighted Avg Recall</td>
                    <td class="metric-value">{overall_metrics['weighted_avg'].get('recall', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Weighted Avg F1-Score</td>
                    <td class="metric-value">{overall_metrics['weighted_avg'].get('f1-score', 0):.4f}</td>
                </tr>
            </table>
            
            <h2>Class-wise Performance</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
        """
        
        # 클래스별 성능 지표 추가
        for metrics in class_metrics:
            html_report += f"""
                <tr>
                    <td>{metrics['class']}</td>
                    <td class="metric-value">{metrics['precision']:.4f}</td>
                    <td class="metric-value">{metrics['recall']:.4f}</td>
                    <td class="metric-value">{metrics['f1-score']:.4f}</td>
                    <td>{metrics['support']}</td>
                </tr>
            """
        
        # 그래프 추가
        html_report += f"""
            </table>
            
            <h2>Evaluation Charts</h2>
            <div class="container">
                <div class="chart">
                    <h3>Confusion Matrix</h3>
                    <img src="confusion_matrix.png" alt="Confusion Matrix">
                </div>
                <div class="chart">
                    <h3>ROC Curve</h3>
                    <img src="roc_curve.png" alt="ROC Curve">
                </div>
                <div class="chart">
                    <h3>Precision-Recall Curve</h3>
                    <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
                </div>
            </div>
        </body>
        </html>
        """
        
        # HTML 보고서 저장
        report_path = os.path.join(eval_dir, "evaluation_report.html")
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        print(f"평가 보고서 생성 완료: {report_path}")
        return report_path

# 메인 실행 코드
if __name__ == "__main__":
    # 경로 설정
    base_dir = "/home/ubuntu/dermscan/model_training"
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    evaluation_dir = os.path.join(base_dir, "evaluation")
    
    # 모델 평가기 초기화
    evaluator = ModelEvaluator(data_dir, models_dir, evaluation_dir)
    
    # 최신 모델 찾기
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("model_")]
    if model_dirs:
        latest_model_dir = max(model_dirs)  # 가장 최근 모델 디렉토리
        model_path = os.path.join(models_dir, latest_model_dir, "best_model.h5")
        
        if os.path.exists(model_path):
            # 모델 로드
            evaluator.load_model(model_path)
            
            # 테스트 데이터 준비
            evaluator.prepare_test_data()
            
            # 모델 평가
            evaluator.evaluate()
            
            # 평가 보고서 생성
            evaluator.generate_evaluation_report()
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
    else:
        print(f"모델 디렉토리를 찾을 수 없습니다: {models_dir}")
