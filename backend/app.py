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

# 데이터베이스 초기화 함수
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 피드백 테이블 초기화
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
    feedback_table_exists = cursor.fetchone() is not None
    
    if not feedback_table_exists:
        # 피드백 테이블 생성 (없는 경우)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            model TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            score INTEGER NOT NULL,
            diagnoses TEXT,  /* JSON 형식으로 저장 */
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        logger.info("피드백 테이블이 생성되었습니다.")
    
    # 분석 로그 테이블 초기화
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_log'")
    log_table_exists = cursor.fetchone() is not None
    
    if not log_table_exists:
        # 분석 로그 테이블 생성 (없는 경우)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            model TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            diagnoses TEXT,  /* JSON 형식으로 저장 */
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        logger.info("분석 로그 테이블이 생성되었습니다.")
    
    conn.commit()
    conn.close()
    logger.info("SQLite 데이터베이스가 초기화되었습니다.")

# 데이터베이스 초기화 실행
init_db()

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 허용할 도메인 목록
ALLOWED_DOMAINS = ['rx.iptime.org', 'localhost', '127.0.0.1']
# 추가 허용 도메인이 있다면 여기에 넣으세요

# IP 주소 패턴 정규식
IP_PATTERN = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')


# 접근 제한 미들웨어
@app.before_request
def check_host():
    # Host 헤더에서 호스트 정보 추출
    host = request.host.split(':')[0]  # 포트 번호 제외

    # 로컬호스트와 127.0.0.1은 항상 허용
    if host in ['localhost', '127.0.0.1']:
        return None

    # 허용된 도메인 검사
    if host in ALLOWED_DOMAINS:
        return None

    # IP 패턴 확인 (IP로 직접 접근하는 경우)
    if IP_PATTERN.match(host):
        logger.warning(f"IP 주소 접근 감지: {host}")
        # Google로 리다이렉트
        return redirect("https://www.google.com")

    # 기타 도메인은 허용 목록에 있는지 확인
    if host not in ALLOWED_DOMAINS:
        logger.warning(f"알 수 없는 도메인 접근 감지: {host}")
        # 같은 방식으로 처리
        return redirect("https://www.google.com")


# 허용된 파일 확장자 확인
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 모델 로드
try:
    logger.info("모델 초기화 시작...")
    # 개별 모델 로드 시도
    try:
        vi_model = ViModel()
        logger.info(f"ViModel 로드 성공: {vi_model.__class__.__name__}")
    except Exception as e:
        logger.error(f"ViModel 로드 실패: {str(e)}")
        vi_model = None
    
    try:
        swin_model = SwinModel()
        logger.info(f"SwinModel 로드 성공: {swin_model.__class__.__name__}")
    except Exception as e:
        logger.error(f"SwinModel 로드 실패: {str(e)}")
        swin_model = None
    
    # 사용 가능한 모델만 등록
    models = {}
    if vi_model is not None:
        models['vi'] = vi_model
        logger.info("vi 모델이 딕셔너리에 추가되었습니다.")
    
    if swin_model is not None:
        models['swin'] = swin_model
        logger.info("swin 모델이 딕셔너리에 추가되었습니다.")
    
    # 사용 가능한 모델 확인
    logger.info(f"사용 가능한 모델 키: {list(models.keys())}")
    
    # 기본 모델 설정
    if 'vi' in models:
        default_model = 'vi'
    elif 'swin' in models:
        default_model = 'swin'
    else:
        raise ValueError("사용 가능한 모델이 없습니다.")
    
    logger.info(f"기본 모델 설정: {default_model}")
    
except Exception as e:
    logger.error(f"모델 초기화 오류: {str(e)}")
    # 최소한 ViModel은 초기화 시도
    vi_model = ViModel()
    models = {'vi': vi_model}
    default_model = 'vi'
    logger.info("기본 ViModel만 로드되었습니다.")


# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 관리자 페이지
@app.route('/admin')
def admin():
    return render_template('admin.html')


# 404 에러 핸들러 - 커스텀 404 페이지
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# 이미지 업로드 및 분석 API
@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 모델 선택 처리 (없으면 기본 모델 사용)
    model_type = request.form.get('model', default_model)
    logger.info(f"요청된 모델 유형: {model_type}")
    
    if model_type not in models:
        logger.warning(f"알 수 없는 모델 유형: {model_type}, 기본 모델로 대체")
        model_type = default_model
    
    # 선택된 모델 가져오기
    model = models[model_type]
    logger.info(f"선택된 모델: {model_type}, 클래스: {model.__class__.__name__}")

    if file and allowed_file(file.filename):
        # 안전한 파일명으로 변경
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            logger.info(f"이미지 분석 시작: {unique_filename}, 모델: {model_type}")
            # 이미지 전처리
            processed_image = model.preprocess_image(file_path)

            # 병변 설명 생성
            description = model.generate_description(processed_image)

            # 진단명 예측
            diagnoses = model.predict(processed_image)

            # 결과 반환
            result = {
                'image_url': url_for('serve_image', filename=unique_filename, _external=True),
                'description': description,
                'diagnoses': diagnoses,
                'model_used': model_type
            }

            # 분석 로그 저장
            try:
                # IP 주소 추출
                ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
                image_path = url_for('serve_image', filename=unique_filename, _external=True)
                diagnoses_json = json.dumps(diagnoses)
                
                # 데이터베이스에 저장
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                
                cursor.execute(
                    'INSERT INTO analysis_log (image_path, model, ip_address, diagnoses) VALUES (?, ?, ?, ?)',
                    (image_path, model_type, ip_address, diagnoses_json)
                )
                
                conn.commit()
                conn.close()
                logger.info(f"분석 로그 저장 성공: {image_path}, 모델: {model_type}, IP: {ip_address}")
            except Exception as e:
                logger.error(f"분석 로그 저장 오류: {str(e)}")
                # 로그 저장 실패해도 결과는 반환
            
            logger.info(f"이미지 분석 완료: {len(diagnoses)}개 진단 결과, 모델: {model_type}")
            return jsonify(result)

        except Exception as e:
            logger.error(f"이미지 분석 오류: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400


# 좋아요/싫어요 API
@app.route('/api/feedback', methods=['POST'])
def save_feedback():
    try:
        logger.info(f"피드백 API 호출 - 요청 데이터: {request.json}")
        
        data = request.json
        
        if not data:
            logger.error("요청 바디가 없거나 JSON 형식이 아닙니다.")
            return jsonify({'error': '요청 바디가 없거나 JSON 형식이 아닙니다.'}), 400
        
        if 'image_path' not in data or 'score' not in data:
            logger.error(f"필수 데이터 누락: {data}")
            return jsonify({'error': '필수 데이터가 누락되었습니다.'}), 400
        
        # 데이터 추출
        image_path = data['image_path']
        score = data['score']  # 좋아요: 1, 싫어요: -1
        diagnoses = data.get('diagnoses', [])  # 진단 정보 (배열)
        model = data.get('model', '')  # 모델 정보

        # 진단 정보를 JSON 문자열로 변환
        diagnoses_json = json.dumps(diagnoses)
        
        # IP 주소 추출 (X-Forwarded-For 헤더가 있으면 사용, 없으면 원격 주소 사용)
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        
        logger.info(f"피드백 데이터: {image_path}, 점수: {score}, IP: {ip_address}, 진단 수: {len(diagnoses)}")
        
        # 데이터베이스에 저장
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO feedback (image_path, ip_address, score, diagnoses, model) VALUES (?, ?, ?, ?, ?)',
                (image_path, ip_address, score, diagnoses_json, model)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"피드백 저장 성공: {image_path}, 점수: {score}, IP: {ip_address}")
            return jsonify({'success': True, 'message': '피드백이 저장되었습니다.'})
        
        except Exception as e:
            logger.error(f"데이터베이스 저장 오류: {str(e)}")
            return jsonify({'error': f'데이터베이스 저장 오류: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"피드백 저장 처리 중 예외 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 관리자용 피드백 목록 API
@app.route('/api/admin/feedback', methods=['GET'])
def get_feedback():
    try:
        # 필터 파라미터 가져오기
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        score_filter = request.args.get('score', 'all')
        ip_filter = request.args.get('ip', '')
        model = request.args.get('model', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # 페이지네이션 계산
        offset = (page - 1) * per_page
        
        # 쿼리 구성
        query = 'SELECT * FROM feedback WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND date(created_at) >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date(created_at) <= ?'
            params.append(end_date)
        
        if score_filter == 'positive':
            query += ' AND score > 0'
        elif score_filter == 'negative':
            query += ' AND score < 0'
        
        if ip_filter:
            query += ' AND ip_address LIKE ?'
            params.append(f'%{ip_filter}%')

        if model:
            query += ' AND model = ?'
            params.append(model)
        
        # 카운트 쿼리
        count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
        
        # 정렬 및 페이지네이션
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([per_page, offset])
        
        # 데이터베이스 연결 및 쿼리 실행
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        cursor = conn.cursor()
        
        # 전체 결과 수 가져오기
        cursor.execute(count_query, params[:-2] if len(params) >= 2 else params)
        total_count = cursor.fetchone()[0]
        
        # 피드백 데이터 가져오기
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 결과 변환
        feedback_data = []
        for row in rows:
            row_dict = dict(row)
            
            # diagnoses 필드가 없을 수 있으므로 안전하게 처리
            diagnoses = []
            if 'diagnoses' in row_dict and row_dict['diagnoses']:
                try:
                    diagnoses = json.loads(row_dict['diagnoses'])
                except json.JSONDecodeError:
                    logger.error(f"JSON 파싱 오류: {row_dict.get('diagnoses', 'None')}")
            
            feedback_item = {
                'id': row_dict['id'],
                'image_path': row_dict['image_path'],
                'ip_address': row_dict['ip_address'],
                'score': row_dict['score'],
                'model': row_dict['model'],
                'diagnoses': diagnoses,
                'created_at': row_dict['created_at']
            }
            
            feedback_data.append(feedback_item)
        
        # 총 페이지 수 계산
        total_pages = (total_count + per_page - 1) // per_page
        
        conn.close()
        
        return jsonify({
            'feedback': feedback_data,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_count': total_count,
                'total_pages': total_pages
            }
        })
    
    except Exception as e:
        logger.error(f"피드백 데이터 조회 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 관리자용 피드백 통계 API
@app.route('/api/admin/statistics', methods=['GET'])
def get_statistics():
    try:
        logger.info("통계 API 호출")
        
        # 필터 파라미터 가져오기
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        ip_filter = request.args.get('ip', '')
        
        logger.info(f"통계 필터: start_date={start_date}, end_date={end_date}, ip={ip_filter}")
        
        # 쿼리 구성
        base_query = 'FROM feedback WHERE 1=1'
        params = []
        
        if start_date:
            base_query += ' AND date(created_at) >= ?'
            params.append(start_date)
        
        if end_date:
            base_query += ' AND date(created_at) <= ?'
            params.append(end_date)
        
        if ip_filter:
            base_query += ' AND ip_address LIKE ?'
            params.append(f'%{ip_filter}%')
        
        # 데이터베이스 연결
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # 총 피드백 수 가져오기
        cursor.execute(f"SELECT COUNT(*) {base_query}", params)
        total_feedback = cursor.fetchone()[0]
        
        # 좋아요 수 가져오기
        cursor.execute(f"SELECT COUNT(*) {base_query} AND score > 0", params)
        total_likes = cursor.fetchone()[0]
        
        # 싫어요 수 가져오기
        cursor.execute(f"SELECT COUNT(*) {base_query} AND score < 0", params)
        total_dislikes = cursor.fetchone()[0]
        
        conn.close()
        
        result = {
            'total_feedback': total_feedback,
            'total_likes': total_likes,
            'total_dislikes': total_dislikes
        }
        
        logger.info(f"통계 결과: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"피드백 통계 조회 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 관리자용 분석 로그 API
@app.route('/api/admin/analysis-log', methods=['GET'])
def get_analysis_log():
    try:
        # 필터 파라미터 가져오기
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        ip_filter = request.args.get('ip', '')
        model = request.args.get('model', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # 페이지네이션 계산
        offset = (page - 1) * per_page
        
        # 쿼리 구성
        query = 'SELECT * FROM analysis_log WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND date(created_at) >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date(created_at) <= ?'
            params.append(end_date)
        
        if ip_filter:
            query += ' AND ip_address LIKE ?'
            params.append(f'%{ip_filter}%')

        if model:
            query += ' AND model = ?'
            params.append(model)
        
        # 카운트 쿼리
        count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
        
        # 정렬 및 페이지네이션
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([per_page, offset])
        
        # 데이터베이스 연결 및 쿼리 실행
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        cursor = conn.cursor()
        
        # 전체 결과 수 가져오기
        cursor.execute(count_query, params[:-2] if len(params) >= 2 else params)
        total_count = cursor.fetchone()[0]
        
        # 로그 데이터 가져오기
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 결과 변환
        log_data = []
        for row in rows:
            row_dict = dict(row)
            
            # diagnoses 필드가 없을 수 있으므로 안전하게 처리
            diagnoses = []
            if 'diagnoses' in row_dict and row_dict['diagnoses']:
                try:
                    diagnoses = json.loads(row_dict['diagnoses'])
                except json.JSONDecodeError:
                    logger.error(f"JSON 파싱 오류: {row_dict.get('diagnoses', 'None')}")
            
            log_item = {
                'id': row_dict['id'],
                'image_path': row_dict['image_path'],
                'ip_address': row_dict['ip_address'],
                'model': row_dict['model'],
                'diagnoses': diagnoses,
                'created_at': row_dict['created_at']
            }
            
            log_data.append(log_item)
        
        # 총 페이지 수 계산
        total_pages = (total_count + per_page - 1) // per_page
        
        conn.close()
        
        return jsonify({
            'logs': log_data,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_count': total_count,
                'total_pages': total_pages
            }
        })
    
    except Exception as e:
        logger.error(f"분석 로그 데이터 조회 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 관리자용 분석 로그 통계 API
@app.route('/api/admin/analysis-log/statistics', methods=['GET'])
def get_analysis_log_statistics():
    try:
        logger.info("분석 로그 통계 API 호출")
        
        # 필터 파라미터 가져오기
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        ip_filter = request.args.get('ip', '')
        model = request.args.get('model', '')
        
        logger.info(f"통계 필터: start_date={start_date}, end_date={end_date}, ip={ip_filter}, model={model}")
        
        # 쿼리 구성
        base_query = 'FROM analysis_log WHERE 1=1'
        params = []
        
        if start_date:
            base_query += ' AND date(created_at) >= ?'
            params.append(start_date)
        
        if end_date:
            base_query += ' AND date(created_at) <= ?'
            params.append(end_date)
        
        if ip_filter:
            base_query += ' AND ip_address LIKE ?'
            params.append(f'%{ip_filter}%')
            
        if model:
            base_query += ' AND model = ?'
            params.append(model)
        
        # 데이터베이스 연결
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # 총 로그 수 가져오기
        cursor.execute(f"SELECT COUNT(*) {base_query}", params)
        total_logs = cursor.fetchone()[0]
        
        # 모델별 사용 집계
        cursor.execute(f"SELECT model, COUNT(*) as count {base_query} GROUP BY model", params)
        model_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        result = {
            'total_logs': total_logs,
            'model_counts': model_counts
        }
        
        logger.info(f"분석 로그 통계 결과: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"분석 로그 통계 조회 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 관리자용 분석 로그 삭제 API
@app.route('/api/admin/analysis-log/delete', methods=['POST'])
def delete_analysis_log():
    try:
        data = request.json
        
        if not data or 'ids' not in data or not data['ids']:
            return jsonify({'error': '삭제할 ID가 누락되었습니다.'}), 400
        
        ids = data['ids']
        logger.info(f"분석 로그 삭제 요청 받음: {ids}")
        
        # 문자열로 들어온 경우 리스트로 변환
        if isinstance(ids, str):
            ids = [ids]
        
        # ID 목록이 비어있는지 확인
        if len(ids) == 0:
            return jsonify({'error': '삭제할 ID가 누락되었습니다.'}), 400
        
        # 데이터베이스에서 삭제
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # 쿼리에 사용할 플레이스홀더 생성 (?, ?, ?, ...)
            placeholders = ', '.join(['?' for _ in ids])
            
            # 삭제 쿼리 실행
            cursor.execute(f'DELETE FROM analysis_log WHERE id IN ({placeholders})', ids)
            
            # 영향 받은 행 수 확인
            affected_rows = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"삭제 완료: {affected_rows}개 분석 로그 항목 삭제됨")
            
            return jsonify({
                'success': True,
                'message': f'{affected_rows}개 항목이 삭제되었습니다.',
                'affected_rows': affected_rows
            })
            
        except Exception as e:
            logger.error(f"데이터베이스 삭제 오류: {str(e)}")
            return jsonify({'error': f'데이터베이스 삭제 오류: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"분석 로그 삭제 처리 중 예외 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 관리자용 피드백 삭제 API
@app.route('/api/admin/feedback/delete', methods=['POST'])
def delete_feedback():
    try:
        data = request.json
        
        if not data or 'ids' not in data or not data['ids']:
            return jsonify({'error': '삭제할 ID가 누락되었습니다.'}), 400
        
        ids = data['ids']
        logger.info(f"삭제 요청 받음: {ids}")
        
        # 문자열로 들어온 경우 리스트로 변환
        if isinstance(ids, str):
            ids = [ids]
        
        # ID 목록이 비어있는지 확인
        if len(ids) == 0:
            return jsonify({'error': '삭제할 ID가 누락되었습니다.'}), 400
        
        # 데이터베이스에서 삭제
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # 쿼리에 사용할 플레이스홀더 생성 (?, ?, ?, ...)
            placeholders = ', '.join(['?' for _ in ids])
            
            # 삭제 쿼리 실행
            cursor.execute(f'DELETE FROM feedback WHERE id IN ({placeholders})', ids)
            
            # 영향 받은 행 수 확인
            affected_rows = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"삭제 완료: {affected_rows}개 항목 삭제됨")
            
            return jsonify({
                'success': True,
                'message': f'{affected_rows}개 항목이 삭제되었습니다.',
                'affected_rows': affected_rows
            })
            
        except Exception as e:
            logger.error(f"데이터베이스 삭제 오류: {str(e)}")
            return jsonify({'error': f'데이터베이스 삭제 오류: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"피드백 삭제 처리 중 예외 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 이미지 파일 제공 라우트
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    logger.info("DermScan 백엔드 서버를 시작합니다. (http://localhost:5000)")
    app.run(host='0.0.0.0', port=5000, debug=True)