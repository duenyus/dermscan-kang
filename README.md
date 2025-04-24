# DermScan (No Crop Version)

피부 병변 이미지를 분석하여 병변을 설명하고 가능성 있는 진단명을 제시하는 웹 애플리케이션입니다.
이 버전은 이미지 크롭 기능이 제거된 버전으로, 업로드된 이미지를 그대로 분석합니다.

## 주요 기능

- 이미지 업로드 및 분석
- 두 가지 AI 모델 지원 (ViT, Swin Transformer)
- 병변 설명 생성
- 가능성 있는 진단명 제시
- 분석 결과에 대한 피드백 시스템
- 관리자 페이지 (통계 및 로그 확인)

## 시스템 요구사항

- Python 3.9 이상
- 웹 브라우저 (Chrome, Firefox, Safari 등)
- 최소 4GB RAM
- 인터넷 연결

## 설치 방법

1. 프로젝트 클론:
```bash
git clone https://github.com/[username]/dermscan_nocrop.git
cd dermscan_nocrop
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

1. 백엔드 서버 실행:
```bash
cd backend
python app.py
```

2. 웹 브라우저에서 접속:
```
http://localhost:5000
```

## 프로젝트 구조

```
dermscan_nocrop/
├── backend/
│   ├── app.py                 # 백엔드 서버 코드
│   ├── models/
│   │   ├── swin_model.py     # Swin Transformer 모델
│   │   └── vi_model.py       # ViT 모델
│   ├── static/
│   │   └── images/           # 업로드된 이미지 저장 폴더
│   └── templates/
│       ├── index.html        # 메인 페이지
│       ├── admin.html        # 관리자 페이지
│       └── 404.html          # 404 에러 페이지
└── requirements.txt          # 프로젝트 의존성
```

## API 엔드포인트

- `POST /api/analyze`: 이미지 분석
- `POST /api/feedback`: 피드백 저장
- `GET /api/admin/feedback`: 피드백 목록 조회
- `GET /api/admin/statistics`: 피드백 통계
- `GET /api/admin/analysis-log`: 분석 로그 조회
- `GET /api/admin/analysis-log/statistics`: 분석 로그 통계
- `POST /api/admin/analysis-log/delete`: 분석 로그 삭제
- `POST /api/admin/feedback/delete`: 피드백 삭제

## 주의사항

- 이 애플리케이션은 의학적 조언을 대체하지 않습니다.
- 정확한 진단은 반드시 의사와 상담하세요.
- 분석 결과는 참고용으로만 사용하세요.
- 개인정보 보호를 위해 업로드된 이미지는 분석 후 서버에서 자동으로 삭제됩니다.

## 라이선스

이 프로젝트는 비공개(Private) 프로젝트입니다. 무단 복제 및 배포를 금지합니다.
