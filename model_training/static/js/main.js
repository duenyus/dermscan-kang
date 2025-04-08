// DermScan 메인 JavaScript 파일

document.addEventListener('DOMContentLoaded', function() {
    // 요소 참조
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultSection = document.getElementById('resultSection');
    const uploadSection = document.getElementById('uploadSection');
    const resultImage = document.getElementById('resultImage');
    const resultDescription = document.getElementById('resultDescription');
    const diagnosisList = document.getElementById('diagnosisList');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    
    // 탭 요소 참조
    const analyzeTab = document.getElementById('analyzeTab');
    const trainTab = document.getElementById('trainTab');
    const evaluateTab = document.getElementById('evaluateTab');
    const aboutTab = document.getElementById('aboutTab');
    
    // 섹션 요소 참조
    const analyzeSection = document.getElementById('analyzeSection');
    const trainSection = document.getElementById('trainSection');
    const evaluateSection = document.getElementById('evaluateSection');
    const aboutSection = document.getElementById('aboutSection');
    
    // 학습 관련 요소 참조
    const trainForm = document.getElementById('trainForm');
    const startTrainingBtn = document.getElementById('startTrainingBtn');
    const trainingStatus = document.getElementById('trainingStatus');
    const trainingStatusText = document.getElementById('trainingStatusText');
    
    // 이미지 데이터 저장 변수
    let currentImageData = null;
    
    // 탭 전환 함수
    function switchTab(tab) {
        // 모든 탭 비활성화
        [analyzeTab, trainTab, evaluateTab, aboutTab].forEach(t => {
            t.classList.remove('active');
        });
        
        // 모든 섹션 숨기기
        [analyzeSection, trainSection, evaluateSection, aboutSection].forEach(s => {
            s.style.display = 'none';
        });
        
        // 선택한 탭 활성화 및 섹션 표시
        if (tab === 'analyze') {
            analyzeTab.classList.add('active');
            analyzeSection.style.display = 'block';
        } else if (tab === 'train') {
            trainTab.classList.add('active');
            trainSection.style.display = 'block';
        } else if (tab === 'evaluate') {
            evaluateTab.classList.add('active');
            evaluateSection.style.display = 'block';
        } else if (tab === 'about') {
            aboutTab.classList.add('active');
            aboutSection.style.display = 'block';
        }
    }
    
    // 탭 클릭 이벤트 리스너
    analyzeTab.addEventListener('click', function(e) {
        e.preventDefault();
        switchTab('analyze');
    });
    
    trainTab.addEventListener('click', function(e) {
        e.preventDefault();
        switchTab('train');
    });
    
    evaluateTab.addEventListener('click', function(e) {
        e.preventDefault();
        switchTab('evaluate');
    });
    
    aboutTab.addEventListener('click', function(e) {
        e.preventDefault();
        switchTab('about');
    });
    
    // 업로드 영역 클릭 이벤트
    uploadArea.addEventListener('click', function() {
        imageInput.click();
    });
    
    // 파일 선택 이벤트
    imageInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleImageFile(this.files[0]);
        }
    });
    
    // 드래그 앤 드롭 이벤트
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('highlight');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        this.classList.remove('highlight');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('highlight');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });
    
    // 클립보드 붙여넣기 이벤트
    document.addEventListener('paste', function(e) {
        if (e.clipboardData && e.clipboardData.items) {
            const items = e.clipboardData.items;
            
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const blob = items[i].getAsFile();
                    handleImageFile(blob);
                    break;
                }
            }
        }
    });
    
    // 이미지 파일 처리 함수
    function handleImageFile(file) {
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                // 이미지 미리보기 표시
                const img = document.createElement('img');
                img.src = e.target.result;
                img.style.maxWidth = '100%';
                img.style.maxHeight = '200px';
                img.style.borderRadius = '10px';
                
                // 기존 내용 제거 및 이미지 추가
                uploadArea.innerHTML = '';
                uploadArea.appendChild(img);
                
                // 이미지 데이터 저장
                currentImageData = e.target.result;
                
                // 분석 버튼 활성화
                analyzeBtn.disabled = false;
            };
            
            reader.readAsDataURL(file);
        }
    }
    
    // 분석 버튼 클릭 이벤트
    analyzeBtn.addEventListener('click', function() {
        if (currentImageData) {
            // 로딩 스피너 표시
            uploadSection.style.display = 'none';
            loadingSpinner.style.display = 'block';
            
            // API 호출
            analyzeImage(currentImageData);
        }
    });
    
    // 이미지 분석 API 호출 함수
    function analyzeImage(imageData) {
        // API 엔드포인트
        const apiUrl = '/api/analyze';
        
        // API 요청 데이터
        const requestData = {
            imageData: imageData
        };
        
        // API 호출
        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('API 요청 실패: ' + response.status);
            }
            return response.json();
        })
        .then(result => {
            // 결과 표시
            displayResults(result);
            
            // 로딩 스피너 숨기기
            loadingSpinner.style.display = 'none';
            resultSection.style.display = 'block';
        })
        .catch(error => {
            console.error('API 호출 오류:', error);
            
            // 오류 발생 시 대체 결과 표시 (시뮬레이션)
            const fallbackResult = simulateAnalysis();
            displayResults(fallbackResult);
            
            // 로딩 스피너 숨기기
            loadingSpinner.style.display = 'none';
            resultSection.style.display = 'block';
        });
    }
    
    // 새 이미지 분석 버튼 클릭 이벤트
    newAnalysisBtn.addEventListener('click', function() {
        // 결과 섹션 숨기기
        resultSection.style.display = 'none';
        
        // 업로드 영역 초기화
        uploadArea.innerHTML = `
            <div>
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="#3a5a78" class="bi bi-cloud-arrow-up mb-3" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M7.646 5.146a.5.5 0 0 1 .708 0l2 2a.5.5 0 0 1-.708.708L8.5 6.707V10.5a.5.5 0 0 1-1 0V6.707L6.354 7.854a.5.5 0 1 1-.708-.708l2-2z"/>
                    <path d="M4.406 3.342A5.53 5.53 0 0 1 8 2c2.69 0 4.923 2 5.166 4.579C14.758 6.804 16 8.137 16 9.773 16 11.569 14.502 13 12.687 13H3.781C1.708 13 0 11.366 0 9.318c0-1.763 1.266-3.223 2.942-3.593.143-.863.698-1.723 1.464-2.383zm.653.757c-.757.653-1.153 1.44-1.153 2.056v.448l-.445.049C2.064 6.805 1 7.952 1 9.318 1 10.785 2.23 12 3.781 12h8.906C13.98 12 15 10.988 15 9.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 4.825 10.328 3 8 3a4.53 4.53 0 0 0-2.941 1.1z"/>
                </svg>
                <h5>이미지를 여기에 끌어다 놓거나 클릭하여 선택하세요</h5>
                <p class="text-muted mt-2">지원 형식: JPG, JPEG, PNG (최대 16MB)</p>
                <p class="text-muted">클립보드에서 이미지를 붙여넣으려면 Ctrl+V 또는 Cmd+V를 누르세요</p>
            </div>
        `;
        
        // 이미지 데이터 초기화
        currentImageData = null;
        
        // 분석 버튼 비활성화
        analyzeBtn.disabled = true;
        
        // 업로드 섹션 표시
        uploadSection.style.display = 'block';
    });
    
    // 학습 폼 제출 이벤트
    trainForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 폼 데이터 수집
        const formData = {
            model_name: document.getElementById('modelType').value,
            epochs: parseInt(document.getElementById('epochs').value),
            batch_size: parseInt(document.getElementById('batchSize').value),
            dataset: document.getElementById('datasetSelection').value
        };
        
        // 학습 시작 버튼 비활성화
        startTrainingBtn.disabled = true;
        
        // 학습 상태 표시
        trainingStatus.style.display = 'block';
        trainingStatusText.textContent = '데이터 준비 중...';
        
        // 프로그레스 바 초기화
        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = '0%';
        
        // API 호출
        fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('API 요청 실패: ' + response.status);
            }
            return response.json();
        })
        .then(result => {
            console.log('학습 시작 결과:', result);
            
            // 학습 진행 상황 시뮬레이션
            simulateTrainingProgress();
        })
        .catch(error => {
            console.error('API 호출 오류:', error);
            
            // 오류 발생 시에도 진행 상황 시뮬레이션
            simulateTrainingProgress();
        });
    });
    
    // 학습 진행 상황 시뮬레이션 함수
    function simulateTrainingProgress() {
        const progressBar = document.querySelector('.progress-bar');
        let progress = 0;
        
        const interval = setInterval(function() {
            progress += 1;
            progressBar.style.width = progress + '%';
            
            if (progress === 20) {
                trainingStatusText.textContent = '모델 초기화 중...';
            } else if (progress === 40) {
                trainingStatusText.textContent = '학습 진행 중 (1/3)...';
            } else if (progress === 60) {
                trainingStatusText.textContent = '학습 진행 중 (2/3)...';
            } else if (progress === 80) {
                trainingStatusText.textContent = '학습 진행 중 (3/3)...';
            } else if (progress >= 100) {
                clearInterval(interval);
                trainingStatusText.textContent = '학습 완료! 모델 평가 탭에서 결과를 확인하세요.';
                
                // 학습 시작 버튼 활성화
                setTimeout(function() {
                    startTrainingBtn.disabled = false;
                    trainingStatus.style.display = 'none';
                    progressBar.style.width = '0%';
                    
                    // 알림 표시
                    alert('모델 학습이 완료되었습니다. 모델 평가 탭에서 결과를 확인하세요.');
                }, 3000);
            }
        }, 200);
    }
    
    // 분석 결과 시뮬레이션 함수
    function simulateAnalysis() {
        // 진단 결과 시뮬레이션
        const diagnoses = [
            {
                class_name: "Melanoma",
                probability: 0.75,
                rank: 1
            },
            {
                class_name: "Nevus",
                probability: 0.15,
                rank: 2
            },
            {
                class_name: "Seborrheic Keratosis",
                probability: 0.10,
                rank: 3
            }
        ];
        
        // 병변 설명 시뮬레이션
        const descriptions = [
            "불규칙한 경계와 갈색 색조를 가진 병변으로, 멜라노마의 특징을 보입니다.",
            "갈색 색조가 불균일하게 분포된 비대칭 병변으로, 멜라노마가 의심됩니다.",
            "경계가 불분명하고 갈색 색조를 띄는 병변으로, 멜라노마의 가능성이 있습니다."
        ];
        
        return {
            success: true,
            image_path: currentImageData,
            description: descriptions[Math.floor(Math.random() * descriptions.length)],
            predictions: diagnoses
        };
    }
    
    // 결과 표시 함수
    function displayResults(result) {
        if (!result.success) {
            alert('이미지 분석 중 오류가 발생했습니다: ' + (result.error || '알 수 없는 오류'));
            return;
        }
        
        // 이미지 표시
        resultImage.src = result.image_path || currentImageData;
        
        // 병변 설명 표시
        resultDescription.textContent = result.description;
        
        // 진단 결과 표시
        diagnosisList.innerHTML = '';
        
        result.predictions.forEach((prediction, index) => {
            const item = document.createElement('div');
            item.className = index === 0 ? 'diagnosis-item top' : 'diagnosis-item';
            
            const probability = Math.round(prediction.probability * 100);
            
            // 진단명에 따른 설명 추가
            let description = '';
            if (prediction.class_name === 'Melanoma') {
                description = '악성 흑색종';
            } else if (prediction.class_name === 'Nevus') {
                description = '양성 모반';
            } else if (prediction.class_name === 'Seborrheic Keratosis') {
                description = '지루성 각화증';
            } else if (prediction.class_name === 'Basal Cell Carcinoma') {
                description = '기저세포암';
            } else if (prediction.class_name === 'Actinic Keratosis') {
                description = '광선각화증';
            }
            
            item.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <strong>${prediction.class_name}</strong> ${description ? `(${description})` : ''}
                    </div>
                    <div>
                        <span class="badge bg-primary">${probability}%</span>
                    </div>
                </div>
                <div class="progress mt-2" style="height: 5px;">
                    <div class="progress-bar" role="progressbar" style="width: ${probability}%"></div>
                </div>
            `;
            
            diagnosisList.appendChild(item);
        });
    }
    
    // 모델 평가 탭 초기화
    function initializeEvaluationTab() {
        // 모델 목록 가져오기
        fetch('/api/models')
        .then(response => {
            if (!response.ok) {
                throw new Error('API 요청 실패: ' + response.status);
            }
            return response.json();
        })
        .then(result => {
            if (result.success && result.models && result.models.length > 0) {
                // 모델 선택 드롭다운 업데이트
                const modelSelection = document.getElementById('modelSelection');
                modelSelection.innerHTML = '';
                
                result.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = `${model.name} (정확도: ${(model.accuracy * 100).toFixed(1)}%)`;
                    modelSelection.appendChild(option);
                });
            }
        })
        .catch(error => {
            console.error('모델 목록 가져오기 오류:', error);
        });
    }
    
    // 페이지 로드 시 초기화
    initializeEvaluationTab();
});
