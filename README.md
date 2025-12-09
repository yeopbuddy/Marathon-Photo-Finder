# 🏃 Marathon Photo Finder  
**얼굴 인식 + OCR 기반 마라톤 사진 자동 검색 시스템**

마라톤 행사에서 촬영된 **수천 장의 사진 속에서 내 사진만 자동으로 골라주는 도구**입니다.  
배번호(OCR)와 얼굴 임베딩(face recognition)을 조합하여 정확한 후보만 선별합니다.

본 프로젝트는 Python의 face_recognition과 easyocr 라이브러리 기반이며 로컬 환경(Windows / Anaconda) 기준으로 제작되었습니다.

---

## 📁 프로젝트 구조

```
marathon_finder/
│
├── data/
│   ├── photos/              # 원본 사진 (폴더 구조 상관 없음, JPG/JPEG)
│   └── me_face_refs/        # 내 얼굴 사진
│
├── outputs/
│   ├── index_faces.json     # 얼굴 인덱싱 결과
│   ├── index_bibs.json      # OCR 인덱싱 결과
│   ├── me_face_encoding.npy # 내 얼굴 벡터
│   └── res/                 # 최종 검색 결과(복사됨)
│
├── src/
│   ├── build_index_faces.py # 얼굴 인덱스 생성
│   ├── build_index_bibs.py  # OCR 인덱스 생성
│   ├── prepare_me.py        # 내 얼굴 벡터 생성
│   └── query.py             # 검색 실행(배번호 + 얼굴 필터)
│
└── README.md
```

---

# 🚀 1. 설치 & 환경 구성

## 1) Conda 환경 생성

```bash
conda create -n marathon39 python=3.9
conda activate marathon39
```

## 2) 필수 패키지 설치

### 얼굴 인식 (dlib + face_recognition)

```bash
conda install -c conda-forge dlib
pip install face_recognition
```

### OCR (EasyOCR + Torch GPU)

```bash
pip install easyocr
```

GPU 충돌 방지:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE    # Windows CMD
$env:KMP_DUPLICATE_LIB_OK="TRUE" # PowerShell
```

### 기타 패키지

```bash
pip install opencv-python pillow tqdm numpy
```

---

# 🧩 2. 사전 준비

### `data/photos/`  
마라톤 사진 저장 (폴더 구조 자유)

### `data/me_face_refs/`  
내 얼굴 사진 여러장 저장 (얼굴이 잘 인식되기 위한 얼굴 중심의 사진 필요)

---

# 🔧 3. 인덱스 생성

### 얼굴 인덱스 생성

```bash
python src/build_index_faces.py
```

### 배번호 OCR 인덱스 생성

```bash
python src/build_index_bibs.py
```

### 내 얼굴 벡터 생성

```bash
python src/prepare_me.py
```

---

# 🔍 4. 검색 실행

### 얼굴 + 배번호 필터

```bash
python src/query.py 1
```

### 배번호만 사용 (얼굴 필터 OFF)

```bash
python src/query.py 0
```

---

# 📂 5. 결과 확인

결과물은:

```
outputs/res/
```

폴더에 자동 복사됨.

---

# ⚙ 파라미터 설정

`query.py` 상단에서 조정:

```python
MY_BIB_NUMBER = "10374"
FACE_THRESHOLD = 0.9
```

---

# 🧠 동작 방식

- easyocr → 모든 텍스트 추출 → 배번호 후보 필터링  
- face_recognition → 얼굴 벡터 비교 → 최종 후보 결정  
- 결과 파일 자동 복사

---

# ⭐ License

MIT License
