# src/build_index_bibs.py
import os
from pathlib import Path
import json

import cv2
import easyocr
from tqdm import tqdm

# OpenMP 충돌 우회
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent.parent
PHOTOS_DIR = BASE_DIR / "data" / "photos"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SIZE = 1600  # Resize

def load_and_resize(path: Path):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise RuntimeError("이미지 로드 실패 (cv2.imread 반환값 None)")

    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h))

    return img_bgr

def main():
    print("[INFO] easyocr Reader 초기화 중")
    reader = easyocr.Reader(['en'], gpu=True)  # GPU 사용

    img_paths = [
        p for p in PHOTOS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg"] # 사용하려는 이미지 포맷에 맞추어 변경
    ]

    if not img_paths:
        print(f"[ERROR] {PHOTOS_DIR} 및 하위 폴더에서 .JPG 파일을 찾지 못했습니다.")
        return

    print(f"[INFO] OCR 대상 이미지 수: {len(img_paths)}")

    index = []

    for img_path in tqdm(img_paths, desc="배번호(OCR) 인덱싱"):
        try:
            img = load_and_resize(img_path)
            result = reader.readtext(img, detail=0) # detail=0 -> 텍스트만 리스트로 반환
            if not result:
                continue

            cleaned = [r.replace(" ", "") for r in result]
            index.append({
                "path": str(img_path.relative_to(BASE_DIR)),
                "texts": cleaned,
            })
        except Exception as e:
            print(f"[WARN] {img_path} 처리 실패: {e}")

    out_path = OUTPUT_DIR / "index_bibs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"[OK] 배번호 인덱스 저장 완료: {out_path} (총 {len(index)}장)")

if __name__ == "__main__":
    main()
