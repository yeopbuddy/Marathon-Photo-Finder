# src/build_index_faces.py
from pathlib import Path
import json

import cv2
import face_recognition
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PHOTOS_DIR = BASE_DIR / "data" / "photos"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SIZE = 1024  # Resize

def load_and_resize(path: Path):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise RuntimeError("이미지 로드 실패")

    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def main():
    index = []

    img_paths = [
        p for p in PHOTOS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg"] # 사용하려는 이미지 포맷에 맞추어 변경
    ]

    if not img_paths:
        print(f"[ERROR] {PHOTOS_DIR} 및 하위 폴더에서 .JPG 파일을 찾지 못했습니다.")
        return

    print(f"[INFO] 총 이미지 수: {len(img_paths)}")

    for img_path in tqdm(img_paths, desc="얼굴 인덱싱(단일 스레드)"):
        try:
            img = load_and_resize(img_path)

            boxes = face_recognition.face_locations(
                img,
                model="hog", # HOG 모델, 업샘플 0 → 속도 우선
                number_of_times_to_upsample=0,
            )
            if not boxes:
                continue

            encs = face_recognition.face_encodings(
                img,
                boxes,
                num_jitters=1, # num_jitters=1 → 속도 우선
            )

            index.append({
                "path": str(img_path.relative_to(BASE_DIR)),
                "face_encodings": [enc.tolist() for enc in encs],
            })
        except Exception as e:
            print(f"[WARN] {img_path} 처리 실패: {e}")

    out_path = OUTPUT_DIR / "index_faces.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"얼굴 인덱스 저장 완료: {out_path} (총 {len(index)}장)")

if __name__ == "__main__":
    main()
