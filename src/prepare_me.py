# src/prepare_me.py
import os
from pathlib import Path

import numpy as np
import face_recognition

BASE_DIR = Path(__file__).resolve().parent.parent
REF_DIR = BASE_DIR / "data" / "me_face_refs"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    encodings = []

    img_paths = list(REF_DIR.glob("*.*"))
    if not img_paths:
        print(f"[ERROR] {REF_DIR} 에 얼굴 기준 사진이 없습니다.")
        return

    for img_path in img_paths:
        print(f"[INFO] 처리 중: {img_path.name}")
        img = face_recognition.load_image_file(img_path)
        boxes = face_recognition.face_locations(img)

        if not boxes:
            print(f"  -> 얼굴을 찾지 못했습니다. 건너뜀.")
            continue

        enc = face_recognition.face_encodings(img, boxes)[0]
        encodings.append(enc)

    if not encodings:
        print("[ERROR] 어떤 사진에서도 얼굴을 찾지 못했습니다.")
        return

    me_face_encoding = np.mean(encodings, axis=0)
    output_path = OUTPUT_DIR / "me_face_encoding.npy"
    np.save(output_path, me_face_encoding)
    print(f"[OK] 내 얼굴 기준 벡터 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
