# src/query.py
import sys
from pathlib import Path
import json
import shutil
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
RES_DIR = OUTPUT_DIR / "res"

FACES_INDEX_PATH = OUTPUT_DIR / "index_faces.json"
BIBS_INDEX_PATH = OUTPUT_DIR / "index_bibs.json"
ME_FACE_PATH = OUTPUT_DIR / "me_face_encoding.npy"

# ===커스텀 설정 파트===
MY_BIB_NUMBER = "107"      # 배번호
FACE_THRESHOLD = 0.9       # 얼굴 유사도 임계값(낮을수록 엄격)
# ====================


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_same_person(enc, me_enc, threshold=0.6):
    enc = np.array(enc)
    dist = np.linalg.norm(enc - me_enc)
    return dist < threshold, float(dist)


def prepare_res_dir():
    if RES_DIR.exists():
        shutil.rmtree(RES_DIR)
    RES_DIR.mkdir(parents=True, exist_ok=True)


def copy_results(result):
    prepare_res_dir()

    for path, _ in result:
        src = BASE_DIR / path
        src_path = Path(src)

        dest = RES_DIR / src_path.name

        if dest.exists(): # 파일명 충돌 예외처리
            stem = src_path.stem
            suffix = src_path.suffix
            idx = 1
            while True:
                cand = RES_DIR / f"{stem}_{idx}{suffix}"
                if not cand.exists():
                    dest = cand
                    break
                idx += 1

        shutil.copy2(src, dest)

    print(f"[INFO] 결과 이미지가 {RES_DIR} 폴더에 복사됨 (총 {len(result)}장)")


def main():
    use_face_filter = 1
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if arg in ["0", "1"]:
            use_face_filter = int(arg)
        else:
            print("[WARN] 얼굴 필터 옵션은 0 or 1 로 입력하세요. 기본값 1 사용.")
    print(f"[INFO] 얼굴 필터 사용 여부: {use_face_filter}")

    # ---- 파일 존재 체크 ----
    if not FACES_INDEX_PATH.exists() or not BIBS_INDEX_PATH.exists():
        print("[ERROR] index_faces.json 또는 index_bibs.json 이 없습니다.")
        return

    if use_face_filter == 1 and not ME_FACE_PATH.exists():
        print("[ERROR] 얼굴 필터를 쓰려면 me_face_encoding.npy 가 필요합니다.")
        return

    print("[INFO] 인덱스 및 얼굴 벡터 로드 중...")
    faces_index = load_json(FACES_INDEX_PATH)
    bibs_index = load_json(BIBS_INDEX_PATH)
    me_face = np.load(ME_FACE_PATH) if use_face_filter else None

    # ---- 1차: 배번호 필터 ----
    print(f"[INFO] 배번호 {MY_BIB_NUMBER} 포함 사진 검색 중...")

    candidates_by_bib = set()
    for item in bibs_index:
        texts = item["texts"]
        joined = " ".join(texts)
        if MY_BIB_NUMBER in joined:
            candidates_by_bib.add(item["path"])

    print(f"[INFO] 배번호 후보 사진 수: {len(candidates_by_bib)}")

    # ---- 2차: 얼굴 필터 (옵션) ----
    result = []
    if use_face_filter == 1:
        print("[INFO] 얼굴 필터 적용 중...")

        for item in faces_index:
            path = item["path"]
            if path not in candidates_by_bib:
                continue

            for enc in item["face_encodings"]:
                same, dist = is_same_person(enc, me_face, FACE_THRESHOLD)
                if same:
                    result.append((path, dist))
                    break

        # dist 기준 정렬
        result.sort(key=lambda x: x[1])
        print(f"[RESULT] 얼굴+배번호 최종 후보 수: {len(result)}")

    else:
        print("[INFO] 얼굴 필터 OFF, 배번호만으로 최종 후보 선정")
        result = [(path, -1) for path in candidates_by_bib]
        print(f"[RESULT] 배번호 최종 후보 수: {len(result)}")

    # ---- 결과 출력 + 복사 ----
    for path, dist in result:
        print(f"{path}  (dist={dist:.4f})")

    if result:
        copy_results(result)
    else:
        print("[INFO] 복사할 결과 없음.")


if __name__ == "__main__":
    main()
