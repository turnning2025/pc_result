import os
import cv2
import numpy as np
from pathlib import Path

# ------------------------
# 모듈 불러오기
# ------------------------
from modules.palette_processor import load_all_palettes
from modules.face_detector import detect_face, FaceNotFoundError
from modules.face_mesh_utils import init_face_mesh
from modules.skin_extractor import process_skin, SkinNotFoundError
from modules.season_classifier import SeasonKNNClassifier
from modules.face_box import save_face_box
from modules.visualize_palette import append_palette_to_face
from modules.face_visualize import visualize_facemesh, FaceNotFoundError as MeshNotFoundError
from modules.eye_extractor import extract_eye_roi, compute_eye_color
from modules.season_visualizer import visualize_skin_position

# 립 관련
from modules.lip_recommender.lip_preprocess import load_and_preprocess_lip_csv
from modules.lip_recommender.lip_recommender import recommend_lip_colors
from modules.lip_recommender.lip_simulator import simulate_lip_color


# ============================================================
# 메인 함수
# ============================================================
def main():
    print("팔레트 로딩 중...")
    BASE_DIR = Path(__file__).resolve().parent
    palettes_dir = BASE_DIR / "palettes"
    palettes = load_all_palettes(palettes_dir)

    # ------------------------
    # 1) 이미지 입력
    # ------------------------
    img_path = Path(input("이미지 경로를 입력하세요: ").strip())
    if not img_path.exists():
        print("이미지를 찾을 수 없습니다.")
        return

    # ------------------------
    # 2) FaceMesh 기반 얼굴 인식 및 박스 저장
    # ------------------------
    print("얼굴 인식 중...")
    try:
        # detect_face는 이제 FaceMesh 기반
        img, face_crop, bbox = detect_face(img_path)
        save_face_box(img_path, bbox)  # bbox 직접 전달
    except FaceNotFoundError:
        print("얼굴을 찾을 수 없습니다.")
        return

    # ------------------------
    # 3) FaceMesh landmarks 추출 + 시각화
    # ------------------------
    print("FaceMesh 시각화 중...")

    mesh = init_face_mesh() 
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        print("FaceMesh 인식 실패")
        landmarks = None
    else:
        face = result.multi_face_landmarks[0]
        landmarks = face.landmark   
        visualize_facemesh(str(img_path))



    # ------------------------
    # 4) 피부 색 추출
    # ------------------------
    print("피부 색 추출 중...")
    try:
        skin_lab, corrected_img, skin_mask = process_skin(img_path)
    except SkinNotFoundError as e:
        print(f"피부 추출 실패: {e}")
        return

    # ------------------------
    # 5) 눈동자 색 추출
    # ------------------------
    print("눈동자 색 추출 중...")
    try:
        img = cv2.imread(str(img_path))
        if landmarks is not None:
            eye_pixels = extract_eye_roi(img, landmarks, eye='both')
            eye_lab = compute_eye_color(eye_pixels)['both']
        else:
            eye_lab = None
    except Exception:
        print("눈동자 인식 실패 → 눈 색 보정 없이 진행")
        eye_lab = None

    # ------------------------
    # 6) 시즌 판정
    # ------------------------
    print("시즌 판정 중...")
    season_clf = SeasonKNNClassifier(palettes)

    # 기본 사용자 입력 값
    season_input = skin_lab.copy().astype(float)

    # 눈색 5% 반영
    if eye_lab is not None:
        season_input[1] = skin_lab[1] * 0.95 + eye_lab[1] * 0.05
        season_input[2] = skin_lab[2] * 0.95 + eye_lab[2] * 0.05

    # L 채널 자동 미세 보정
    L = skin_lab[0]
    if L < 40:
        season_input[0] = L * 1.03
    elif L > 70:
        season_input[0] = L * 0.97
    else:
        season_input[0] = L

    # 시즌 예측
    user_season = season_clf.predict_season(season_input)

    print(f"판정된 시즌: {user_season}")
    print("skin_lab:", skin_lab)
    print("season_input:", season_input)

    # 피부 위치 시각화
    visualize_skin_position(
        palettes,
        season_input,
        save_path=str(img_path.parent / "skin_position.jpg")
    )

    # ------------------------
    # 7) 시즌 팔레트 시각화
    # ------------------------
    print("팔레트 합성 중...")
    try:
        palette_df = palettes[user_season]

        save_path = img_path.parent / "test_images" / "palette_result.jpg"
        save_path.parent.mkdir(exist_ok=True)

        append_palette_to_face(
            img_path,
            palette_df,
            save_path=str(save_path),
            block_size=100,
            max_rows=2
        )
    except Exception as e:
        print(f"팔레트 합성 실패: {e}")

    # ------------------------
    # 8) 립 CSV 로드
    # ------------------------
    print("립 데이터 로딩 중...")
    lip_csv_path = BASE_DIR / "modules" / "lip_data" / "colorchips_data.csv"

    try:
        lip_df = load_and_preprocess_lip_csv(lip_csv_path)
    except Exception as e:
        print(f"립 CSV 불러오기 실패: {e}")
        return

    # ------------------------
    # 9) 립 추천
    # ------------------------
    print("립 추천 계산 중...")
    recommended = recommend_lip_colors(
        season_classifier=season_clf,
        user_season=user_season,
        skin_lab=season_input,
        lip_df=lip_df
    )

    print("최종 추천 TOP 5:")
    try:
        print(recommended[["brand", "option", "hex", "r", "g", "b"]])
    except:
        print(recommended)

    # ------------------------
    # 10) 립 합성 이미지 생성
    # ------------------------
    print("립 합성 이미지 생성 중...")

    save_dir = img_path.parent / "test_images"
    save_dir.mkdir(exist_ok=True)

    # 기존 파일 제거
    for f in save_dir.glob("lip_result_*.jpg"):
        f.unlink()

    for idx, (_, row) in enumerate(recommended.iterrows(), start=1):
        if idx > 5:
            break
        
        color_rgb = (row["r"], row["g"], row["b"])
        result_img = simulate_lip_color(str(img_path), color_rgb)

        save_path = save_dir / f"lip_result_{idx}.jpg"
        cv2.imwrite(str(save_path), result_img)

        print(f"{idx}번 옵션 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
