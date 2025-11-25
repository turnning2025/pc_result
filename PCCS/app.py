import os
from pathlib import Path
from modules.palette_processor import load_all_palettes
from modules.face_detector import detect_face, FaceNotFoundError
from modules.skin_extractor import process_skin, SkinNotFoundError
from modules.season_classifier import classify_season
from modules.face_box import save_face_box
from modules.lip_simulator import generate_before_after, TEST_LIP_COLORS
from modules.visualize_palette import append_palette_to_face
from modules.face_visualize import visualize_facemesh, FaceNotFoundError as MeshNotFoundError
from modules.eye_extractor import extract_eye_roi, compute_eye_color
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ------------------------------
# 메인 기능
# ------------------------------
def main():
    palettes_dir = Path("C:/PCCS/palettes")
    palettes = load_all_palettes(palettes_dir)

    img_path = Path(input("이미지 경로를 입력하세요: ").strip())
    if not img_path.exists():
        print(f"이미지를 찾을 수 없음: {img_path}")
        return

    # 얼굴 인식
    print("얼굴 인식 중...")
    try:
        _, face_crop, bbox = detect_face(img_path)
    except FaceNotFoundError:
        print("얼굴을 찾지 못했습니다.")
        return

    # 얼굴 박스 저장
    save_face_box(img_path)
    print("얼굴 박스 저장 완료")

    # 피부 색 추출
    print("피부 색 추출 중...")
    try:
        skin_lab, _, _ = process_skin(img_path)
    except SkinNotFoundError:
        print("피부 색 추출 실패")
        return

    # 눈동자 색 추출
    print("눈동자 색 추출 중...")
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eye_lab = None
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5) as mesh:
            result = mesh.process(img_rgb)
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                eye_pixels = extract_eye_roi(img, landmarks, eye='both')
                eye_lab_dict = compute_eye_color(eye_pixels)
                eye_lab = eye_lab_dict['both']  # 좌우 평균
            else:
                print("FaceMesh에서 눈 랜드마크를 찾지 못함")
    except Exception as e:
        print(f"눈동자 색 추출 실패: {e}")

    # 시즌 판정 (피부 + 눈색 통합)
    print("시즌 판정 중...")
    season_score_input = skin_lab.copy()
    if eye_lab is not None:
        # 단순 통합: L 채널 기준 가중치 합산 (예시)
        season_score_input[0] = skin_lab[0]*0.7 + eye_lab[0]*0.3

    season = classify_season(season_score_input, palettes)
    print(f"판정된 시즌: {season}")

    # 팔레트 합성 (2줄 자동 분할 + 검은색 문제 해결)
    print("퍼스널컬러 팔레트 합성 중...")
    try:
        palette_df = palettes[season]
        append_palette_to_face(
            img_path,
            palette_df,
            save_path=str(img_path.parent / f"{img_path.stem}_palette.jpg"),
            block_size=100,
            max_rows=2
        )
    except Exception as e:
        print(f"팔레트 합성 실패: {e}")

    # 립 합성
    print("립 합성 이미지 생성 중...")
    try:
        generate_before_after(str(img_path), TEST_LIP_COLORS["pink"])
    except Exception as e:
        print(f"립 합성 실패: {e}")

    # FaceMesh 시각화
    print("FaceMesh 시각화 중...")
    try:
        visualize_facemesh(str(img_path))
    except MeshNotFoundError:
        print("FaceMesh 랜드마크를 찾지 못했습니다.")
    except FileNotFoundError:
        print("이미지를 찾을 수 없음")

# ------------------------------
# 실행부
# ------------------------------
if __name__ == "__main__":
    main()
