import cv2
import numpy as np
from skimage import color

# FaceMesh 눈 영역 인덱스
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]


# -------------------------------------------------------
# Polygon 확장 (눈 ROI 너무 작게 잡히는 문제 해결)
# -------------------------------------------------------
def expand_polygon(points, scale=1.25):
    center = np.mean(points, axis=0)
    expanded = []
    for p in points:
        direction = p - center
        expanded.append(center + direction * scale)
    return np.array(expanded, dtype=np.int32)


# -------------------------------------------------------
# 눈 영역 ROI 추출
# -------------------------------------------------------
def extract_eye_roi(img, landmarks, eye='both'):
    """
    FaceMesh landmarks 기반 눈 영역 마스크 생성
    :param img: BGR 이미지
    :param landmarks: FaceMesh landmarks
    :param eye: 'left', 'right', 'both'
    :return: dict {'left': BGR array, 'right': BGR array}
    """
    h, w, _ = img.shape
    eye_pixels = {}

    eyes_to_use = []
    if eye in ['left', 'both']:
        eyes_to_use.append(('left', LEFT_EYE_INDICES))
    if eye in ['right', 'both']:
        eyes_to_use.append(('right', RIGHT_EYE_INDICES))

    mask = np.zeros((h, w), dtype=np.uint8)

    for name, indices in eyes_to_use:
        # FaceMesh 좌표 → 픽셀 좌표
        points = np.array([
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in indices
        ], np.int32)

        # --- 핵심: 눈 ROI 확장 (AI 이미지/실사 안정성↑) ---
        points = expand_polygon(points, scale=1.25)

        # 폴리곤 기반 마스크 생성
        cv2.fillPoly(mask, [points], 255)

        # 실제 눈 픽셀 추출
        pixels = img[mask > 0]

        # 픽셀이 너무 적으면 fallback 처리
        if len(pixels) < 20:
            # 다시 한 번 더 넓게 확장 시도
            points2 = expand_polygon(points, scale=1.35)
            mask[:] = 0
            cv2.fillPoly(mask, [points2], 255)
            pixels = img[mask > 0]

        eye_pixels[name] = pixels

        # 다음 눈을 위해 초기화
        mask[:] = 0

    return eye_pixels


# -------------------------------------------------------
# 눈 동자색 LAB 계산
# -------------------------------------------------------
def compute_eye_color(eye_pixels):
    """
    눈 영역 픽셀(BGR) → 간단 화이트밸런스 → LAB 평균
    :return: dict {'left': LAB, 'right': LAB, 'both': LAB}
    """
    result = {}
    all_pixels = []

    for name, pixels in eye_pixels.items():
        if pixels is None or len(pixels) == 0:
            result[name] = None
            continue

        # 화이트밸런스 보정
        avg_bgr = np.mean(pixels, axis=0)
        adjusted = pixels.astype(np.float32) * (128 / (avg_bgr + 1e-6))
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        # LAB 변환
        rgb = adjusted[:, ::-1]  # BGR → RGB
        lab = color.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        lab_mean = lab.mean(axis=0)

        result[name] = lab_mean
        all_pixels.append(adjusted)

    # 좌/우 combined 색상
    if all_pixels:
        combined = np.vstack(all_pixels)
        rgb = combined[:, ::-1]
        lab_combined = color.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3).mean(axis=0)
        result['both'] = lab_combined
    else:
        result['both'] = None

    return result
