# eye_extractor.py
import cv2
import numpy as np
from skimage import color

# FaceMesh 눈 영역 인덱스
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

def extract_eye_roi(img, landmarks, eye='both'):
    """
    FaceMesh landmarks 기반 눈 영역 마스크 생성
    :param img: BGR 이미지
    :param landmarks: FaceMesh landmarks
    :param eye: 'left', 'right', 'both'
    :return: dict {'left': BGR array, 'right': BGR array} 눈 영역 픽셀
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
        points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], np.int32)
        cv2.fillPoly(mask, [points], 255)
        pixels = img[mask > 0]
        eye_pixels[name] = pixels
        mask[:] = 0  # 다음 눈 초기화

    return eye_pixels

def compute_eye_color(eye_pixels):
    """
    눈 영역 픽셀(BGR) → 화이트밸런스 + LAB 평균
    :param eye_pixels: dict {'left': pixels, 'right': pixels}
    :return: dict {'left': LAB, 'right': LAB, 'both': LAB}
    """
    result = {}
    all_pixels = []

    for name, pixels in eye_pixels.items():
        if len(pixels) == 0:
            result[name] = None
            continue
        # 간단 화이트밸런스 (간접)
        avg_bgr = np.mean(pixels, axis=0)
        adjusted = pixels.astype(np.float32) * (128 / (avg_bgr + 1e-6))
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        lab = color.rgb2lab(adjusted.reshape(-1,1,3)[:, :, ::-1]).reshape(-1,3).mean(axis=0)
        result[name] = lab
        all_pixels.append(adjusted)

    if all_pixels:
        combined = np.vstack(all_pixels)
        lab_combined = color.rgb2lab(combined.reshape(-1,1,3)[:, :, ::-1]).reshape(-1,3).mean(axis=0)
        result['both'] = lab_combined
    else:
        result['both'] = None

    return result
