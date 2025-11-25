import cv2
import numpy as np
from skimage import color
from pathlib import Path

class SkinNotFoundError(Exception):
    """피부 영역을 찾지 못했을 때 발생"""
    pass

def white_balance(img, strength=0.2):
    """LAB 색공간 기반 단순 화이트밸런스"""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] -= (avg_a - 128) * strength
    result[:, :, 2] -= (avg_b - 128) * strength
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def extract_skin_mask(img):
    """피부 영역 마스크 추출 (YCrCb 색공간 기반)"""
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(img_ycrcb, lower, upper)
    return mask

def process_skin(image_path):
    """이미지 경로 → 피부 영역 색상 추출 + 화이트밸런스 적용"""
    img_path = Path(image_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    corrected = white_balance(img, 0.2)
    skin_mask = extract_skin_mask(corrected)
    skin_pixels = corrected[skin_mask > 0]

    if len(skin_pixels) == 0:
        raise SkinNotFoundError(f"피부 영역을 찾지 못했습니다: {image_path}")

    skin_lab = color.rgb2lab(skin_pixels.reshape(-1, 1, 3)).reshape(-1, 3).mean(axis=0)
    return skin_lab, corrected, skin_mask
