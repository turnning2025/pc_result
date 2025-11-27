import cv2
import numpy as np
from skimage import color
from pathlib import Path


class SkinNotFoundError(Exception):
    pass


# -------------------------------------------------------
# 1) 최소 화이트밸런스 (L만 5~8% 조절, a/b 보정 금지)
# -------------------------------------------------------
def minimal_white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[:, :, 0]
    avg_L = np.mean(L)

    # 5~8% 보정 — 절대 과보정 X
    target_L = 75.0
    ratio = target_L / (avg_L + 1e-6)
    ratio = np.clip(ratio, 0.92, 1.08)

    lab[:, :, 0] = np.clip(L * ratio, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# -------------------------------------------------------
# 2) 가장 넓은 중립 YCrCb 피부 범위
# -------------------------------------------------------
def skin_mask_ycrcb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 중립+보편 범위 (붉은/노란 피부 모두 감싸는 최소 조건)
    lower = np.array([0, 120, 90], dtype=np.uint8)
    upper = np.array([255, 168, 165], dtype=np.uint8)

    return cv2.inRange(ycrcb, lower, upper)


# -------------------------------------------------------
# 3) 가장 넓은 중립 HSV 피부 범위
# -------------------------------------------------------
def skin_mask_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 10, 40], dtype=np.uint8)
    upper1 = np.array([25, 200, 255], dtype=np.uint8)

    lower2 = np.array([160, 10, 40], dtype=np.uint8)
    upper2 = np.array([179, 200, 255], dtype=np.uint8)

    return cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )


# -------------------------------------------------------
# 4) 그림자/하이라이트 최소 제거 (필수만)
# -------------------------------------------------------
def minimal_extreme_filter(gray, mask):
    dark = (gray < 55).astype(np.uint8) * 255
    bright = (gray > 230).astype(np.uint8) * 255  # 밝은 부분 허용 넓힘

    exclude = cv2.bitwise_or(dark, bright)
    return cv2.bitwise_and(mask, cv2.bitwise_not(exclude))


# -------------------------------------------------------
# 5) 최종 피부 Lab 추출 (최소 가공)
# -------------------------------------------------------
def process_skin(image_path):
    img_path = Path(image_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    # 1) 최소 WB만 적용
    corrected = minimal_white_balance(img)

    # 2) 넓은 범위 피부마스크
    mask_y = skin_mask_ycrcb(corrected)
    mask_h = skin_mask_hsv(corrected)
    mask = cv2.bitwise_and(mask_y, mask_h)

    # 3) 최소 조명 제거
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    mask = minimal_extreme_filter(gray, mask)

    # 4) 가장 약한 blur
    mask = cv2.GaussianBlur(mask, (5, 5), 1)

    skin_pixels = corrected[mask > 0]
    if len(skin_pixels) < 400:   # threshold 완화
        raise SkinNotFoundError("피부 픽셀이 충분하지 않습니다.")

    # 5) Lab 변환 — 보정 없음
    rgb = skin_pixels[..., ::-1]
    lab = color.rgb2lab(rgb).reshape(-1, 3)

    # 6) 최종 피부색 = median
    skin_lab = np.median(lab, axis=0)

    return skin_lab, corrected, mask
