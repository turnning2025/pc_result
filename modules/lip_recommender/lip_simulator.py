import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh

class LipNotFoundError(Exception):
    pass

# Mediapipe 입술 인덱스
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# 입 안쪽 경계 인덱스
INNER_MOUTH = [13, 14, 312, 311, 310, 415, 308, 324]


# -----------------------------
# Polygon 확장
# -----------------------------
def expand_polygon(points, scale=1.08):
    center = np.mean(points, axis=0)
    expanded = []
    for p in points:
        direction = p - center
        expanded.append(center + direction * scale)
    return np.array(expanded, dtype=np.int32)


# -----------------------------
# 입 안쪽 polygon
# -----------------------------
def get_inner_mouth_polygon(face, w, h, scale=1.15):
    pts = []
    for idx in INNER_MOUTH:
        pts.append([
            face.landmark[idx].x * w,
            face.landmark[idx].y * h
        ])
    pts = np.array(pts, np.float32)
    pts = expand_polygon(pts, scale=scale)
    return pts.astype(np.int32)


# -----------------------------
# 입벌림 정도
# -----------------------------
def get_inner_mask_strength(face, h):
    upper_inner = face.landmark[13].y * h
    lower_inner = face.landmark[14].y * h
    gap = abs(lower_inner - upper_inner)

    if gap < 4:
        return 0.0
    if gap > 12:
        return 1.0

    weight = (gap - 4) / (12 - 4)
    return weight * 0.7


# -----------------------------
# 치아/입 안쪽 제거 마스크
# -----------------------------
def get_teeth_mask(image, base_mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    bright = cv2.threshold(v, 170, 255, cv2.THRESH_BINARY)[1]
    low_sat = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY_INV)[1]

    raw_teeth = cv2.bitwise_and(bright, low_sat)
    teeth = cv2.bitwise_and(raw_teeth, base_mask)

    teeth = cv2.GaussianBlur(teeth, (9, 9), 4)
    return teeth


# -----------------------------
# 입술 마스크 생성
# -----------------------------
def get_lip_mask(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as mesh:

        results = mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise LipNotFoundError("입술 인식 실패")

        h, w, _ = image.shape
        face = results.multi_face_landmarks[0]

        upper = np.array([
            (int(face.landmark[i].x * w), int(face.landmark[i].y * h))
            for i in UPPER_LIP
        ], np.int32)

        lower = np.array([
            (int(face.landmark[i].x * w), int(face.landmark[i].y * h))
            for i in LOWER_LIP
        ], np.int32)

        upper = expand_polygon(upper, 1.08)
        lower = expand_polygon(lower, 1.10)

        lip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lip_mask, [upper], 255)
        cv2.fillPoly(lip_mask, [lower], 255)

        strength = get_inner_mask_strength(face, h)

        inner_poly = get_inner_mouth_polygon(face, w, h, scale=1.18)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inner_mask, [inner_poly], 255)

        teeth_mask = get_teeth_mask(image, base_mask=cv2.bitwise_or(lip_mask, inner_mask))

        inner_mask_blur = cv2.GaussianBlur(inner_mask, (13, 13), 6)
        inner_float = (inner_mask_blur.astype(np.float32) / 255.0) * strength
        teeth_float = (teeth_mask.astype(np.float32) / 255.0) * 1.0

        remove_float = np.clip(inner_float + teeth_float, 0.0, 1.0)
        lip_mask = (lip_mask.astype(np.float32) * (1.0 - remove_float)).astype(np.uint8)

        lip_mask = cv2.GaussianBlur(lip_mask, (13, 13), 8)
        return lip_mask


# -----------------------------
# 립 합성
# -----------------------------
def apply_lip_color(image, lip_mask, color_rgb):
    h, w, _ = image.shape

    # -----------------------------
    # 1) 목표 색(BGR) & 결과 색 템플릿 생성
    # -----------------------------
    desired_color_BGR = np.array(color_rgb[::-1], dtype=np.uint8)
    desired = np.full((h, w, 3), desired_color_BGR, dtype=np.uint8)

    # -----------------------------
    # 2) 입술 텍스처 보존 (원본 - 블러)
    # -----------------------------
    blur = cv2.GaussianBlur(image, (21, 21), 10)
    texture = cv2.subtract(image, blur)
    texture = np.clip(texture * 1.30, 0, 255)

    # -----------------------------
    # 3) 부드러운 그라데이션 마스크(입술 중심 → 외곽)
    # -----------------------------
    dist = cv2.distanceTransform((lip_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist = dist / (dist.max() + 1e-6)
    grad_mask = (0.7 + dist * 0.3)  # shape: (H, W)
    grad_mask = grad_mask[:, :, None]  # (H, W, 1)

    # -----------------------------
    # 4) gloss(광택)
    # -----------------------------
    gloss = cv2.GaussianBlur(desired, (17, 17), 15)
    gloss = np.clip((gloss - desired) * 0.25, 0, 255)

    # -----------------------------
    # 5) 최종 결과 색 생성
    # -----------------------------
    result = np.clip(desired + texture + gloss, 0, 255).astype(np.float32)

    # -----------------------------
    # 6) lip_mask를 안정적으로 3채널로 변환 (핵심 fix)
    # -----------------------------
    mask = lip_mask.astype(np.float32) / 255.0  # (H, W) 또는 (H, W, 1)

    # (H, W) → (H, W, 1)
    if mask.ndim == 2:
        mask = mask[:, :, None]

    # grad_mask와 결합 → (H, W, 1)
    blend_strength = 0.70
    mask = mask * grad_mask * blend_strength

    # 이제 반드시 (H, W, 3)로 확장
    if mask.shape[2] == 1:
        mask = np.repeat(mask, 3, axis=2)

    # -----------------------------
    # 7) Blending
    # -----------------------------
    blended = image.astype(np.float32) * (1 - mask) + result * mask
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# -----------------------------
# 비포/애프터 생성
# -----------------------------
def simulate_lip_color(image_path, color_rgb):
    img_path = Path(image_path)
    image = cv2.imread(str(img_path))

    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    lip_mask = get_lip_mask(image)
    result_image = apply_lip_color(image, lip_mask, color_rgb=color_rgb)

    # 저장하지 않고 이미지 반환만
    return result_image
