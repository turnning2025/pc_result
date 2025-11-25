import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh

class LipNotFoundError(Exception):
    pass

######## TEST LIP COLOR ########
TEST_LIP_COLORS = {
    "deep_red": (200, 30, 70),
    "pink": (220, 100, 150),
    "coral": (255, 90, 70)
}

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
    """입 안쪽 폴리곤을 약간 키워서 치아 영역까지 넉넉히 포함"""
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
# 3단계 입벌림 계산 (부분 제거용)
# -----------------------------
def get_inner_mask_strength(face, h):
    upper_inner = face.landmark[13].y * h
    lower_inner = face.landmark[14].y * h
    gap = abs(lower_inner - upper_inner)

    # 완전 닫힘
    if gap < 4:
        return 0.0

    # 완전 벌림
    if gap > 12:
        return 1.0

    # 4~12px → 0~0.7까지 선형 증가
    weight = (gap - 4) / (12 - 4)
    return weight * 0.7


# -----------------------------
# 치아/밝은 입안 마스크 생성
# -----------------------------
def get_teeth_mask(image, base_mask):
    """
    HSV 밝기/채도 기반으로 치아 영역 추정.
    base_mask (lip_mask or inner_mask)와 AND해서 입 주변만 사용.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 치아 특징: 밝고(v↑), 채도 낮음(s↓)
    bright = cv2.threshold(v, 170, 255, cv2.THRESH_BINARY)[1]
    low_sat = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY_INV)[1]

    raw_teeth = cv2.bitwise_and(bright, low_sat)

    # 입 주변에만 제한
    teeth = cv2.bitwise_and(raw_teeth, base_mask)

    # 살짝 블러로 부드럽게
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

        # 입술 좌표
        upper = np.array([(int(face.landmark[i].x * w),
                           int(face.landmark[i].y * h)) for i in UPPER_LIP], np.int32)
        lower = np.array([(int(face.landmark[i].x * w),
                           int(face.landmark[i].y * h)) for i in LOWER_LIP], np.int32)

        # 안정적 polygon 확장
        upper = expand_polygon(upper, 1.08)
        lower = expand_polygon(lower, 1.10)

        lip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lip_mask, [upper], 255)
        cv2.fillPoly(lip_mask, [lower], 255)

        # ===== 입 안쪽 + 치아 제거 로직 =====
        strength = get_inner_mask_strength(face, h)

        # inner polygon (입안 영역 넉넉히)
        inner_poly = get_inner_mouth_polygon(face, w, h, scale=1.18)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inner_mask, [inner_poly], 255)

        # 치아 마스크 (입 주변 밝은 영역)
        teeth_mask = get_teeth_mask(image, base_mask=cv2.bitwise_or(lip_mask, inner_mask))

        # inner_mask는 부분 제거 (strength), teeth_mask는 강력 제거(1.0)
        inner_mask_blur = cv2.GaussianBlur(inner_mask, (13, 13), 6)
        inner_float = (inner_mask_blur.astype(np.float32) / 255.0) * strength

        teeth_float = (teeth_mask.astype(np.float32) / 255.0) * 1.0

        # 두 제거 마스크 합치기
        remove_float = np.clip(inner_float + teeth_float, 0.0, 1.0)

        # lip_mask에서 제거
        lip_mask = (lip_mask.astype(np.float32) * (1.0 - remove_float)).astype(np.uint8)

        # Feathering (최종 경계 부드럽게)
        lip_mask = cv2.GaussianBlur(lip_mask, (13, 13), 8)
        return lip_mask


# -----------------------------
# 립 합성 (프리미엄 버전)
# -----------------------------
def apply_lip_color(image, lip_mask, color_rgb):
    h, w, _ = image.shape

    # 1) 원하는 립 색 (BGR)
    desired_color_BGR = np.array(color_rgb[::-1], dtype=np.uint8)
    desired = np.tile(desired_color_BGR, (h, w, 1))

    # 2) 입술 텍스처 추출 및 강화
    blur = cv2.GaussianBlur(image, (21, 21), 10)
    texture = cv2.subtract(image, blur)
    texture = np.clip(texture * 1.30, 0, 255)

    # 3) 외곽 더 진한 자연 그라데이션 (원하면 mask에 곱해서 쓸 수도 있음)
    dist = cv2.distanceTransform((lip_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist = dist / (dist.max() + 1e-6)
    grad_mask = (0.7 + dist * 0.3)  # 중앙 0.7, 외곽 1.0
    grad_mask = grad_mask[..., None]

    # 4) Gloss 반사광
    gloss = cv2.GaussianBlur(desired, (17, 17), 15)
    gloss = gloss - desired
    gloss = np.clip(gloss * 0.25, 0, 255)

    # 5) 전체 결과
    result = desired + texture + gloss
    result = np.clip(result, 0, 255)

    # 6) 립 색 강도 조절 + 그라데이션 반영
    blend_strength = 0.70
    mask = (lip_mask.astype(np.float32) / 255.0) * blend_strength
    mask = mask[..., None]
    mask = mask * grad_mask  # 중앙 살짝 약하게, 외곽 진하게

    blended = image * (1 - mask) + result * mask
    return blended.astype(np.uint8)


# -----------------------------
# 비포/애프터 생성
# -----------------------------
def generate_before_after(image_path, color_rgb):
    img_path = Path(image_path)
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    lip_mask = get_lip_mask(image)
    result_image = apply_lip_color(image, lip_mask, color_rgb=color_rgb)

    save_path = img_path.with_name(img_path.stem + "_lip.jpg")
    cv2.imwrite(str(save_path), result_image)
    return result_image
