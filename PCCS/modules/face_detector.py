# modules/face_detector.py
import cv2
import mediapipe as mp
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh

class FaceNotFoundError(Exception):
    """FaceMesh로 얼굴을 찾지 못했을 때 발생"""
    pass


# ---------------------------------------
# FaceMesh 전체 랜드마크 추출
# ---------------------------------------
def get_facemesh_landmarks(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as mesh:

        result = mesh.process(img_rgb)

        if not result.multi_face_landmarks:
            raise FaceNotFoundError(f"FaceMesh 랜드마크를 찾을 수 없음: {image_path}")

        return result.multi_face_landmarks[0]


# ---------------------------------------
# FaceMesh → 얼굴 박스(bounding box) 생성
# ---------------------------------------
def get_facemesh_bbox(landmarks, img_shape):
    h, w, _ = img_shape

    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    return (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, w, h)


# ---------------------------------------
# 최종 얼굴 인식 + 얼굴 crop 반환 (Haar 대체)
# ---------------------------------------
def detect_face(image_path):
    img_path = Path(image_path)
    img = cv2.imread(str(img_path))

    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    # FaceMesh landmarks
    landmarks = get_facemesh_landmarks(img_path)
    x, y, w, h = get_facemesh_bbox(landmarks, img.shape)

    # crop
    face_crop = img[y:y + h, x:x + w]

    return img, face_crop, (x, y, w, h)
