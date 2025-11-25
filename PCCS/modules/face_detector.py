import cv2
import mediapipe as mp
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh

class FaceNotFoundError(Exception):
    """얼굴을 찾지 못했을 때 발생하는 예외"""
    pass

def detect_face(image_path):
    """Haar Cascade로 얼굴 검출"""
    img_path = Path(image_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        raise FaceNotFoundError(f"이미지에서 얼굴을 찾을 수 없음: {image_path}")

    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    return img, face_crop, (x, y, w, h)

def visualize_facemesh(image_path, save_path=None):
    """FaceMesh 랜드마크 시각화"""
    img_path = Path(image_path)
    img = cv2.imread(str(img_path))
    if img is None:
        print("이미지를 찾을 수 없음")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as mesh:
        result = mesh.process(img_rgb)
        if not result.multi_face_landmarks:
            print("FaceMesh 감지 실패")
            return

        h, w, _ = img.shape
        for lm in result.multi_face_landmarks[0].landmark:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)

    save_path = Path(save_path) if save_path else img_path.parent / "face_mesh_output.jpg"
    cv2.imwrite(str(save_path), img)
    print(f"FaceMesh 시각화 저장 완료 → {save_path}")
