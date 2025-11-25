# face_box.py
import cv2
from pathlib import Path
from .face_detector import detect_face, FaceNotFoundError

def save_face_box(image_path, save_path=None):
    """얼굴 위치에 박스를 그려 이미지 저장"""
    img_path = Path(image_path)
    save_path = Path(save_path) if save_path else img_path.parent / "face_box.jpg"

    try:
        img, _, bbox = detect_face(img_path)
    except FaceNotFoundError:
        print("얼굴 감지 실패")
        return None

    x, y, w, h = bbox
    boxed = img.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(str(save_path), boxed)
    return str(save_path)
