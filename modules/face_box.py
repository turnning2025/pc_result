# modules/face_box.py
import cv2
from pathlib import Path
from modules.face_detector import get_facemesh_landmarks, get_facemesh_bbox, FaceNotFoundError

def save_face_box(img_path, bbox=None, save_path=None):
    """
    FaceMesh 기반 얼굴 박스를 그려 저장.
    bbox 인자를 직접 전달할 수 있고,
    없으면 내부에서 FaceMesh로 다시 계산한다.
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))

    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {img_path}")

    # bbox가 없으면 FaceMesh로 계산
    if bbox is None:
        landmarks = get_facemesh_landmarks(img_path)
        if landmarks is None:
            raise FaceNotFoundError("FaceMesh 랜드마크를 찾지 못함")
        bbox = get_facemesh_bbox(landmarks, img.shape)

    x, y, w, h = bbox

    # 얼굴 박스 그리기
    img_box = img.copy()
    cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 저장 경로 처리
    save_path = Path(save_path) if save_path else img_path.parent / "face_box.jpg"

    cv2.imwrite(str(save_path), img_box)
    print(f"얼굴 박스 이미지 저장 완료 → {save_path}")
