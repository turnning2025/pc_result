# face_visualize.py
import cv2
from pathlib import Path
from .face_mesh_utils import init_face_mesh

class FaceNotFoundError(Exception):
    """FaceMesh 랜드마크를 찾지 못했을 때 발생"""
    pass

def visualize_facemesh(image_path, save_path=None):
    """FaceMesh 랜드마크를 이미지에 표시하고 저장"""
    img_path = Path(image_path)
    save_path = Path(save_path) if save_path else img_path.parent / "face_mesh_result.jpg"

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mesh = init_face_mesh()
    result = mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        raise FaceNotFoundError(f"FaceMesh 랜드마크를 찾을 수 없음: {image_path}")

    landmarks = result.multi_face_landmarks[0]
    h, w, _ = img.shape

    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    cv2.imwrite(str(save_path), img)
    print(f"FaceMesh 시각화 이미지 저장됨 → {save_path}")
    return str(save_path)
