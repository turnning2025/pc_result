# face_mesh_utils.py
import mediapipe as mp

def init_face_detection(model_selection=1, min_detection_confidence=0.5):
    """Mediapipe 얼굴 검출 초기화"""
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence
    )

def init_face_mesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
):
    """Mediapipe FaceMesh 초기화"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence
    )
