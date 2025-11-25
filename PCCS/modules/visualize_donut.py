import cv2
import numpy as np
from pathlib import Path

def append_face_to_donut(face_image_path, donut_image_path, save_path="face_in_donut.jpg", scale=1.0):
    """
    얼굴 이미지를 도넛 중앙 빈 공간에 배치하여 합성
    face_image_path: 얼굴 이미지 경로
    donut_image_path: 도넛 이미지 경로 (투명 배경 PNG 권장)
    scale: 도넛 이미지 전체 크기 조절 비율
    """
    # 얼굴 이미지 읽기
    face_img = cv2.imread(str(face_image_path))
    if face_img is None:
        raise FileNotFoundError(f"얼굴 이미지를 찾을 수 없음: {face_image_path}")

    # 도넛 이미지 읽기 (알파 채널 포함)
    donut_img = cv2.imread(str(donut_image_path), cv2.IMREAD_UNCHANGED)
    if donut_img is None:
        raise FileNotFoundError(f"도넛 이미지를 찾을 수 없음: {donut_image_path}")

    # 도넛 알파 채널 분리
    if donut_img.shape[2] == 4:
        alpha_donut = donut_img[:, :, 3] / 255.0
        donut_bgr = cv2.cvtColor(donut_img, cv2.COLOR_BGRA2BGR)
    else:
        alpha_donut = None
        donut_bgr = donut_img

    # 도넛 크기 조절
    h_d, w_d = donut_bgr.shape[:2]
    new_w = int(w_d * scale)
    new_h = int(h_d * scale)
    donut_resized = cv2.resize(donut_bgr, (new_w, new_h))
    if alpha_donut is not None:
        alpha_resized = cv2.resize(alpha_donut, (new_w, new_h))
    else:
        alpha_resized = None

    # 얼굴 이미지 도넛 중앙 구멍 크기에 맞게 조정
    # 도넛 구멍 크기는 도넛 세로 크기의 약 50% 정도로 가정
    hole_diameter = int(new_h * 0.5)
    face_resized = cv2.resize(face_img, (hole_diameter, hole_diameter))

    # 얼굴 합성 좌표
    y_offset = new_h // 2 - hole_diameter // 2
    x_offset = new_w // 2 - hole_diameter // 2

    # 합성
    combined = donut_resized.copy()
    face_h, face_w = face_resized.shape[:2]

    # 단순 오버레이 (alpha blending optional)
    combined[y_offset:y_offset+face_h, x_offset:x_offset+face_w] = face_resized

    # 저장
    cv2.imwrite(save_path, combined)
    print(f"얼굴 합성 도넛 이미지 저장 완료 → {save_path}")
    return save_path
