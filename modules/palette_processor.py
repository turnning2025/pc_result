import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import color


def rgb_to_lab(rgb):
    """
    RGB 배열([R,G,B], 0~255)을 정확한 Lab으로 변환
    skimage.rgb2lab은 반드시 0~1 float 입력이어야 함
    """
    rgb_norm = np.array(rgb) / 255.0
    lab = color.rgb2lab([[rgb_norm]])[0][0]
    return lab


def process_palette(image_path, num_colors=24):
    """
    도넛형 팔레트 이미지에서 24개 색상 추출하여 DataFrame 반환.
    (번호 텍스트 포함되어 있어도 wedge 평균은 충분히 안정적임)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"팔레트 파일을 찾을 수 없음: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # 중심점
    cx, cy = w // 2, h // 2

    # 도넛 반경
    outer_r = min(cx, cy) * 0.95
    inner_r = outer_r * 0.65

    # 각 픽셀의 각도/거리 계산
    Y, X = np.indices((h, w))
    dx = X - cx
    dy = cy - Y
    dist = np.sqrt(dx**2 + dy**2)
    mask_donut = (dist <= outer_r) & (dist >= inner_r)

    angle = np.arctan2(dy, dx)
    angle[angle < 0] += 2 * np.pi

    # 24 등분
    delta = 2 * np.pi / num_colors

    rgb_list, lab_list = [], []

    for i in range(num_colors):
        wedge_mask = mask_donut & (angle >= i * delta) & (angle < (i + 1) * delta)
        pixels = img_rgb[wedge_mask]

        if pixels.size == 0:
            avg = np.array([0, 0, 0], dtype=np.uint8)
        else:
            avg = pixels.mean(axis=0).astype(int)

        rgb_list.append(avg)

        # 핵심 수정: 반드시 0~1 정규화 후 lab 변환
        lab = rgb_to_lab(avg)
        lab_list.append(lab)

    df = pd.DataFrame({
        "번호": range(1, num_colors + 1),
        "R": [c[0] for c in rgb_list],
        "G": [c[1] for c in rgb_list],
        "B": [c[2] for c in rgb_list],
        "L*": [l[0] for l in lab_list],
        "a*": [l[1] for l in lab_list],
        "b*": [l[2] for l in lab_list],
    })

    return df


def load_all_palettes(palette_dir):
    """
    palette_dir 안에서 spring/summer/autumn/winter 팔레트 로딩
    """
    palette_dir = Path(palette_dir)

    season_files = {
        "spring": palette_dir / "spring_numbered.png",
        "summer": palette_dir / "summer_numbered.png",
        "autumn": palette_dir / "autumn_numbered.png",
        "winter": palette_dir / "winter_numbered.png",
    }

    palettes = {}
    for season, path in season_files.items():
        df = process_palette(path)
        palettes[season] = df

    return palettes
