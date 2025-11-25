import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def rgb_to_lab(rgb):
    """RGB 배열([R,G,B])을 Lab으로 변환"""
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0,0]
    return lab

def process_palette(image_path, num_colors=24):
    """도넛형 팔레트 이미지에서 색상 추출 후 DataFrame 반환"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"팔레트 파일을 찾을 수 없음: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    cx, cy = w // 2, h // 2
    outer_r = min(cx, cy) * 0.95
    inner_r = outer_r * 0.65

    Y, X = np.indices((h, w))
    dx = X - cx
    dy = cy - Y
    dist = np.sqrt(dx**2 + dy**2)
    mask_donut = (dist <= outer_r) & (dist >= inner_r)

    angle = np.arctan2(dy, dx)
    angle[angle < 0] += 2 * np.pi

    delta = 2 * np.pi / num_colors
    avg_rgbs, lab_vals = [], []

    for i in range(num_colors):
        wedge_mask = mask_donut & (angle >= i*delta) & (angle < (i+1)*delta)
        pixels = img_rgb[wedge_mask]
        avg = pixels.mean(axis=0).astype(int) if pixels.size > 0 else np.array([0,0,0], dtype=int)
        avg_rgbs.append(avg)
        lab_vals.append(rgb_to_lab(avg))

    df = pd.DataFrame({
        "번호": range(1, num_colors+1),
        "R": [c[0] for c in avg_rgbs],
        "G": [c[1] for c in avg_rgbs],
        "B": [c[2] for c in avg_rgbs],
        "L*": [l[0] for l in lab_vals],
        "a*": [l[1] for l in lab_vals],
        "b*": [l[2] for l in lab_vals],
    })
    return df

def load_all_palettes(palette_dir):
    """폴더에서 4계절 팔레트 불러오기"""
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
