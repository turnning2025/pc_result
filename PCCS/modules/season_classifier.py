import numpy as np
from .skin_extractor import SkinNotFoundError

def classify_season(skin_lab, palettes):
    """피부 Lab 색상과 계절별 팔레트 비교 → 가장 가까운 계절 반환"""
    if skin_lab is None:
        raise SkinNotFoundError("피부 색상을 찾지 못했습니다.")

    distances = {}
    for season, df in palettes.items():
        pal_lab = df[["L*", "a*", "b*"]].values
        dist = np.mean(np.linalg.norm(pal_lab - skin_lab, axis=1))
        distances[season] = dist

    return min(distances, key=distances.get)
