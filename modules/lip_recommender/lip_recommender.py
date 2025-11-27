import numpy as np
import pandas as pd
from pathlib import Path

# palette_processor에서 PNG → DF 변환 함수 읽기
from modules.palette_processor import process_palette


# ================================================================
# 0) 기본 경로 설정
# ================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
PALETTE_DIR = BASE_DIR / "palettes"


# ================================================================
# 1) ΔE 계산 함수 (CIE76)
# ================================================================
def delta_e(lab1, lab2):
    lab1 = np.array(lab1)
    lab2 = np.array(lab2)
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


# ================================================================
# 2) 시즌 팔레트 로딩 (PNG → DataFrame)
# ================================================================
def load_season_palette(season):
    palette_path = PALETTE_DIR / f"{season}_numbered.png"
    if not palette_path.exists():
        raise FileNotFoundError(f"시즌 팔레트 파일 없음: {palette_path}")

    df = process_palette(palette_path)
    df = df.rename(columns={"L*": "L", "a*": "a", "b*": "b"})
    return df


# ================================================================
# 3) 립 색상 → 시즌 자동 분류 (LAB-KNN 버전)
# ================================================================
def classify_lip_season_knn(lip_lab, season_classifier):
    """
    lip_lab: (L, a, b)
    season_classifier: SeasonKNNClassifier (LAB-KNN 버전)
    """
    return season_classifier.predict_season(lip_lab)


# ================================================================
# 4) 사용자 피부색 기준 ΔE 정렬
# ================================================================
def sort_by_lab_distance(lip_df, target_lab):
    lip_df = lip_df.copy()
    lip_df["delta_e"] = lip_df.apply(
        lambda r: delta_e(target_lab, (r["L"], r["a"], r["b"])),
        axis=1
    )
    return lip_df.sort_values("delta_e")


# ================================================================
# 5) ΔE < 2 중복 제거 (최대 5개 출력)
# ================================================================
def remove_duplicates(lip_df, threshold=2.0, max_count=5):
    final = []
    used = []

    for _, row in lip_df.iterrows():
        current = np.array([row["L"], row["a"], row["b"]])

        duplicated = any(delta_e(current, u) < threshold for u in used)
        if duplicated:
            continue

        used.append(current)
        final.append(row)

        if len(final) >= max_count:
            break

    return pd.DataFrame(final)


# ================================================================
# 6) 최종 립 추천 (LAB-KNN 시즌 매칭 + ΔE 정렬 + 중복 제거)
# ================================================================
def recommend_lip_colors(season_classifier, user_season, skin_lab, lip_df):
    """
    season_classifier : SeasonKNNClassifier (LAB-KNN 버전)
    user_season       : 사용자 판정 시즌 (spring/summer/autumn/winter)
    skin_lab          : 사용자 피부 Lab (L, a, b)
    lip_df            : 전체 립 CSV (L,a,b 포함)
    """

    # ---------------------------------------------------
    # 1) 립 CSV 전체에 대해 시즌 라벨 자동 부착 (LAB-KNN)
    # ---------------------------------------------------
    assigned_seasons = []
    for _, row in lip_df.iterrows():
        lip_lab = (row["L"], row["a"], row["b"])
        lip_season = classify_lip_season_knn(lip_lab, season_classifier)
        assigned_seasons.append(lip_season)

    lip_df = lip_df.copy()
    lip_df["season_knn"] = assigned_seasons

    # ---------------------------------------------------
    # 2) 사용자 시즌과 일치하는 립만 사용
    # ---------------------------------------------------
    filtered = lip_df[lip_df["season_knn"] == user_season].copy()
    if filtered.empty:
        # 시즌 내 립이 하나도 없다면 전체에서 진행 (fallback)
        filtered = lip_df.copy()

    # ---------------------------------------------------
    # 3) 피부 Lab 기준 ΔE 거리 정렬
    # ---------------------------------------------------
    sorted_df = sort_by_lab_distance(filtered, skin_lab)

    # ---------------------------------------------------
    # 4) ΔE < 2 중복 제거, TOP 5
    # ---------------------------------------------------
    final = remove_duplicates(sorted_df, threshold=2.0, max_count=5)

    return final
