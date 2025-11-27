import numpy as np
import matplotlib.pyplot as plt

def visualize_skin_position(palettes, skin_lab, save_path="skin_position.jpg"):
    """
    피부 Lab 값을 시즌 팔레트 위에 시각화 + 시즌 분포도(KNN 기반 퍼센트) 출력
    """

    plt.figure(figsize=(8, 8))

    season_colors = {
        "spring": "#FFB347",
        "summer": "#7EC8E3",
        "autumn": "#C97F3D",
        "winter": "#6A5ACD",
    }


    # 퍼센트
    all_rows = []
    for season, df in palettes.items():
        for _, row in df.iterrows():
            dist = np.linalg.norm(row[["L*", "a*", "b*"]].values - skin_lab)
            all_rows.append((season, dist))

    # 거리 정렬
    all_rows = sorted(all_rows, key=lambda x: x[1])

    K = 50
    top_k = all_rows[:K]

    # 시즌별 개수 계산
    from collections import Counter
    cnt = Counter([s for s, _ in top_k])

    # 퍼센트 변환
    knn_percent = {s: round((cnt.get(s, 0) / K) * 100, 2) for s in palettes.keys()}

    print("\n===== 시즌 색상 분포도 (KNN 기반 퍼센트) =====")
    for s, p in knn_percent.items():
        print(f"{s:7s}: {p:5.2f}%")
    print("=============================================\n")

    # 시각화
    for season, df in palettes.items():
        plt.scatter(
            df["a*"], df["L*"],
            s=40,
            alpha=0.6,
            label=season,   # 퍼센트 제거
            c=season_colors.get(season, "gray")
        )

    plt.scatter(
        skin_lab[1], skin_lab[0],
        s=250,
        c="red",
        edgecolors="black",
        marker="X",
        label="SKIN"
    )

    plt.title("Skin Lab Position inside Season Palettes (L vs a)", fontsize=13)
    plt.xlabel("a* (녹색  ← 0 →  빨강)", fontsize=11)
    plt.ylabel("L* (명도)", fontsize=11)

    plt.xlim(-60, 60)
    plt.ylim(100, 0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # 이미지 저장
    plt.savefig(save_path, dpi=250)
    plt.close()

    print(f"피부 Lab 위치 시각화 저장 완료 → {save_path}")
