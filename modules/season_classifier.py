import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class SeasonKNNClassifier:
    def __init__(self, palettes, k=7):
        """ LAB 기반 KNN 시즌 분류기 """
        self.k = k
        self.palettes = palettes

        # LAB 데이터와 레이블 구성
        self.X_lab = []
        self.y = []

        for season, df in palettes.items():
            for _, row in df.iterrows():
                self.X_lab.append([row["L*"], row["a*"], row["b*"]])
                self.y.append(season)

        self.X_lab = np.array(self.X_lab, dtype=np.float32)
        self.y = np.array(self.y)

        # ΔE76 ≒ euclidean
        self.knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
            weights="distance"
        )
        self.knn.fit(self.X_lab, self.y)

    # ------------------------------
    # 시즌 예측
    # ------------------------------
    def predict_season(self, lab_input):
        lab_input = np.array(lab_input).reshape(1, -1)
        return self.knn.predict(lab_input)[0]

    # ------------------------------
    # KNN 표 기반 득표율
    # ------------------------------
    def get_knn_votes(self, lab_input):
        lab_input = np.array(lab_input).reshape(1, -1)
        distances, indices = self.knn.kneighbors(lab_input)

        neighbor_labels = self.y[indices[0]]
        from collections import Counter
        cnt = Counter(neighbor_labels)

        total = self.k
        season_percent = {
            s: round(cnt.get(s, 0) / total * 100, 2)
            for s in ["spring", "summer", "autumn", "winter"]
        }
        return season_percent

    # ------------------------------
    # 시즌별 거리 상세정보(avg/min/sum)
    # ------------------------------
    def get_knn_detail(self, lab_input):
        lab_input = np.array(lab_input).reshape(1, -1)
        distances, indices = self.knn.kneighbors(lab_input)

        neighbor_labels = self.y[indices[0]]
        neighbor_dist = distances[0]

        from collections import defaultdict
        info = defaultdict(lambda: {"votes":0, "sum":0, "min":999, "avg":0})

        for season, dist in zip(neighbor_labels, neighbor_dist):
            info[season]["votes"] += 1
            info[season]["sum"] += dist
            info[season]["min"] = min(info[season]["min"], dist)

        for season in info:
            info[season]["avg"] = info[season]["sum"] / info[season]["votes"]

        return info
