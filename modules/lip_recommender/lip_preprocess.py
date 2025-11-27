from pathlib import Path
import pandas as pd

def load_and_preprocess_lip_csv(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일이 존재하지 않음: {csv_path}")

    df = pd.read_csv(csv_path)

    numeric_cols = ["r", "g", "b"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    return df
