import cv2
import numpy as np
from math import ceil

def draw_palette(df, block_size=120):
    """
    DataFrame의 R, G, B 컬럼을 받아 팔레트 이미지 생성
    """
    if df.empty:
        return np.zeros((block_size, 0, 3), dtype=np.uint8)

    num_colors = len(df)
    palette_img = np.zeros((block_size, block_size * num_colors, 3), dtype=np.uint8)

    for i, row in enumerate(df.itertuples(index=False)):
        r, g, b = row.R, row.G, row.B
        x1, x2 = i * block_size, (i + 1) * block_size
        palette_img[:, x1:x2] = [b, g, r]

    return palette_img

def append_palette_to_face(face_image_path, palette_df, save_path="face_with_palette.jpg",
                            block_size=80, max_rows=2):
    """
    얼굴 사진 아래 팔레트를 합성
    - 팔레트 폭이 사진보다 좁으면 한 줄, 넓으면 max_rows 줄로 나눔
    """
    img = cv2.imread(str(face_image_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {face_image_path}")

    img_h, img_w = img.shape[:2]
    num_colors = len(palette_df)
    max_blocks_per_row = img_w // block_size

    if num_colors <= max_blocks_per_row:
        rows_needed = 1
        blocks_per_row = num_colors
    else:
        rows_needed = min(max_rows, ceil(num_colors / max_blocks_per_row))
        blocks_per_row = ceil(num_colors / rows_needed)

    palette_rows = []
    for r in range(rows_needed):
        start_idx = r * blocks_per_row
        end_idx = min((r + 1) * blocks_per_row, num_colors)
        sub_df = palette_df.iloc[start_idx:end_idx]
        if sub_df.empty:
            continue
        row_img = draw_palette(sub_df, block_size=block_size)

        # 폭 맞춤
        row_w = row_img.shape[1]
        if row_w < img_w:
            pad_left = (img_w - row_w) // 2
            pad_right = img_w - row_w - pad_left
            row_img = cv2.copyMakeBorder(
                row_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        elif row_w > img_w:
            start = (row_w - img_w) // 2
            row_img = row_img[:, start:start + img_w]

        palette_rows.append(row_img)

    if not palette_rows:
        raise ValueError("팔레트 생성 실패: 유효한 색 데이터가 없습니다.")

    palette_final = np.vstack(palette_rows)
    combined = np.vstack([img, palette_final])
    cv2.imwrite(save_path, combined)
    print(f"퍼스널컬러 비교 이미지 저장 완료 → {save_path}")
    return save_path
