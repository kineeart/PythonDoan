import os
import glob
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'KeypointImages')
SEQ_LEN = 60
N_FEATURES = 201  # 67 landmarks * (x,y,z)
IMG_SIZE = (640, 640)

# Index mapping: 25 pose + 21 left hand + 21 right hand
N_UP = 25
N_HAND = 21

def split_landmarks(vec):
    # vec shape: (201,)
    pts = vec.reshape(-1, 3)  # (67,3)
    pose = pts[:N_UP]
    lhand = pts[N_UP:N_UP+N_HAND]
    rhand = pts[N_UP+N_HAND:N_UP+2*N_HAND]
    return pose, lhand, rhand

def to_canvas_xy(x, y, w, h):
    # Mediapipe coords normalized [0,1], y top->bottom
    return int(x * w), int(y * h)

def draw_points(img, points, color, radius=3, alpha=1.0):
    overlay = img.copy()
    h, w = img.shape[:2]
    for (x, y, _z) in points:
        cx, cy = to_canvas_xy(x, y, w, h)
        cv2.circle(overlay, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_sequence_composite(seq, out_path):
    # seq: (60, 201)
    canvas = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)
    # vẽ mờ theo thời gian (frame đầu nhạt, frame cuối đậm)
    for t in range(seq.shape[0]):
        pose, lhand, rhand = split_landmarks(seq[t])
        alpha = 0.2 + 0.8 * (t / (SEQ_LEN - 1))
        draw_points(canvas, pose,  (255, 255, 255), radius=2, alpha=alpha)  # trắng
        draw_points(canvas, lhand, (0, 255, 0),     radius=3, alpha=alpha)  # xanh lá
        draw_points(canvas, rhand, (0, 128, 255),   radius=3, alpha=alpha)  # cam
    cv2.imwrite(out_path, canvas)

def export_split(split):
    split_dir = os.path.join(DATA_DIR, split)
    npz_paths = sorted(glob.glob(os.path.join(split_dir, 'class_*', '*.npz')))
    if not npz_paths:
        print(f"No .npz found in {split_dir}")
        return
    out_split = os.path.join(OUT_DIR, split)
    os.makedirs(out_split, exist_ok=True)

    for i, npz_path in enumerate(npz_paths):
        data = np.load(npz_path)
        seq = data['sequence']
        lbl = int(data['label'])
        # reshape nếu cần
        if seq.ndim == 1 and seq.size == SEQ_LEN * N_FEATURES:
            seq = seq.reshape(SEQ_LEN, N_FEATURES)
        if seq.shape != (SEQ_LEN, N_FEATURES):
            # bỏ file sai kích thước
            continue
        base = os.path.splitext(os.path.basename(npz_path))[0]
        out_name = f"class_{lbl}_{base}_composite.png"
        out_path = os.path.join(out_split, out_name)
        draw_sequence_composite(seq, out_path)

    print(f"Exported {split}: {len(npz_paths)} sequences -> {out_split}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ['test']:
        export_split(split)
    print("Done.")

if __name__ == "__main__":
    main()