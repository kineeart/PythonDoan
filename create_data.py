import os
import glob
import json
import random
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import interp1d

# ========= Cấu hình nguồn dữ liệu =========
DATAIMAGE_ROOT = os.path.join('DataImage')
SPLITS = ['train', 'valid', 'test']

# Danh sách lớp khớp với data.yaml (nc: 9)
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'L']

# ========= Cấu hình trích xuất =========
DRAW_SKELETON = True  # lưu ảnh có vẽ xương tay để kiểm tra
SEQUENCE_LENGTH = 60  # số frame mỗi chuỗi
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_HAND_LANDMARKS * 2  # 2 bàn tay
FEATURES_PER_FRAME = N_TOTAL_LANDMARKS * 3  # 42 * 3 = 126; sẽ pad lên 201

# ========= Thư mục đầu ra =========
DATA_PATH = os.path.join('Data')  # Data/{train,valid,test}/class_{id}/*.npz
LOG_PATH  = os.path.join('Logs')
PREVIEW_DIR = os.path.join(LOG_PATH, 'KeypointPreview')

os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# ========= MediaPipe =========
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# ========= Tiện ích =========
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return image, results

def extract_keypoints(results):
    # chỉ lấy 2 bàn tay
    left = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)
    right = np.zeros((N_HAND_LANDMARKS, 3), dtype=np.float32)
    if results and results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:N_HAND_LANDMARKS]):
            left[i] = [lm.x, lm.y, lm.z]
    if results and results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:N_HAND_LANDMARKS]):
            right[i] = [lm.x, lm.y, lm.z]
    kps = np.concatenate([left, right], axis=0).flatten()  # (126,)
    # pad lên 201 để khớp với model hiện tại
    if kps.size < 201:
        kps = np.pad(kps, (0, 201 - kps.size), mode='constant', constant_values=0.0)
    return kps  # (201,)

def interpolate_keypoints(keypoints_sequence, target_len=SEQUENCE_LENGTH):
    if len(keypoints_sequence) == 0:
        return None
    if len(keypoints_sequence) == 1:
        frame = keypoints_sequence[0]
        return np.repeat(frame[None, :], target_len, axis=0)

    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = keypoints_sequence[0].shape[0]
    interpolated = np.zeros((target_len, num_features), dtype=np.float32)

    for f_idx in range(num_features):
        values = [frame[f_idx] for frame in keypoints_sequence]
        interpolator = interp1d(
            original_times, values,
            kind='cubic', bounds_error=False, fill_value="extrapolate"
        )
        interpolated[:, f_idx] = interpolator(target_times).astype(np.float32)
    return interpolated

def draw_hand_connections(image_bgr, results):
    if results is None:
        return image_bgr
    img = image_bgr.copy()
    h, w = img.shape[:2]

    def draw_one_hand(landmarks, color):
        if not landmarks:
            return
        for lm in landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)
        for a, b in HAND_CONNECTIONS:
            la = landmarks.landmark[a]
            lb = landmarks.landmark[b]
            ax, ay = int(la.x * w), int(la.y * h)
            bx, by = int(lb.x * w), int(lb.y * h)
            cv2.line(img, (ax, ay), (bx, by), color, 2, lineType=cv2.LINE_AA)

    if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
        draw_one_hand(results.left_hand_landmarks, (0, 255, 0))
    if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
        draw_one_hand(results.right_hand_landmarks, (0, 165, 255))
    return img

def sequence_from_image(image_path, holistic, preview_save_path=None):
    frame = cv2.imread(image_path)
    if frame is None:
        return []
    _, results = mediapipe_detection(frame, holistic)
    kps = extract_keypoints(results)

    if DRAW_SKELETON and preview_save_path:
        debug_img = draw_hand_connections(frame, results)
        os.makedirs(os.path.dirname(preview_save_path), exist_ok=True)
        cv2.imwrite(preview_save_path, debug_img)

    if kps is None or (isinstance(kps, np.ndarray) and kps.size == 0):
        return []
    return [kps]

# ========= Đọc Label YOLO =========
def read_yolo_class(label_path):
    """Đọc class_id từ dòng đầu tiên của file label .txt"""
    if not os.path.exists(label_path):
        return None
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            if not line:
                return None
            parts = line.split()
            return int(parts[0]) # Lấy số đầu tiên (class_id)
    except Exception:
        return None

# ========= Main Pipeline =========
def main():
    print(f"{datetime.now()} Start processing data...")
    
    # Tạo label_map.json từ danh sách cứng
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    with open(os.path.join(LOG_PATH, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Classes: {label_map}")

    total_images = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for split in SPLITS:
            images_dir = os.path.join(DATAIMAGE_ROOT, split, 'images')
            labels_dir = os.path.join(DATAIMAGE_ROOT, split, 'labels')
            
            if not os.path.isdir(images_dir):
                print(f"Warning: {images_dir} not found. Skipping {split}.")
                continue

            # Lấy danh sách ảnh
            image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
                image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
            
            label_counters = {} # Đếm số file cho mỗi class trong split này

            for image_path in tqdm(image_paths, desc=f'Processing {split}'):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                label_path = os.path.join(labels_dir, base_name + '.txt')
                
                class_id = read_yolo_class(label_path)
                if class_id is None:
                    # Không có label hoặc file lỗi -> bỏ qua
                    continue
                
                # Kiểm tra class_id hợp lệ
                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    continue

                # Trích xuất keypoint
                preview_path = os.path.join(PREVIEW_DIR, split, f'{base_name}_skel.jpg') if DRAW_SKELETON else None
                frames = sequence_from_image(image_path, holistic, preview_save_path=preview_path)
                
                if not frames:
                    continue

                # Xử lý chuỗi (nhân bản -> nội suy)
                base_seq = frames * 8
                seq = interpolate_keypoints(base_seq, target_len=SEQUENCE_LENGTH)
                if seq is None:
                    continue

                # Lưu file .npz
                save_root = os.path.join(DATA_PATH, split, f'class_{class_id}')
                os.makedirs(save_root, exist_ok=True)
                
                idx = label_counters.get(class_id, 0)
                out_path = os.path.join(save_root, f'{idx}.npz')
                np.savez(out_path, sequence=seq, label=class_id)
                
                label_counters[class_id] = idx + 1
                total_images += 1

    print(f"{'-'*50}\nDATA PROCESSING COMPLETED.")
    print(f"Total processed images: {total_images}")

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main()