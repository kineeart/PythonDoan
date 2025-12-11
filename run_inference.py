import os
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json

# --- Cấu hình ---
ROOT = os.path.dirname(__file__)
# Ưu tiên load model final, nếu không có thì load checkpoint
MODEL_PATH = os.path.join(ROOT, 'Models', 'sequence_cnn_lstm_final.keras')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT, 'Models', 'checkpoints', 'best_model.keras')

SEQ_LEN = 60
N_FEATURES = 201 # Khớp với lúc train (126 features tay + pad)
CONFIDENCE_THRESHOLD = 0.7 # Chỉ nhận diện khi độ tin cậy > 70%

# Load Label Map
LABEL_MAP_PATH = os.path.join(ROOT, 'Logs', 'label_map.json')
try:
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map_dict = json.load(f)
    # Đảo ngược dict: {0: "A", 1: "B", ...}
    LABEL_MAP = {v: k for k, v in label_map_dict.items()}
    print(f"Loaded labels: {LABEL_MAP}")
except Exception as e:
    print(f"Warning: Could not load label_map.json. Using default. Error: {e}")
    LABEL_MAP = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"G",6:"H",7:"I",8:"L"}

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

def draw_hand_skeleton(frame_bgr, results):
    """Vẽ xương tay lên hình để debug"""
    if results is None:
        return frame_bgr
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    def draw_one_hand(landmarks, color):
        if not landmarks: return
        # Vẽ điểm
        for lm in landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)
        # Vẽ nối
        for a, b in HAND_CONNECTIONS:
            la = landmarks.landmark[a]
            lb = landmarks.landmark[b]
            ax, ay = int(la.x * w), int(la.y * h)
            bx, by = int(lb.x * w), int(lb.y * h)
            cv2.line(img, (ax, ay), (bx, by), color, 2, lineType=cv2.LINE_AA)

    if results.left_hand_landmarks:
        draw_one_hand(results.left_hand_landmarks, (0, 255, 0)) # Trái: Xanh
    if results.right_hand_landmarks:
        draw_one_hand(results.right_hand_landmarks, (0, 165, 255)) # Phải: Cam
    return img

def extract_keypoints(results):
    """
    Trích xuất keypoint CHỈ TAY (42 điểm) và pad về 201 features
    để khớp với logic trong create_data_augment.py
    """
    # 21 điểm tay trái + 21 điểm tay phải
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            left[i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            right[i] = [lm.x, lm.y, lm.z]
    
    # Gộp lại thành (126,)
    kps = np.concatenate([left, right], axis=0).flatten()
    
    # Pad thêm số 0 để đủ 201 features (nếu model yêu cầu 201)
    if kps.size < N_FEATURES:
        kps = np.pad(kps, (0, N_FEATURES - kps.size), mode='constant', constant_values=0.0)
    
    return kps

def preprocess_frame(frame, holistic):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = holistic.process(img_rgb)
    img_rgb.flags.writeable = True
    
    kps = extract_keypoints(results)
    vis_frame = draw_hand_skeleton(frame, results)
    return kps, vis_frame

def predict_sequence(model, seq):
    # seq shape: (60, 201)
    # Thêm batch dimension -> (1, 60, 201)
    logits = model.predict(seq[None, ...], verbose=0)[0]
    cls_idx = int(np.argmax(logits))
    prob = float(np.max(logits))
    return cls_idx, prob

def run_webcam(cam_index=0):
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Tìm layer Normalization để adapt nếu cần (thường model đã lưu trạng thái này)
    norm_layer = None
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Normalization):
            norm_layer = l
            break
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    buffer = [] # Chứa 60 frame gần nhất
    last_text = ""
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. Xử lý frame
            kps, vis_frame = preprocess_frame(frame, holistic)
            
            # 2. Cập nhật buffer
            buffer.append(kps)
            if len(buffer) > SEQ_LEN:
                buffer.pop(0)

            # 3. Dự đoán khi đủ dữ liệu
            if len(buffer) == SEQ_LEN:
                seq = np.array(buffer, dtype=np.float32) # (60, 201)
                
                # Nếu model có layer Normalization chưa được build trong load_model (hiếm gặp), 
                # ta có thể gọi thủ công, nhưng thường model.predict tự lo.
                
                cls_idx, prob = predict_sequence(model, seq)
                
                if prob > CONFIDENCE_THRESHOLD:
                    label_name = LABEL_MAP.get(cls_idx, "Unknown")
                    last_text = f"{label_name} ({prob:.0%})"
                else:
                    last_text = "..."

            # 4. Hiển thị
            # Vẽ nền đen cho chữ dễ đọc
            cv2.rectangle(vis_frame, (0,0), (300, 60), (0,0,0), -1)
            cv2.putText(vis_frame, last_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            cv2.imshow("VSL Recognition", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC để thoát
                break

    cap.release()
    cv2.destroyAllWindows()

def run_video(video_path):
    if not os.path.exists(video_path):
        print(f"Không tìm thấy video: {video_path}")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    norm_layer = None
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Normalization):
            norm_layer = l
            break

    cap = cv2.VideoCapture(video_path)
    buffer = []
    last_pred = None
    last_prob = 0.0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kps, vis = preprocess_frame(frame, holistic)
            buffer.append(kps.astype(np.float32))
            if len(buffer) > SEQ_LEN:
                buffer.pop(0)
            if len(buffer) == SEQ_LEN:
                seq = np.stack(buffer, axis=0)
                if norm_layer is not None:
                    seq = norm_layer(seq[None, ...]).numpy()[0]
                cls, prob = predict_sequence(model, seq)
                last_pred, last_prob = cls, prob
            if last_pred is not None:
                text = f"{LABEL_MAP[last_pred]} ({last_prob:.2f})"
                cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("VSL Inference", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Mặc định chạy webcam. Đổi sang run_video(r"path\to\video.mp4") nếu muốn.
    run_webcam(0)