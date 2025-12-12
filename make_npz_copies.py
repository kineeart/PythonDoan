import os
import glob
import numpy as np
from tqdm import tqdm
import random

# Cấu hình
DATA_DIR = 'Data'
SPLITS = ['train', 'valid', 'test']
SEQ_LEN = 60
N_FEATURES = 201
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def augment_keypoints(sequence, label, augment_id):
    """
    Áp dụng augmentation lên chuỗi keypoints (60, 201)
    """
    seq = sequence.copy()
    
    if augment_id == 0:
        # Aug 0: Gaussian Noise (nhiễu nhẹ)
        noise = np.random.normal(0, 0.015, seq.shape).astype(np.float32)
        seq = seq + noise
        
    elif augment_id == 1:
        # Aug 1: Scale ngẫu nhiên (phóng to/thu nhỏ)
        scale = np.random.uniform(0.85, 1.15)
        seq = seq * scale
        
    elif augment_id == 2:
        # Aug 2: Gaussian Noise + Scale kết hợp
        noise = np.random.normal(0, 0.02, seq.shape).astype(np.float32)
        scale = np.random.uniform(0.9, 1.1)
        seq = (seq + noise) * scale
        
    elif augment_id == 3:
        # Aug 3: Horizontal Flip - Đổi tay trái/phải
        # 21 điểm tay trái (index 0-62) <-> 21 điểm tay phải (index 63-125)
        left_hand = seq[:, 0:63].copy()   # 21 * 3 = 63
        right_hand = seq[:, 63:126].copy()  # 21 * 3 = 63
        
        # Swap và flip x-coordinate (0::3 là tọa độ x)
        seq[:, 0:63] = right_hand
        seq[:, 63:126] = left_hand
        seq[:, 0::3] = 1.0 - seq[:, 0::3]  # Flip x coordinates
        
    elif augment_id == 4:
        # Aug 4: Time Warping - Nén/giãn thời gian
        # Chọn ngẫu nhiên 40-80 frames rồi interpolate về 60
        n_frames = np.random.randint(40, 80)
        indices = np.linspace(0, SEQ_LEN-1, n_frames)
        original_indices = np.arange(SEQ_LEN)
        
        warped_seq = np.zeros_like(seq)
        for feat_idx in range(N_FEATURES):
            values = seq[:, feat_idx]
            warped_values = np.interp(original_indices, indices, 
                                     np.interp(indices, original_indices, values))
            warped_seq[:, feat_idx] = warped_values
        seq = warped_seq.astype(np.float32)
    
    # Clip để đảm bảo giá trị hợp lệ
    seq = np.clip(seq, -2.0, 2.0)
    
    return seq, label

def augment_split(split, target_multiplier=2.5):
    """
    Augment dữ liệu trong một split
    target_multiplier: nhân số lượng file lên bao nhiêu lần
    """
    split_dir = os.path.join(DATA_DIR, split)
    
    # Lấy tất cả file .npz
    npz_files = sorted(glob.glob(os.path.join(split_dir, 'class_*', '*.npz')))
    
    if not npz_files:
        print(f"[{split}] Không tìm thấy file .npz")
        return
    
    print(f"\n[{split}] Tìm thấy {len(npz_files)} files gốc")
    
    # Tính số augmentation cần tạo
    n_augment = int(len(npz_files) * (target_multiplier - 1))
    
    print(f"[{split}] Sẽ tạo thêm {n_augment} files augmented")
    
    # Random chọn files để augment
    files_to_augment = random.choices(npz_files, k=n_augment)
    
    created_count = 0
    
    for npz_path in tqdm(files_to_augment, desc=f'Augmenting {split}'):
        try:
            # Load dữ liệu gốc
            data = np.load(npz_path)
            sequence = data['sequence']
            label = int(data['label'])
            
            # Đảm bảo shape đúng
            if sequence.ndim == 1:
                sequence = sequence.reshape(SEQ_LEN, N_FEATURES)
            
            if sequence.shape != (SEQ_LEN, N_FEATURES):
                continue
            
            # Random chọn augmentation method (0-4)
            aug_id = random.randint(0, 4)
            
            # Apply augmentation
            aug_sequence, aug_label = augment_keypoints(sequence, label, aug_id)
            
            # Tạo tên file mới
            class_dir = os.path.dirname(npz_path)
            base_name = os.path.splitext(os.path.basename(npz_path))[0]
            
            # Tìm số thứ tự chưa dùng
            counter = 0
            while True:
                new_name = f"{base_name}_aug{aug_id}_{counter}.npz"
                new_path = os.path.join(class_dir, new_name)
                if not os.path.exists(new_path):
                    break
                counter += 1
            
            # Lưu file augmented
            np.savez(new_path, sequence=aug_sequence, label=aug_label)
            created_count += 1
            
        except Exception as e:
            print(f"Lỗi khi xử lý {npz_path}: {e}")
            continue
    
    print(f"[{split}] Đã tạo thành công {created_count} files")
    
    # Kiểm tra lại số lượng
    new_count = len(glob.glob(os.path.join(split_dir, 'class_*', '*.npz')))
    print(f"[{split}] Tổng số files sau augment: {new_count}")

def main():
    print("="*60)
    print("AUGMENTATION DỮ LIỆU .NPZ - TĂNG SỐ LƯỢNG LÊN 4000+")
    print("="*60)
    
    # Kiểm tra số lượng hiện tại
    print("\nSố lượng file .npz TRƯỚC augmentation:")
    total_before = 0
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        count = len(glob.glob(os.path.join(split_dir, 'class_*', '*.npz')))
        print(f"  {split}: {count} files")
        total_before += count
    print(f"  TỔNG: {total_before} files")
    
    # Tính target multiplier để đạt ~4000 files
    target_total = 6000
    multiplier = target_total / total_before if total_before > 0 else 2.5
    
    print(f"\nTarget: {target_total} files")
    print(f"Multiplier: {multiplier:.2f}x")
    
    # Augment từng split
    for split in SPLITS:
        augment_split(split, target_multiplier=multiplier)
    
    # Kiểm tra lại tổng số
    print("\n" + "="*60)
    print("Số lượng file .npz SAU augmentation:")
    total_after = 0
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        count = len(glob.glob(os.path.join(split_dir, 'class_*', '*.npz')))
        print(f"  {split}: {count} files")
        total_after += count
    print(f"  TỔNG: {total_after} files")
    print("="*60)
    
    if total_after >= 4000:
        print(f"\n✓ ĐẠT YÊU CẦU: {total_after} files >= 4000")
    else:
        print(f"\n✗ Còn thiếu: {4000 - total_after} files")
    
    print("\nHoàn thành! Các augmentation được áp dụng:")
    print("  0: Gaussian Noise (nhiễu nhẹ)")
    print("  1: Random Scale (phóng to/thu nhỏ)")
    print("  2: Noise + Scale kết hợp")
    print("  3: Horizontal Flip (đổi tay trái/phải)")
    print("  4: Time Warping (nén/giãn thời gian)")

if __name__ == '__main__':
    main()
