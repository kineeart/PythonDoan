import os
import glob
import shutil
import random
from collections import defaultdict

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, 'Data')
SPLITS = ['train', 'valid', 'test']
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def list_class_files(split_dir):
    class_files = defaultdict(list)
    for class_dir in sorted(glob.glob(os.path.join(split_dir, 'class_*'))):
        cls = int(os.path.basename(class_dir).split('_')[1])
        files = sorted(glob.glob(os.path.join(class_dir, '*.npz')))
        class_files[cls] = files
    return class_files

def oversample_split(split):
    split_dir = os.path.join(DATA_DIR, split)
    class_files = list_class_files(split_dir)
    if not class_files:
        print(f"[{split}] No data found.")
        return

    # Tính số lượng lớn nhất
    max_count = max(len(files) for files in class_files.values())
    print(f"[{split}] Target count per class: {max_count}")

    for cls, files in class_files.items():
        count = len(files)
        if count == 0:
            print(f"  class_{cls}: 0 -> skip (không có mẫu để oversample)")
            continue
        if count >= max_count:
            print(f"  class_{cls}: {count} -> OK")
            continue

        need = max_count - count
        print(f"  class_{cls}: {count} -> oversample +{need}")
        # Lặp ngẫu nhiên các file hiện có để nhân bản
        out_dir = os.path.join(split_dir, f'class_{cls}')
        for i in range(need):
            src = random.choice(files)
            base = os.path.splitext(os.path.basename(src))[0]
            new_name = f"{base}_dup{i}.npz"
            dst = os.path.join(out_dir, new_name)
            shutil.copy2(src, dst)

def main():
    for split in SPLITS:
        oversample_split(split)
    print("Balancing done.")

if __name__ == "__main__":
    main()