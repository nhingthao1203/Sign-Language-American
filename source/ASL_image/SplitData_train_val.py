import os
import random
import shutil
from math import floor

# Đường dẫn đến thư mục chứa các thư mục con của AtoZ
data_dir = r"D:\SLA\archive (1)\Train_Alphabet"

# Tạo các thư mục đầu ra cho train và val
output_dirs = ['train', 'val']
for output_dir in output_dirs:
    os.makedirs(os.path.join(data_dir, output_dir), exist_ok=True)

# Tỷ lệ phân chia
train_ratio = 0.9
val_ratio = 0.1

# Lấy danh sách các thư mục con (tương ứng với các ký tự A, B, C,... và blank)
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d not in output_dirs]

# Duyệt qua từng thư mục con (A, B, C,...)
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)

    # Lấy danh sách tất cả các file trong thư mục hiện tại
    files = os.listdir(class_dir)

    # Shuffle các file để việc phân chia là ngẫu nhiên
    random.shuffle(files)

    # Chia tập tin theo tỷ lệ 90:10
    num_files = len(files)
    train_end = floor(train_ratio * num_files)

    train_files = files[:train_end]
    val_files = files[train_end:]

    # Hàm để copy file đến thư mục đích
    def copy_files(file_list, split):
        split_dir = os.path.join(data_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for file_name in file_list:
            src = os.path.join(class_dir, file_name)
            dest = os.path.join(split_dir, file_name)
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, dest)
                except Exception as e:
                    print(f"Không thể sao chép {file_name}: {e}")

    # Copy các tập tin vào các thư mục tương ứng
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

print("Hoàn thành việc phân chia dữ liệu.")
