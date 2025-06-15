import os
import random
import shutil
from math import floor

# Đường dẫn đến thư mục chứa các thư mục con của AtoZ
data_dir = r"E:\MSASL-valid-dataset-downloader\dataset\dataset"

# Tạo các thư mục đầu ra cho train, test, và val
output_dirs = ['train', 'test', 'val']
for output_dir in output_dirs:
    os.makedirs(os.path.join(data_dir, output_dir), exist_ok=True)

# Tỷ lệ phân chia
train_ratio = 0.6
test_ratio = 0.2
val_ratio = 0.2

# Lấy danh sách các thư mục con (tương ứng với các ký tự A, B, C,... và blank)
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Duyệt qua từng thư mục con (A, B, C,...)
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)

    # Bỏ qua nếu đây là thư mục train, test, hoặc val
    if class_name in output_dirs:
        continue

    # Lấy danh sách tất cả các file trong thư mục hiện tại
    files = os.listdir(class_dir)

    # Shuffle các file để việc phân chia là ngẫu nhiên
    random.shuffle(files)

    # Chia tập tin theo tỷ lệ 60:20:20
    num_files = len(files)
    train_end = floor(train_ratio * num_files)
    test_end = train_end + floor(test_ratio * num_files)

    train_files = files[:train_end]
    test_files = files[train_end:test_end]
    val_files = files[test_end:]


    # Hàm để copy file đến thư mục đích
    def copy_files(file_list, split):
        split_dir = os.path.join(data_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)  # Tạo thư mục đích nếu chưa tồn tại
        for file_name in file_list:
            src = os.path.join(class_dir, file_name)
            dest = os.path.join(split_dir, file_name)

            # Kiểm tra nếu file tồn tại trước khi sao chép
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, dest)  # Sử dụng copy2 để sao chép cả metadata
                except Exception as e:
                    print(f"Không thể sao chép {file_name}: {e}")


    # Copy các tập tin vào các thư mục tương ứng
    copy_files(train_files, 'train')
    copy_files(test_files, 'test')
    copy_files(val_files, 'val')

print("Hoàn thành việc phân chia dữ liệu.")
