import pandas as pd
import shutil
import os

# Đọc file CSV để lấy danh sách video và gloss
csv_file = r'E:\ASL_Citizen\pythonProject\ASL_Citizen\splits\val.csv'  # Thay đổi đường dẫn đến file CSV của bạn
df = pd.read_csv(csv_file)

# Lấy danh sách các video từ cột 'Video file' trong CSV và gloss từ cột 'Gloss'
video_files = df['Video file'].tolist()
glosses = df['Gloss'].tolist()

# Đường dẫn thư mục chứa video gốc
source_dir = r'E:\ASL_Citizen\pythonProject\ASL_Citizen\videos'  # Thay đổi đường dẫn thư mục chứa video gốc

# Đường dẫn thư mục đích (tập test)
destination_dir = r'E:\ASL_Citizen\pythonProject\val'  # Thay đổi đường dẫn thư mục đích

# Đảm bảo thư mục đích tồn tại, nếu không sẽ tạo mới
os.makedirs(destination_dir, exist_ok=True)

# Sao chép từng video từ danh sách vào thư mục đích, tạo thư mục con theo gloss
for video, gloss in zip(video_files, glosses):
    # Tạo đường dẫn cho thư mục gloss con trong thư mục test
    gloss_folder = os.path.join(destination_dir, gloss)
    os.makedirs(gloss_folder, exist_ok=True)

    source_file = os.path.join(source_dir, video)
    destination_file = os.path.join(gloss_folder, video)

    # Kiểm tra nếu file video tồn tại và là định dạng video hợp lệ (ví dụ: .mp4)
    if os.path.exists(source_file) and video.endswith('.mp4'):  # Hoặc thay đổi theo loại video của bạn
        shutil.copy(source_file, destination_file)
        print(f"Đã sao chép {video} vào thư mục {gloss_folder}.")
    else:
        print(f"Không tìm thấy file video {video} hoặc không phải là định dạng hợp lệ.")
