import os

# Đường dẫn đến thư mục chứa các thư mục con
root_folder_path = r'E:\MSASL-valid-dataset-downloader\MS-ASL100_TEST'
root_folder_path1 = r'E:\MSASL-valid-dataset-downloader\MS-ASL100_TRAIN'
root_folder_path2 = r'E:\MSASL-valid-dataset-downloader\MS-ASL100_VAL'
# Hàm đếm số lượng video trong tất cả các thư mục con
def count_video_files_in_subfolders(root_folder):
    video_count = 0
    for subdir, _, files in os.walk(root_folder):
        # Lọc ra các file có định dạng .mp4
        video_count += len([file for file in files if file.endswith('.mp4')])
    return video_count

# Đếm số lượng video
total_video_files_1 = count_video_files_in_subfolders(root_folder_path)
total_video_files_2 = count_video_files_in_subfolders(root_folder_path1)
total_video_files_3 = count_video_files_in_subfolders(root_folder_path2)
print(total_video_files_1 )
print(total_video_files_2 )
print(total_video_files_3 )
print(total_video_files_1 + total_video_files_2 + total_video_files_3)