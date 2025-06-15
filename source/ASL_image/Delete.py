import os


data_dir = "E:\\Sign-Language-To-Text\\AtoZ\\"


classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)


    files = os.listdir(class_dir)

    # Xóa các ảnh có số thứ tự lớn hơn ..._1100
    for file_name in files:
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            try:
                # Tách phần số từ tên file
                base_name, index = file_name.rsplit('_', 1)
                index = int(index.split('.')[0])
                if index > 1100:
                    os.remove(os.path.join(class_dir, file_name))
                    print(f"Đã xóa: {file_name}")
            except ValueError:
                continue

print("Done")
