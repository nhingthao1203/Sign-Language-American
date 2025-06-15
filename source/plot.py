import pandas as pd
import matplotlib.pyplot as plt
import os

# Đọc các file CSV
files = [
    'E:\MSASL-valid-dataset-downloader\ASLLETTER\densenet121.csv',  # Thêm các đường dẫn file vào đây
    'E:\MSASL-valid-dataset-downloader\ASLLETTER\efficientnet.csv',
    'E:\MSASL-valid-dataset-downloader\ASLLETTER\googlenet.csv',
    'E:\MSASL-valid-dataset-downloader\ASLLETTER\mobilenetv2.csv',
    r'E:\MSASL-valid-dataset-downloader\ASLLETTER\resnet50.csv',
    'E:\MSASL-valid-dataset-downloader\ASLLETTER\SwinT.csv',
    r'E:\MSASL-valid-dataset-downloader\ASLLETTER\vgg16.csv',
    'E:\MSASL-valid-dataset-downloader\ASLLETTER\ViT.csv'
]

# Tạo thư mục lưu báo cáo nếu chưa có
os.makedirs('report', exist_ok=True)

# List để chứa các DataFrame
train_acc_data = []
train_loss_data = []
val_acc_data = []
val_loss_data = []
file_names = []

# Đọc từng file và thêm vào các list tương ứng
for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        train_acc_data.append(df['train_acc'])
        train_loss_data.append(df['train_loss'])
        val_acc_data.append(df['val_acc'])
        val_loss_data.append(df['val_loss'])
        file_names.append(os.path.basename(file).split('.')[0])  # Lấy tên file làm tên mô hình
    else:
        print(f"File {file} không tồn tại.")

# Gộp các dữ liệu lại thành một DataFrame duy nhất
epochs = df['epoch']  # Dùng số epoch của file đầu tiên
train_acc_combined = pd.concat(train_acc_data, axis=1)
train_loss_combined = pd.concat(train_loss_data, axis=1)
val_acc_combined = pd.concat(val_acc_data, axis=1)
val_loss_combined = pd.concat(val_loss_data, axis=1)

# 1. Biểu đồ độ chính xác huấn luyện
plt.figure(figsize=(12, 6))
for idx in range(train_acc_combined.shape[1]):
    plt.plot(epochs, train_acc_combined.iloc[:, idx], label=f'{file_names[idx]} Train Accuracy')
plt.title('Train Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('report/train_accuracy_comparison.png')
plt.close()

# 2. Biểu đồ độ chính xác kiểm tra
plt.figure(figsize=(12, 6))
for idx in range(val_acc_combined.shape[1]):
    plt.plot(epochs, val_acc_combined.iloc[:, idx], label=f'{file_names[idx]} Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('report/validation_accuracy_comparison.png')
plt.close()

# 3. Biểu đồ mất mát huấn luyện
plt.figure(figsize=(12, 6))
for idx in range(train_loss_combined.shape[1]):
    plt.plot(epochs, train_loss_combined.iloc[:, idx], label=f'{file_names[idx]} Train Loss', linestyle='--')
plt.title('Train Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('report/train_loss_comparison.png')
plt.close()

# 4. Biểu đồ mất mát kiểm tra
plt.figure(figsize=(12, 6))
for idx in range(val_loss_combined.shape[1]):
    plt.plot(epochs, val_loss_combined.iloc[:, idx], label=f'{file_names[idx]} Validation Loss', linestyle=':')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('report/validation_loss_comparison.png')
plt.close()

print("Charts saved in the 'report' folder.")