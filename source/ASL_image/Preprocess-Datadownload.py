import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

input_base_dir = r"D:\SLA\archive\Test_Alphabet"
output_base_dir = r"D:\SLA\archive\processed_data"
os.makedirs(output_base_dir, exist_ok=True)

detector = HandDetector(maxHands=1)
offset = 20
crop_size = (224, 224)

# Duyệt qua từng thư mục Letter_A -> Letter_Z
for letter in [chr(i) for i in range(ord('A'), ord('Z') + 1)]:
    input_dir = os.path.join(input_base_dir, letter)
    output_dir = os.path.join(output_base_dir, letter)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing folder: {input_dir}")

    # Duyệt qua từng ảnh trong thư mục
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Cannot read image: {img_path}")
                continue

            img = cv2.GaussianBlur(img, (5, 5), 0)

            hands, _ = detector.findHands(img, draw=False)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                if w > 60 and h > 60 and w < 224 and h < 224 and w * h >= 96 * 96:
                    y1 = max(0, y - offset)
                    y2 = min(img.shape[0], y + h + offset)
                    x1 = max(0, x - offset)
                    x2 = min(img.shape[1], x + w + offset)

                    img_crop = img[y1:y2, x1:x2]

                    if img_crop.size > 0:
                        img_crop = cv2.resize(img_crop, crop_size)

                        new_img_name = f"{letter.lower()}_{len(os.listdir(output_dir)) + 1}.png"
                        cv2.imwrite(os.path.join(output_dir, new_img_name), img_crop)

print("Done processing all folders.")
