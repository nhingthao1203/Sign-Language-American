import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

p_dir = 'A'
c_dir = p_dir.lower()
output_dir = f"D:\\SLA\\AtoZImage\\Letter_{p_dir}\\"
os.makedirs(output_dir, exist_ok=True)

offset = 20
crop_size = (224, 224)
flag = False


try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.GaussianBlur(img, (5, 5), 0)

        hands, img = detector.findHands(img)

        image_count = len([name for name in os.listdir(output_dir) if name.endswith(('.png', '.jpg', '.jpeg'))])

        cv2.putText(img, f'Image Count: {image_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(img, f'Folder: {p_dir}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if hands:
            hand = hands[0]

            x, y, w, h = hand['bbox']


            if w > 224 or h > 224 or w < 60 or h < 60 or w * h < 96 * 96:
                if w > 224 or h > 224:
                    cv2.putText(img, 'Error: Move your hand further to the screen!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
                    flag = False
                else:
                    cv2.putText(img, 'Error: Move your hand closer to the screen', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
                    flag = False
            else:
                y1 = max(0, y - offset)
                y2 = min(img.shape[0], y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(img.shape[1], x + w + offset)

                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size > 0:
                    imgCrop = cv2.resize(imgCrop, crop_size)


                    cv2.imshow("ImageCrop", imgCrop)

                    if flag:
                        cv2.imwrite(f"{output_dir}{c_dir}_{len(os.listdir(output_dir)) + 1}.png", imgCrop)
                        if (len(os.listdir(output_dir))) >= 1000:
                            flag = False

        cv2.imshow("Image", img)

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break
        elif interrupt & 0xFF == ord('n'):  # Nhấn phím 'n' để thay đổi ký tự thư mục
            p_dir = chr((ord(p_dir) + 1 - ord('A')) % 26 + ord('A'))
            c_dir = p_dir.lower()
            output_dir = f"D:\\SLA\\AtoZImage\\Letter_{p_dir}\\"
            os.makedirs(output_dir, exist_ok=True)
        elif interrupt & 0xFF == ord('a'):  # Nhấn phím 'a' để bật/tắt chế độ lưu ảnh
            flag = not flag

except Exception as e:
    print(f"Something wrong: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
