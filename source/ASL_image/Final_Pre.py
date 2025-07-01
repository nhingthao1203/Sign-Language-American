import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 35)

model_path = r'D:\Sign-Language-To-Text\Sign-Language-To-Text\resnet50_best_model.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Cannot load model weights: {e}")
    exit()

offset = 20
crop_size = (224, 224)

labels = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'blank'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.GaussianBlur(img, (5, 5), 0)

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            hand_type = hand['type']

            if hand_type == "Left":
                cv2.putText(img, 'Alert: Left hand detected!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            x, y, w, h = hand['bbox']

            # Provide feedback based on the size of the bounding box (hand distance)
            if w > 224 or h > 224:
                cv2.putText(img, 'Error: Move your hand further from the screen!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            elif w < 60 or h < 60:
                cv2.putText(img, 'Error: Move your hand closer to the screen!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            else:
                y1 = max(0, y - offset)
                y2 = min(img.shape[0], y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(img.shape[1], x + w + offset)

                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size > 0:
                    imgCrop = cv2.resize(imgCrop, crop_size)
                    cv2.imshow("ImageCrop", imgCrop)

                    img_pil = Image.fromarray(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB))
                    input_tensor = transform(img_pil).unsqueeze(0)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)

                        label = labels[predicted.item()] if predicted.item() < len(labels) else "Unknown"

                    cv2.putText(img, f'Prediction: {label}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Image", img)

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
