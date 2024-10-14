import cv2
import streamlit as st
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

st.title("Sign Language to Text Conversion")
st.markdown("<h3 style='text-align: center; color: white;'>Real-time Hand Sign Language Detection</h3>",
            unsafe_allow_html=True)

if 'predicted_sequence' not in st.session_state:
    st.session_state['predicted_sequence'] = ''
if 'current_symbol' not in st.session_state:
    st.session_state['current_symbol'] = ''

#st.markdown(f"### Stored Sequence: {st.session_state['predicted_sequence']}")

if st.button("Clear Sequence"):
    st.session_state['predicted_sequence'] = ''

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Camera Feed**")
    camera_feed = st.empty()

with col2:
    st.markdown("**Processed Feed**")
    processed_feed = st.empty()

model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 35)
model_path = r'D:\SLAProject\vgg16_best_model.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

labels = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'blank', 'backspace'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

stop = st.button("Stop Camera")
offset = 20
crop_size = (224, 224)

if cap.isOpened():
    while True:
        if stop:
            break

        success, img = cap.read()
        if not success:
            st.error("Failed to capture image from webcam.")
            break

        hands, img = detector.findHands(img)

        label = "No hand detected"
        img_crop_resized = np.zeros_like(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            if w > 224 or h > 224 or w * h < 96 * 96:
                cv2.putText(img, 'Error: Move your hand further from the screen!', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif w < 60 or h < 60 or w * h < 96 * 96:
                cv2.putText(img, 'Error: Move your hand closer to the screen!', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                y1 = max(0, y - offset)
                y2 = min(img.shape[0], y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(img.shape[1], x + w + offset)

                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size > 0:
                    imgCrop = cv2.resize(imgCrop, crop_size)

                    img_pil = Image.fromarray(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB))
                    input_tensor = transform(img_pil).unsqueeze(0)

                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)

                        label = labels[predicted.item()] if predicted.item() < len(labels) else "Unknown"

                    img_crop_resized = cv2.resize(imgCrop, (img.shape[1], img.shape[0]))

                    cv2.putText(img, f'Prediction: {label}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if label == 'backspace' and len(st.session_state['predicted_sequence']) > 0:
                        st.session_state['predicted_sequence'] = st.session_state['predicted_sequence'][:-1]
                    elif label != 'blank' and label != 'backspace':
                        st.session_state['predicted_sequence'] += label

                    #st.markdown(f"### Stored Sequence: {st.session_state['predicted_sequence']}")

        camera_feed.image(img, channels="BGR", use_column_width=True)
        processed_feed.image(cv2.cvtColor(img_crop_resized, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

cap.release()
cv2.destroyAllWindows()
