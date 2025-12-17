!pip install -q streamlit pyngrok torch torchvision pillow matplotlib opencv-python grad-cam
%%writefile app.py
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.strip().split("\n")

st.set_page_config(
    page_title="AI Image Classifier + Grad-CAM",
    layout="wide"
)

st.title("🧠 AI Image Classifier with Explainability")
st.write("Upload an image to see **prediction + model attention (Grad-CAM)**")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = outputs.argmax(dim=1).item()
        predicted_class = labels[predicted_idx]

    st.success(f"✅ **Prediction:** {predicted_class}")

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(predicted_idx)]
    )[0]

    rgb_img = np.array(img.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    st.subheader("🔥 Grad-CAM Heatmap")
    st.image(visualization, use_column_width=True)
from pyngrok import ngrok
import subprocess
import time

ngrok.set_auth_token("YOUR_NGROK_TOKEN")

ngrok.kill()

process = subprocess.Popen(
    [
        "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.enableCORS", "false"
    ]
)
time.sleep(5)

# Open public URL
public_url = ngrok.connect(8501)
print("🚀 Your app is live at:", public_url)
