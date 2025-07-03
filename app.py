import streamlit as st
from PIL import Image
import numpy as np
import requests
import io
import torch
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import rrdb as arch  # Adjust path if needed
import os
import base64
from io import BytesIO

def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



# Load ESRGAN model
model_path = 'weights/model_epoch718.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# Page setup
st.set_page_config(page_title="ESRGAN Image Enhancer", layout="wide")

# Custom styles
st.markdown(
    """
    <style>
    body {
        background-color: #0f1117;
        color: #e0e0e0;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #00f0ff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #007ACC;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stFileUploader>div>div>div {
        color: #f0f0f0;
    }
    .uploadedFileName {
        color: #ffffff !important;
    }
    .css-1aumxhk {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 1rem;
    }
    img {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✨ ESRGAN Image Enhancer")
st.write("🖼️ Upload an image to **enhance its resolution** using **ESRGAN** deep learning.")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load and show input image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Preprocess input for ESRGAN
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Convert output to RGB
    output_img = np.transpose(output, (1, 2, 0))
    output_img = (output_img * 255.0).round().astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # Resize input image for fair comparison
    resized_input = cv2.resize(np.array(image), (output_img.shape[1], output_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Calculate SSIM & PSNR
    ssim_score = compare_ssim(resized_input, output_img, channel_axis=-1, data_range=255, win_size=7)
    psnr_score = compare_psnr(resized_input, output_img, data_range=255)

    # Convert output to PIL for display
    output_pil = Image.fromarray(output_img)

    # Show images centered and resized
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧾 Input Image")
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<img src='data:image/png;base64,{image_to_base64(image)}' width='250' style='border-radius: 10px;'/>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("🚀 Enhanced Output")
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<img src='data:image/png;base64,{image_to_base64(output_pil)}' width='250' style='border-radius: 10px;'/>"
            f"<p style='font-size: 16px;'><br>🧪 <b>SSIM:</b> {ssim_score:.4f}<br>📏 <b>PSNR:</b> {psnr_score:.2f} dB</p>"
            f"</div>",
            unsafe_allow_html=True
        )

