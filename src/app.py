# src/app.py
import streamlit as st
from PIL import Image
import torchvision
import torch
from model import get_model
from explain import lime_explain, grad_cam, load_model
from dataset import get_dataloaders

st.set_page_config(
    layout="centered",
    page_title="Explainable AI Demo",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.title("Interactive Explainable AI — Demo")
st.write("Upload an image (CIFAR-10 style); model predicts, explains, and you can provide feedback.")

uploaded = st.file_uploader("Upload image", type=['png','jpg','jpeg'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimize resource loading
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_resources():
    _, _, classes = get_dataloaders(batch_size=1)
    model = load_model(path='models/resnet_cifar10.pth', num_classes=len(classes))
    # Ensure model is in eval mode and using no_grad
    model.eval()
    return model, classes

# Cache image preprocessing
@st.cache_data
def preprocess_image(image_bytes):
    # Convert bytes to PIL Image using BytesIO
    from io import BytesIO
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    return preprocess(image).unsqueeze(0)

model, classes = load_resources()

if uploaded is not None:
    # Read the file once
    file_bytes = uploaded.getvalue()
    pil = Image.open(uploaded).convert('RGB')
    st.image(pil, caption='Uploaded image', width='stretch')

    # Use cached preprocessing with bytes for consistent hashing
    x = preprocess_image(file_bytes).to(device)
    
    # Move predictions outside of caching since we're dealing with tensors
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top5 = torch.topk(probs, 5)

    st.write("Top predictions:")
    for p, idx in zip(top5.values.cpu().numpy(), top5.indices.cpu().numpy()):
        st.write(f"{classes[idx]} — {p:.3f}")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Show LIME explanation", key="lime"):
            with st.spinner("Computing LIME explanation..."):
                # Clear memory before LIME
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                lime_result = lime_explain(model, classes, pil, num_samples=50)
                if lime_result is not None:
                    st.image(lime_result, caption='LIME Explanation', width='content')
                
    with col2:
        if st.button("Show Grad-CAM", key="gradcam"):
            with st.spinner("Computing Grad-CAM..."):
                gradcam_result = grad_cam(model, pil, target_layer_name='layer4')
                if gradcam_result is not None:
                    st.image(gradcam_result, caption='Grad-CAM Explanation', width='content')

    st.write("---")
    st.write("Feedback: did the model reason correctly (for the top prediction)?")
    feedback = st.radio("Is the prediction correct?", ("Yes", "No"))
    comment = st.text_input("Optional comment for feedback:")
    if st.button("Submit feedback"):
        # naive feedback saving: append to CSV
        import os, csv
        os.makedirs('feedback', exist_ok=True)
        with open('feedback/feedback.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([classes[top5.indices[0].item()], float(top5.values[0].item()), feedback, comment])
        st.success("Thanks — feedback saved.")
