# src/explain.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from model import get_model
from dataset import get_dataloaders
import torchvision
import torch.nn.functional as F
import cv2
import os
from functools import lru_cache

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache transforms to avoid recreating them
@lru_cache(maxsize=None)
def get_transforms():
    return {
        'basic': transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2470, 0.2435, 0.2616)),
        ]),
        'grad_cam': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

# helper to load saved model
def load_model(path='models/resnet_cifar10.pth', num_classes=10):
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# transform to convert PIL to model input
def preprocess_pil(pil_image):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    return transform(pil_image).unsqueeze(0)

# LIME explanation
def lime_explain(model, classes, pil_image, num_samples=50):  # Reduced samples
    # Optimize batch prediction with tensor operations
    def batch_predict(images):
        model.eval()
        
        # Process in smaller batches to save memory
        batch_size = 16  # Smaller batch size
        n_batches = (len(images) + batch_size - 1) // batch_size
        predictions = []
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(images))
                batch_images = images[start_idx:end_idx]
                
                # Convert to tensor efficiently
                batch_tensor = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float() / 255.0
                
                # Move to device and normalize
                batch_tensor = batch_tensor.to(device)
                mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1).to(device)
                std = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1).to(device)
                batch_tensor = (batch_tensor - mean) / std
                
                # Get predictions
                preds = model(batch_tensor)
                preds = F.softmax(preds, dim=1).cpu()
                
                predictions.append(preds)
                
                # Clear memory
                del batch_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return torch.cat(predictions, dim=0).numpy()

    # Resize image to smaller size for LIME to reduce memory usage
    small_size = (128, 128)  # Reduced size for LIME processing
    pil_image_small = pil_image.resize(small_size, Image.Resampling.LANCZOS)
    np_img = np.array(pil_image_small)
    
    # Create explainer with optimized parameters
    explainer = lime_image.LimeImageExplainer(feature_selection='auto')
    
    # Optimize explanation parameters
    explanation = explainer.explain_instance(
        np_img, 
        batch_predict,
        top_labels=1,  # Reduce to only what we need
        hide_color=0,
        num_samples=num_samples,
        batch_size=16,  # Smaller batch size
        segmentation_fn=None  # Use quickshift for faster processing
    )

    # Efficient visualization
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    # Create figure
    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.imshow(np_img)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(mark_boundaries(temp/255.0, mask))
    plt.title(f'LIME: {classes[top_label]}')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img_data  # Return the image data instead of showing it

# Grad-CAM with optimized memory usage
def grad_cam(model, pil_image, target_layer_name='layer4'):
    model.eval()
    transforms_dict = get_transforms()
    
    # Efficient preprocessing
    x = transforms_dict['grad_cam'](pil_image).unsqueeze(0).to(device)
    img = cv2.cvtColor(np.array(pil_image.resize((224,224))), cv2.COLOR_RGB2BGR)

    target_layer = dict([*model.named_modules()])[target_layer_name]
    
    gradients = []
    activations = []
    
    def save_grad(module, grad_input, grad_output):
        # Full backward hook
        gradients.append(grad_output[0].detach())
        return None
        
    def save_act(module, input, output):
        activations.append(output)
        return None

    handles = [
        target_layer.register_full_backward_hook(save_grad),
        target_layer.register_forward_hook(save_act)
    ]

    # Enable gradients for this computation
    x.requires_grad = True
    
    # Forward pass
    outputs = model(x)
    pred = outputs.argmax(dim=1).item()
    
    # Compute gradients
    model.zero_grad()
    if torch.cuda.is_available():
        score = outputs[0, pred].cuda()
    else:
        score = outputs[0, pred]
        
    score.backward()

    try:
        # Remove hooks
        for handle in handles:
            handle.remove()

        # Efficient CAM computation
        with torch.no_grad():
            grad = gradients[-1][0]
            act = activations[-1][0]
            weights = torch.mean(grad, dim=(1,2))
            cam = torch.sum(weights.view(-1, 1, 1) * act, dim=0)
            cam = torch.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
            cam = cam.cpu().numpy()
            
    finally:
        # Clean up
        del gradients
        del activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Efficient visualization
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    plt.ioff()
    fig = plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img_data  # Return the image data instead of showing it

if __name__ == '__main__':
    # load model and classes
    _, _, classes = get_dataloaders(batch_size=1)
    model = load_model(path='models/resnet_cifar10.pth', num_classes=len(classes))
    # take a test image from dataset
    import random
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    idx = random.randint(0, len(testset)-1)
    pil_img, label = testset[idx]
    print("True label:", testset.classes[label])
    lime_explain(model, testset.classes, pil_img, num_samples=100)
    grad_cam(model, pil_img, target_layer_name='layer4')
