import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_shap(model, input_tensor, background_tensors):
    """
    model: PyTorch model
    input_tensor: (1, 3, 224, 224)
    background_tensors: (N, 3, 224, 224) - subset of training data
    """
    # model must be in eval mode
    model.eval()
    
    # Define a function to explain - SHAP expects model output
    # For binary classification with Sigmoid, the output is (batch, 1)
    explainer = shap.GradientExplainer(model, background_tensors)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_tensor)
    
    # shap_values is a list for each output. For single output, it's [values]
    # each element is (batch, 3, 224, 224)
    return shap_values[0] # Return the importance map for the pneumonia class

def plot_shap(shap_values, original_image):
    """
    shap_values: (1, 3, 224, 224)
    original_image: numpy (224, 224, 3) normalized [0, 1]
    """
    # Transpose shap values to (224, 224, 3) for visualization
    # We take the absolute mean across channels for a simple heatmap
    shap_img = np.abs(shap_values[0]).mean(axis=0)
    
    # Normalize for display
    shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min() + 1e-8)
    
    return shap_img

if __name__ == "__main__":
    from models.resnet_wrapper import get_resnet_model
    from torchvision import transforms
    from PIL import Image
    import os
    
    model = get_resnet_model(pretrained=False)
    model.load_state_dict(torch.load('models/resnet18_weights.pth', map_location='cpu'))
    
    # Background images (needed for GradientExplainer)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get a few background images
    bg_images = []
    img_dir = 'data/images'
    for f in os.listdir(img_dir)[:5]: # 5 samples for speed
        if f.endswith(('.jpeg', '.jpg', '.png')):
            img = Image.open(os.path.join(img_dir, f)).convert('RGB')
            bg_images.append(transform(img))
    
    bg_tensors = torch.stack(bg_images)
    
    # Test image
    img_path = os.path.join(img_dir, os.listdir(img_dir)[10])
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    
    print("Generating SHAP values...")
    shap_vals = generate_shap(model, input_tensor, bg_tensors)
    shap_viz = plot_shap(shap_vals, None)
    
    plt.imsave('shap_output.png', shap_viz, cmap='hot')
    print("SHAP generated: shap_output.png")
