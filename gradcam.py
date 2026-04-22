import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, input_tensor, target_layer, original_image):
    """
    model: PyTorch model
    input_tensor: preprocessed tensor (1, 3, 224, 224)
    target_layer: list of layers to target
    original_image: numpy array (224, 224, 3) normalized to [0, 1]
    """
    # Ensure gradients are enabled globally for this call
    with torch.enable_grad():
        # Ensure gradients are enabled for all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        input_tensor.requires_grad = True
        
        # We use the underlying resnet model for more direct hooking
        actual_model = model.resnet if hasattr(model, 'resnet') else model
        cam = GradCAM(model=actual_model, target_layers=target_layer)
        
        # For binary output (1 neuron), index 0 is the logit
        targets = [ClassifierOutputTarget(0)]
        
        # Generate heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam

if __name__ == "__main__":
    from models.resnet_wrapper import get_resnet_model
    from torchvision import transforms
    from PIL import Image
    
    # Load model
    model = get_resnet_model(pretrained=False)
    model.load_state_dict(torch.load('models/resnet18_weights.pth', map_location='cpu'))
    model.eval()
    
    # Sample image
    img_path = 'data/images/radiol.2020200490.fig3.jpeg' # Just an example
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img).unsqueeze(0)
        
        # Target layer for ResNet18
        target_layers = [model.resnet.layer4[-1]]
        
        vis, mask = generate_gradcam(model, input_tensor, target_layers, img_np)
        cv2.imwrite('gradcam_output.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("Grad-CAM generated: gradcam_output.jpg")
