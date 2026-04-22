import torch
from torchvision import transforms
from PIL import Image
from models.resnet_wrapper import get_resnet_model
import numpy as np
import os


class XAIInference:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Fix model path (works from anywhere)
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'models', 'resnet18_weights.pth')

        # Load model
        self.model = get_resnet_model(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        image: PIL image
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()

        prediction = "Pneumonia" if prob > 0.5 else "Normal"
        confidence = prob if prob > 0.5 else 1 - prob

        return prediction, confidence, input_tensor


def get_original_np(image):
    """
    Get normalized numpy for Grad-CAM overlay
    """
    img_resized = image.resize((224, 224))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    return img_np