import numpy as np

def compare_explanations(gradcam_mask, shap_mask, threshold=0.5):
    """
    gradcam_mask: (224, 224) normalized 0-1
    shap_mask: (224, 224) normalized 0-1
    """
    # Threshold masks to get "Regions of Interest"
    gc_roi = gradcam_mask > threshold
    shap_roi = shap_mask > threshold
    
    intersection = np.logical_and(gc_roi, shap_roi).sum()
    union = np.logical_or(gc_roi, shap_roi).sum()
    
    iou = intersection / (union + 1e-8)
    
    # Qualitative summary
    if iou > 0.6:
        summary = "High Agreement: Both Grad-CAM and SHAP highlight the same diagnostic regions."
    elif iou > 0.3:
        summary = "Moderate Agreement: There is partial overlap between the two explanation methods."
    else:
        summary = "Low Agreement: Grad-CAM and SHAP focus on different features of the image."
        
    return iou, summary

if __name__ == "__main__":
    # Test with dummy data
    gc = np.zeros((224, 224))
    gc[50:150, 50:150] = 1.0
    
    sh = np.zeros((224, 224))
    sh[60:160, 60:160] = 1.0
    
    iou, summary = compare_explanations(gc, sh)
    print(f"IoU: {iou:.4f}")
    print(f"Summary: {summary}")
