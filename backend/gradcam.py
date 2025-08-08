import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import io
import base64

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            # For single output models, use index 0
            if len(output.shape) == 2 and output.shape[1] == 1:
                class_idx = 0
            else:
                class_idx = torch.argmax(output, dim=1)
        # Backward pass
        self.model.zero_grad()
        if len(output.shape) == 2 and output.shape[1] == 1:
            # Single output case
            output[0, 0].backward(retain_graph=True)
        else:
            output[0, class_idx].backward(retain_graph=True)
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        # Pool the gradients across the spatial dimensions
        pooled_gradients = torch.mean(gradients, dim=[1, 2])
        # Weight the channels by corresponding gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=0)
        # ReLU on top of the heatmap
        heatmap = F.relu(heatmap)
        # Normalize the heatmap more aggressively for better visibility
        heatmap_min = torch.min(heatmap)
        heatmap_max = torch.max(heatmap)
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            # Apply power function to enhance high values (make them more visible)
            heatmap = torch.pow(heatmap, 0.5)  # Square root for better contrast
        else:
            heatmap = torch.zeros_like(heatmap)
        # Detach before converting to numpy
        return heatmap.detach().cpu().numpy()

def create_gradcam_image(original_image, heatmap):
    """Create a Grad-CAM visualization overlaid on the original image"""
    # Resize heatmap to match original image size
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
        if original_image.shape[0] == 3:  # CHW format
            original_image = np.transpose(original_image, (1, 2, 0))
    
    # Validate heatmap
    if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.size == 0:
        print("Invalid heatmap for Grad-CAM visualization")
        return original_image.astype(np.uint8)
    
    # Ensure heatmap is 2D
    if len(heatmap.shape) > 2:
        heatmap = np.squeeze(heatmap)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
        print("Heatmap contains NaN or infinite values")
        return original_image.astype(np.uint8)
    
    h, w = original_image.shape[:2]
    
    # Ensure heatmap has valid dimensions before resizing
    if heatmap.shape[0] == 0 or heatmap.shape[1] == 0:
        print("Heatmap has zero dimensions, skipping Grad-CAM")
        return original_image.astype(np.uint8)
    
    # Convert heatmap to float32 for better OpenCV compatibility
    heatmap_float = heatmap.astype(np.float32)
    
    try:
        heatmap_resized = cv2.resize(heatmap_float, (w, h), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"OpenCV resize error: {e}")
        print(f"Heatmap shape: {heatmap.shape}, Target size: ({w}, {h})")
        print(f"Heatmap dtype: {heatmap.dtype}, min: {np.min(heatmap)}, max: {np.max(heatmap)}")
        # Fallback: use numpy resize
        from scipy import ndimage
        try:
            zoom_factors = (h / heatmap.shape[0], w / heatmap.shape[1])
            heatmap_resized = ndimage.zoom(heatmap, zoom_factors, order=1)
        except:
            print("Fallback resize also failed, returning original image")
            return original_image.astype(np.uint8)
    
    # Normalize heatmap to 0-255 with better contrast
    if np.max(heatmap_resized) > 0:
        heatmap_resized = heatmap_resized / np.max(heatmap_resized)  # Normalize to 0-1
        # Apply gamma correction for better visibility
        heatmap_resized = np.power(heatmap_resized, 0.7)  # Enhance contrast
    
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap (JET gives red for high values)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Convert original image to uint8 if needed
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    
    # Ensure both images are in RGB format
    if len(original_image.shape) == 3:
        original_image = np.uint8(original_image)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create a mask for significant heatmap values (threshold to only show strong activations)
    mask = heatmap_resized > 50  # Only overlay where heatmap is significant
    
    # Create the overlay
    superimposed = original_image.copy()
    superimposed[mask] = (0.3 * original_image[mask] + 0.7 * heatmap_colored[mask]).astype(np.uint8)
    superimposed = np.uint8(superimposed)
    
    return superimposed

def image_to_base64(image):
    """Convert numpy array image to base64 string"""
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    return None
