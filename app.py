import gradio as gr
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights

class PetClassifier:
    def __init__(self):
        """Initialize the model and required components"""
        # Load pre-trained ResNet model with proper weights
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        # Initialize transforms for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # ImageNet class indices for cats and dogs
        self.cat_indices = list(range(281, 286))  # Cat classes in ImageNet
        self.dog_indices = list(range(151, 269))  # Dog classes in ImageNet

    def compute_saliency_map(self, input_tensor, target_class):
        """
        Compute saliency map using gradient computation
        
        Args:
            input_tensor: Preprocessed image tensor
            target_class: Target class index
            
        Returns:
            Saliency map as numpy array
        """
        # Enable gradient computation
        input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero all existing gradients
        self.model.zero_grad()
        
        # Calculate gradients
        output[0, target_class].backward()
        
        # Get gradients from input tensor
        gradients = input_tensor.grad.data.abs()
        
        # Pool the gradients across the color channels
        saliency_map = torch.max(gradients, dim=1)[0].squeeze()
        
        return saliency_map.cpu().numpy()

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to preprocessed tensor"""
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0)

    def classify_image(self, image: np.ndarray):
        """
        Classify image and generate saliency map
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing classification results and visualizations
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Get predicted class probabilities
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        
        # Calculate cat and dog probabilities
        cat_prob = sum(probabilities[i] for i in self.cat_indices)
        dog_prob = sum(probabilities[i] for i in self.dog_indices)
        
        # Normalize probabilities
        total_prob = cat_prob + dog_prob
        if total_prob > 0:
            cat_prob = cat_prob / total_prob
            dog_prob = dog_prob / total_prob
        
        # Get predicted class index (among all ImageNet classes)
        pred_class = prediction.argmax().item()
        
        # Generate saliency map using gradients
        saliency_map = self.compute_saliency_map(input_tensor, pred_class)
        
        # Normalize saliency map for visualization
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        saliency_map = (saliency_map * 255).astype(np.uint8)
        
        # Resize saliency map to match original image size
        saliency_map = Image.fromarray(saliency_map).resize(
            (image.shape[1], image.shape[0]),
            Image.BILINEAR
        )
        
        # Apply colormap for better visualization
        saliency_map = np.array(saliency_map)
        colored_map = np.zeros((saliency_map.shape[0], saliency_map.shape[1], 3), dtype=np.uint8)
        colored_map[:, :, 2] = saliency_map  # Blue channel
        
        # Determine final classification
        predicted_class = "CAT" if cat_prob > dog_prob else "DOG"
        
        return {
            'original_image': image,
            'saliency_map': colored_map,
            'prediction': f"Predicted class: {predicted_class}",
            'confidence': f"Confidence scores:\nCat: {cat_prob:.2%}\nDog: {dog_prob:.2%}"
        }

def create_gradio_interface():
    """Create and launch Gradio interface"""
    classifier = PetClassifier()
    
    def process_and_display(input_image):
        if input_image is None:
            return None, None, "No image provided", ""
        
        results = classifier.classify_image(input_image)
        return [
            results['original_image'],
            results['saliency_map'],
            results['prediction'],
            results['confidence']
        ]
    
    # Define Gradio interface
    iface = gr.Interface(
        fn=process_and_display,
        inputs=gr.Image(),
        outputs=[
            gr.Image(label="Original Image"),
            gr.Image(label="Saliency Map"),
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Confidence Scores")
        ],
        title="Cat vs Dog Classifier with Visualization",
        description="Upload an image of a cat or dog to see the classification results and explanation."
    )
    
    return iface

if __name__ == "__main__":
    # Install required packages
    # !pip install torch torchvision gradio pillow numpy
    
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch(share=True)  # share=True enables Colab compatibility