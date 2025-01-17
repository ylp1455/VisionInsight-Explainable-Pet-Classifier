# 🔍 VisionInsight: Explainable Pet Classifier

## 🌟 Overview
VisionInsight is an advanced image classification system that not only identifies cats and dogs but also explains its decision-making process through visual saliency maps. This project demonstrates the practical application of Explainable AI (XAI) techniques in deep learning, making neural network decisions more transparent and interpretable.

## 🎯 Features
- Real-time cat and dog classification using ResNet50
- Interactive web interface powered by Gradio
- Visual explanation through saliency maps
- Confidence score visualization
- Easy deployment in Google Colab

## 🧠 Explainable AI Components
This project implements several key concepts in Explainable AI:

### Saliency Maps
Saliency maps highlight the regions of an input image that most strongly influence the model's classification decision. Our implementation uses gradient-based visualization, which:
- Computes gradients of the output with respect to the input image
- Identifies pixels that would most significantly affect the classification if changed
- Visualizes these important regions using a blue-scale heatmap

### Confidence Scores
The system provides detailed confidence scores for both categories (cats and dogs), offering:
- Normalized probability distributions
- Clear percentage-based confidence metrics
- Visual representation of decision certainty

## 🛠️ Technical Implementation
### Model Architecture
- Base Model: ResNet50 (pretrained on ImageNet)
- Input Processing: 224x224 image size with normalization
- Output: Binary classification (Cat vs Dog) with confidence scores

### Key Components
1. **Image Preprocessing**
   - Resize to standard dimensions
   - Normalize using ImageNet statistics
   - Convert to PyTorch tensors

2. **Classification Pipeline**
   - Forward pass through ResNet50
   - Probability computation using softmax
   - Class-specific confidence calculation

3. **Saliency Computation**
   - Gradient computation w.r.t input
   - Gradient pooling across channels
   - Normalization and visualization

## 🚀 Getting Started
### Prerequisites
```bash
pip install torch torchvision gradio pillow numpy
```

### Running the Application
```python
python app.py
```

### Google Colab Usage
1. Open the provided notebook
2. Run the installation cell
3. Execute the application cell
4. Click the generated public URL

## 📊 Example Usage
```python
from pet_classifier import PetClassifier

# Initialize the classifier
classifier = PetClassifier()

# Classify an image
results = classifier.classify_image(image)

# Access results
original_image = results['original_image']
saliency_map = results['saliency_map']
prediction = results['prediction']
confidence = results['confidence']
```

## 🔧 Customization
You can customize various aspects of the system:
- Modify the confidence threshold
- Adjust saliency map coloring
- Change the model architecture
- Add support for additional classes

## 🧪 Technical Details

### Saliency Map Generation
The saliency maps are generated using the following process:
```python
def compute_saliency_map(input_tensor, target_class):
    input_tensor.requires_grad_()
    output = model(input_tensor)
    output[0, target_class].backward()
    gradients = input_tensor.grad.data.abs()
    saliency_map = torch.max(gradients, dim=1)[0]
    return saliency_map
```

### Model Configuration
The system uses standard ImageNet normalization:
```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

## 📚 Further Reading
- [Understanding Saliency Maps](https://arxiv.org/abs/1312.6034)
- [ResNet Architecture](https://arxiv.org/abs/1512.03385)
- [Explainable AI Techniques](https://christophm.github.io/interpretable-ml-book/)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## ✨ Acknowledgments
- The PyTorch team for their excellent deep learning framework
- The Gradio team for their user interface components
- The scientific community for advancing Explainable AI techniques

## 📧 Contact
For questions and feedback, please open an issue in the GitHub repository.
