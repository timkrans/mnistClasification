# Handwritten Digit Classification with Deep Learning

A comprehensive deep learning project that demonstrates neural network implementations for image classification using the MNIST dataset. This project compares the performance of feedforward neural networks with convolutional neural networks (CNNs) and extends the analysis to Fashion MNIST classification.

## 🎯 Project Overview

This project explores the effectiveness of different neural network architectures for image classification tasks:

- **MNIST Digit Classification**: Classifying handwritten digits (0-9)
- **Fashion MNIST Classification**: Classifying clothing items (10 categories)
- **Architecture Comparison**: Feedforward vs Convolutional Neural Networks

## 📊 Key Results

### MNIST Digit Classification
- **Feedforward Neural Network**: 96.5% accuracy
- **CNN (LeNet-5)**: 98.6% accuracy
- **Improvement**: 2.1% accuracy boost with CNN

### Fashion MNIST Classification
- **Custom CNN**: 92% accuracy on clothing classification

## 🏗️ Architecture Details

### Feedforward Neural Network
```
Input (784) → Dense(64) + ReLU + Dropout(0.5) → Dense(64) + ReLU + Dropout(0.5) → Dense(10) + Softmax
```

### Convolutional Neural Network (LeNet-5)
```
Input (32×32×1) → Conv2D(6, 5×5) + ReLU → MaxPool(2×2) → Conv2D(16, 5×5) + ReLU → MaxPool(2×2) → Flatten → Dense(120) + ReLU → Dense(84) + ReLU → Dense(10) + Softmax
```

### Fashion MNIST CNN
```
Input (32×32×1) → Conv2D(32, 3×3) + ReLU → MaxPool(2×2) → Conv2D(64, 3×3) + ReLU → MaxPool(2×2) → Flatten → Dense(128) + ReLU → Dense(10) + Softmax
```

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 📁 Project Structure

```
├── Project_MNIST.ipynb  # Main project notebook
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see requirements.txt)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install tensorflow numpy matplotlib jupyter
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**:
   - Navigate to `MNIST_Classification_Project.ipynb`
   - Run all cells to execute the complete analysis

## 📈 Key Features

### 1. Data Preprocessing
- Automatic data loading and normalization
- Reshaping for different network architectures
- One-hot encoding for multi-class classification

### 2. Model Implementation
- **Feedforward Network**: Simple dense layers with dropout regularization
- **LeNet-5 CNN**: Classic convolutional architecture
- **Custom CNN**: Modern architecture for Fashion MNIST

### 3. Training and Evaluation
- Comprehensive training history tracking
- Performance comparison between architectures
- Visualization of training curves

### 4. Visualization
- Sample image display
- Training/validation loss and accuracy plots
- Model performance comparison charts

## 🔍 Technical Insights

### Why CNNs Outperform Feedforward Networks?

1. **Spatial Feature Learning**: CNNs automatically learn spatial patterns in images
2. **Parameter Sharing**: Convolutional layers share parameters across spatial locations
3. **Translation Invariance**: CNNs are robust to small translations in input
4. **Hierarchical Feature Extraction**: Multiple layers capture increasingly complex features

### Key Findings

- **Architecture Matters**: CNN achieved 2.1% better accuracy than feedforward network
- **Feature Learning**: CNNs automatically discover relevant features without manual engineering
- **Generalization**: Same CNN architecture works well on different image classification tasks
- **Efficiency**: CNNs require fewer parameters while achieving better performance

## 📊 Performance Analysis

### Training Characteristics
- **Feedforward Network**: Shows signs of overfitting after ~15 epochs
- **CNN**: More stable training with better generalization
- **Fashion MNIST**: More challenging task requiring careful architecture design

### Model Comparison
| Model | Test Accuracy | Test Loss | Training Time |
|-------|---------------|-----------|---------------|
| Feedforward NN | 96.5% | 0.166 | ~30s |
| LeNet-5 CNN | 98.6% | 0.039 | ~2min |
| Fashion CNN | 92.0% | 0.244 | ~3min |

## 🎓 Learning Outcomes

This project demonstrates:

1. **Deep Learning Fundamentals**: Understanding of neural network architectures
2. **Computer Vision**: Application of CNNs to image classification
3. **Model Comparison**: Systematic evaluation of different approaches
4. **Data Preprocessing**: Proper handling of image data for neural networks
5. **Performance Analysis**: Interpretation of training curves and metrics

## 🔮 Future Enhancements

Potential improvements and extensions:

- **Data Augmentation**: Improve model robustness
- **Advanced Architectures**: ResNet, VGG, or Transformer-based models
- **Hyperparameter Tuning**: Systematic optimization of model parameters
- **Real-time Inference**: Web application for digit recognition
- **Transfer Learning**: Pre-trained models for improved performance

## 📚 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [LeNet-5 Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting improvements
- Adding new features
- Improving documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Note**: This project is designed for educational purposes and demonstrates fundamental concepts in deep learning and computer vision. The models achieve competitive performance on standard benchmarks while maintaining clarity and educational value. 