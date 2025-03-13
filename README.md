# hideNet: Hierarchical Generative Compression Framework

![hideNet Architecture](./demo.jpg)

hideNet is an innovative neural network framework for generative image compression that leverages hierarchical tree-based decomposition to achieve efficient compression and high-quality reconstruction. By recursively splitting images into odd and even pixel subsets, hideNet creates a multi-resolution representation that can be effectively encoded and later reconstructed using learned generative models.

## Core Concept

hideNet works on a simple yet powerful principle: during compression, an image is hierarchically decomposed into multiple resolution levels by extracting odd and even pixels. During decompression, the process is reversed using neural networks to predict missing pixel values at each level.

### Compression Process (Training Stage)
1. At each level, the image is split into two components:
   * `Io` (odd pixels) - extracted and stored
   * `Ie` (even pixels) - extracted and used for the next decomposition level

2. This process repeats recursively, creating a tree structure where each level has 1/2^n of the original image size

3. The system stores a small representation of the lowest level and learns to predict odd pixels from even pixels

### Decompression Process (Sampling Stage)
1. Starting with the compact representation (lowest level), the system expands the image level by level
2. At each level, the model predicts the missing odd pixels (`Io`) from the available even pixels (`Ie`)
3. Once both pixel sets are available, they are merged to reconstruct a higher resolution image
4. This process repeats until the full image is restored

## Architecture

The neural architecture consists of:
- **UNetLikeLite**: A lightweight U-Net inspired model that handles the pixel prediction task
- **CNNP**: A multi-head convolutional network with residual connections for enhanced feature extraction
- Custom split and merge operations for efficient pixel manipulation

## Implementation Details

### Requirements
```
gdown
jupyterlab
matplotlib
numpy
opencv-python
scikit-image
scikit-learn
scipy
torch
torchvision
```

### Key Components

- **Pixel Splitting**: The `split_image` function extracts odd and even pixel subsets
- **Image Merging**: The `merge_images` function reconstructs images by interleaving pixel subsets
- **Training Pipeline**: Implements progressive compression and decompression cycles
- **Evaluation**: Uses structural similarity (SSIM) metrics to quantify reconstruction quality

### Model Configuration
- Training runs for 10 epochs by default
- Supports multiple convolutional heads for feature extraction
- Uses a UNet-like architecture with skip connections
- Optimized with AdamW and cosine annealing learning rate scheduler

## Usage

To train and evaluate hideNet:

```python
# Clone the repository
git clone https://github.com/kanavgoyal898/hidenet.git
cd hideNet/

# Install dependencies
pip install -r requirements.txt

# Run the training script
cd src/
python main.py
```

## How It Works

1. **Training Stage**:
   - The model learns to predict odd pixels from even pixels at multiple resolution levels
   - Loss is calculated using MSE between predicted and actual odd pixels
   - Structural similarity is tracked to monitor reconstruction quality

2. **Compression**:
   - Input image is repeatedly decomposed by extracting odd and even pixels
   - Only the smallest representation needs to be stored

3. **Decompression**:
   - The smallest representation is progressively expanded
   - At each level, the model predicts missing pixels
   - Predicted and available pixels are merged to form the next resolution level

## Applications

- Efficient image and video compression
- Progressive image transmission
- Low-bandwidth image communication
- Neural network-based codec systems

## Future Work

- Extending to video compression with temporal coherence
- Exploring adaptive compression based on content complexity
- Implementing attention mechanisms to improve detail preservation
- Optimizing for specific domains (medical imaging, satellite imagery, etc.)
