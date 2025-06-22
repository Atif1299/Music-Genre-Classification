# Enhanced Music Genre Classification Project

## Project Overview

This project implements an advanced music genre classification system using multiple deep learning architectures and enhanced preprocessing techniques. The system achieves **85.37% accuracy** on the GTZAN dataset, classifying music into 10 different genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

## Dataset

- **GTZAN Dataset**: 1000 songs (30 seconds each) across 10 genres
- **3-second features**: 9990 samples (10 segments per song)
- **30-second features**: 1000 samples (1 segment per song)
- **Spectrogram images**: Generated from audio files for CNN-based models

## Implementation Methodology

### Data Processing Strategy

The project uses a dual approach combining both audio features and spectrogram images:

1. **Audio Feature Extraction**: 57 musical features including:

   - Chroma features (chroma_stft_mean, chroma_stft_var)
   - Spectral features (spectral_centroid, spectral_bandwidth, rolloff)
   - MFCC coefficients (20 features with mean and variance)
   - Rhythm features (tempo, zero_crossing_rate)
   - Harmony and perceptual features

2. **Spectrogram Generation**: Converting audio to visual representations for CNN processing

### Advanced Preprocessing Pipeline

#### Feature Engineering

- **Multiple scaling options**: StandardScaler, MinMaxScaler, RobustScaler
- **Statistical feature selection**: SelectKBest with f_classif
- **Optional PCA**: Dimensionality reduction for better performance
- **Temporal sequence creation**: Converting features into time-series data for RNN models

#### Image Enhancement

- **Noise reduction**: Gaussian filtering for cleaner spectrograms
- **Contrast enhancement**: 20% increase in contrast
- **Brightness adjustment**: 10% increase in brightness
- **Sharpness enhancement**: 10% increase in sharpness
- **Data augmentation**: Training-time augmentation for better generalization

#### Data Quality Assurance

- **Missing value detection**: Comprehensive data quality checks
- **Duplicate removal**: Automatic duplicate detection and removal
- **Class balancing**: Automatic balancing for imbalanced datasets
- **Stratified splitting**: Proper train/validation/test splits maintaining class distribution

### Enhanced Model Architectures

#### CNN Model (64.53% accuracy)

```python
- Conv2D layers with BatchNormalization
- GlobalAveragePooling2D (reduces overfitting)
- Dropout layers (0.3-0.5 rates)
- Dense layers with ReLU activation
```

#### Bidirectional LSTM (59.50% accuracy)

```python
- Bidirectional LSTM layers for better temporal modeling
- BatchNormalization after each LSTM layer
- Dropout for regularization
- Dense layers with decreasing units
```

#### Enhanced GRU (68.00% accuracy)

```python
- Bidirectional GRU with attention-like mechanisms
- Multiple GRU layers with decreasing complexity
- Advanced dropout scheduling
- BatchNormalization for stable training
```

#### CNN-LSTM Ensemble (85.37% accuracy) - **BEST MODEL**

```python
- CNN branch for spatial feature extraction
- LSTM branch for temporal feature modeling
- Feature fusion through concatenation
- Multi-level dense layers with dropout
```

#### Multi-Input Fusion Model

```python
- Combines spectrogram images and extracted features
- Parallel processing branches
- Late fusion strategy
- Enhanced regularization techniques
```

### Training Optimizations

#### Callback Strategy

- **Early Stopping**: Prevents overfitting with patience=15
- **ReduceLROnPlateau**: Adaptive learning rate reduction
- **ModelCheckpoint**: Saves best model based on validation accuracy

#### Hyperparameter Optimization

- **Learning Rate**: 0.001 with adaptive scheduling
- **Batch Size**: 32 for optimal memory-performance balance
- **Epochs**: Up to 100 with early stopping
- **Optimizer**: Adam with optimal beta values

#### Evaluation Metrics

- **Accuracy**: Primary metric
- **Top-3 Accuracy**: Additional evaluation metric
- **Per-class analysis**: Detailed genre-wise performance
- **Confusion Matrix**: Visual error analysis

## Results Analysis

### Model Performance Comparison

| Model                 | Accuracy   | Key Features               |
| --------------------- | ---------- | -------------------------- |
| Enhanced CNN          | 64.53%     | Spatial feature extraction |
| Enhanced LSTM         | 59.50%     | Temporal sequence modeling |
| Enhanced GRU          | 68.00%     | Attention-like mechanisms  |
| **Enhanced Ensemble** | **85.37%** | **CNN-LSTM fusion**        |

### Per-Class Performance (Best Model)

| Genre     | Accuracy | Precision | Recall | F1-Score |
| --------- | -------- | --------- | ------ | -------- |
| Blues     | 87.50%   | 0.98      | 0.88   | 0.92     |
| Classical | 100.00%  | 0.67      | 1.00   | 0.80     |
| Country   | 59.30%   | 1.00      | 0.59   | 0.74     |
| Disco     | 85.50%   | 0.94      | 0.85   | 0.90     |
| Hiphop    | 97.50%   | 0.72      | 0.97   | 0.83     |
| Jazz      | 95.45%   | 0.80      | 0.95   | 0.87     |
| Metal     | 95.50%   | 0.97      | 0.95   | 0.96     |
| Pop       | 69.00%   | 0.98      | 0.69   | 0.81     |
| Reggae    | 83.50%   | 0.94      | 0.83   | 0.89     |
| Rock      | 80.40%   | 0.81      | 0.80   | 0.81     |

### Key Insights

1. **Best Performance**: Enhanced Ensemble (CNN-LSTM) achieved the highest accuracy
2. **Strong Genres**: Classical, Hiphop, Jazz, and Metal show excellent performance
3. **Challenging Genres**: Country and Pop show lower accuracy, indicating need for further optimization
4. **Overall Improvement**: 15-25% improvement over baseline models

## Technical Implementation Details

### Feature Preprocessing Functions

```python
def advanced_feature_preprocessing(features_df, feature_selection=True,
                                 scaling_method='standard', pca_components=None):
    # Comprehensive feature preprocessing pipeline
    # Including scaling, selection, and temporal sequence creation
```

### Image Processing Functions

```python
def advanced_image_preprocessing(image_path, target_size=(128, 128),
                               augment=False, noise_reduction=True):
    # Advanced spectrogram enhancement with multiple techniques
```

### Model Architecture Functions

```python
def create_enhanced_cnn_model(input_shape, num_classes, dropout_rate=0.3):
    # Enhanced CNN with GlobalAveragePooling and better regularization

def create_enhanced_lstm_model(input_shape, num_classes, dropout_rate=0.3):
    # Bidirectional LSTM with BatchNormalization

def create_enhanced_gru_model(input_shape, num_classes, dropout_rate=0.3):
    # Enhanced GRU with attention-like mechanisms

def create_ensemble_cnn_lstm(input_shape, num_classes, dropout_rate=0.3):
    # CNN-LSTM ensemble for best performance
```

### Training Configuration

```python
def get_enhanced_callbacks(model_name, patience=15):
    # Comprehensive callback strategy for optimal training

def compile_model_with_optimal_settings(model, learning_rate=0.001):
    # Optimal compilation settings with Top-3 accuracy metric
```

## Setup and Implementation

### Prerequisites

```bash
pip install tensorflow==2.15.0
pip install librosa
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
pip install pillow opencv-python scipy scikit-image
```

### Project Structure

```
ANN Project/
├── archive (2)/Data/
│   ├── Complete_Project.ipynb          # Main implementation
│   ├── features_3_sec.csv             # 3-second features
│   ├── features_30_sec.csv            # 30-second features
│   ├── images_original/               # Spectrogram images
│   ├── Best_Models/                   # Saved models
│   └── genres_original/               # Original audio files
├── models/                            # Model architectures
├── preprocessing/                     # Data preprocessing
├── training/                          # Training scripts
├── utils/                             # Utility functions
└── web_app/                           # Web application
```

### Configuration Parameters

```python
# Data Configuration
USE_3SEC = True                    # Use 3-second or 30-second features
APPLY_BALANCING = True             # Apply class balancing
USE_FEATURE_SELECTION = True       # Apply feature selection
SCALING_METHOD = 'standard'        # Scaling method
PCA_COMPONENTS = None              # PCA components (None to skip)

# Model Configuration
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
```

### Training Process

1. **Data Loading**: Load features and spectrogram images
2. **Preprocessing**: Apply scaling, feature selection, and augmentation
3. **Model Building**: Create enhanced architectures
4. **Training**: Train with callbacks and monitoring
5. **Evaluation**: Comprehensive performance analysis
6. **Saving**: Save models and preprocessing components

## Model Files

### Saved Models

- `enhanced_cnn_model.h5` - CNN model for spectrogram classification
- `enhanced_lstm_model.h5` - Bidirectional LSTM model
- `enhanced_gru_model.h5` - Enhanced GRU model
- `enhanced_ensemble_model.h5` - CNN-LSTM ensemble (best model)
- `multi_input_fusion_model.h5` - Multi-input fusion model

### Preprocessing Components

- `preprocessing_components.pkl` - Label encoder, scalers, selectors
- `enhanced_models_results.pkl` - Training results and metadata

## Usage

### Training

```python
# Run the Complete_Project.ipynb notebook
# Or use the training scripts in the training/ directory
```

### Inference

```python
import tensorflow as tf
import pickle

# Load model and preprocessing components
model = tf.keras.models.load_model('enhanced_ensemble_model.h5')
with open('preprocessing_components.pkl', 'rb') as f:
    components = pickle.load(f)

# Preprocess new data and predict
# (Implementation depends on input type - audio, features, or images)
```

## Key Success Factors

1. **Ensemble Approach**: Combining CNN and LSTM architectures
2. **Advanced Preprocessing**: Multiple enhancement techniques
3. **Proper Regularization**: Dropout, BatchNormalization, Early Stopping
4. **Feature Engineering**: Temporal sequence creation and feature selection
5. **Data Augmentation**: Image enhancement and augmentation
6. **Hyperparameter Tuning**: Optimized learning rates and batch sizes

## Future Improvements

1. **Attention Mechanisms**: Implement transformer-based architectures
2. **Data Augmentation**: More sophisticated audio augmentation
3. **Transfer Learning**: Pre-trained models for better feature extraction
4. **Ensemble Voting**: Multiple model predictions combination
5. **Cross-Validation**: K-fold validation for more robust evaluation

## Conclusion

This enhanced music genre classification system demonstrates the effectiveness of combining multiple deep learning architectures with advanced preprocessing techniques. The CNN-LSTM ensemble approach achieved 85.37% accuracy, significantly outperforming individual models and providing a robust solution for music genre classification.

The project showcases the importance of:

- **Comprehensive preprocessing** for better feature representation
- **Ensemble methods** for improved performance
- **Proper regularization** to prevent overfitting
- **Advanced evaluation metrics** for thorough analysis
- **Modular architecture** for easy experimentation and improvement
