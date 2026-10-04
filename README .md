# Waste Material Classification with CNN

## Overview

This project builds a deep learning image classifier for identifying waste materials from photographs. The model uses **transfer learning with MobileNetV2**, pretrained on ImageNet, with a custom neural-network classification head trained for nine waste categories.

The project focuses on:

- Image preprocessing and dataset splitting
- Transfer learning with a frozen MobileNetV2 backbone
- Class imbalance handling with class weights
- Early stopping to reduce overfitting
- Hyperparameter tuning of the custom classification head
- Confusion-matrix and per-class performance analysis
- Comparison with traditional machine-learning models using extracted CNN features
- Saving the final trained model for later use

## Waste Categories

The classifier recognizes the following nine classes:

1. Cardboard
2. Food Organics
3. Glass
4. Metal
5. Miscellaneous Trash
6. Paper
7. Plastic
8. Textile Trash
9. Vegetation

## Dataset

The notebook expects the **RealWaste** dataset to be available in a directory named:

```text
RealWaste/
```

Each waste category should have its own subdirectory:

```text
RealWaste/
├── Cardboard/
├── Food Organics/
├── Glass/
├── Metal/
├── Miscellaneous Trash/
├── Paper/
├── Plastic/
├── Textile Trash/
└── Vegetation/
```

The images are resized to **224 × 224 pixels** and processed in batches of 32.

### Dataset split

The dataset is divided into:

| Dataset | Images |
|---|---:|
| Training | 3,802 |
| Validation | 475 |
| Test | 475 |
| Total | 4,752 |

The validation and test images are separated using a stratified split to preserve class proportions. A hash-based check is also used to verify that no images occur in both sets.

## Model Architecture

The final classifier uses **MobileNetV2** as a frozen feature extractor.

```text
Input Image
    │
    ▼
224 × 224 × 3
    │
    ▼
Rescaling
0–255 → -1–1
    │
    ▼
MobileNetV2
ImageNet pretrained
Frozen
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout (0.5)
    │
    ▼
Dense Layer (256 units, ReLU)
    │
    ▼
Dense Layer (9 units, Softmax)
    │
    ▼
Waste Class
```

### Why MobileNetV2?

MobileNetV2 provides useful pretrained visual features without requiring the entire convolutional network to be trained from scratch. Freezing the backbone reduces training time and helps limit overfitting given the relatively small dataset.

### Model parameters

The initial MobileNetV2 model contains approximately **2.42 million parameters**:

- Trainable parameters: approximately 165,129
- Frozen parameters: approximately 2,257,984

## Training

The model uses:

- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss:** Categorical Cross-Entropy
- **Metric:** Accuracy
- **Batch size:** 32
- **Class weights:** Balanced class weighting
- **Early stopping:** Validation loss with patience of 3 epochs

Class weights were calculated from the training set so that errors on underrepresented classes have greater influence during training.

## Hyperparameter Tuning

Several classification heads were evaluated using the frozen MobileNetV2 features:

| Configuration | Validation Loss |
|---|---:|
| Baseline: 128 units, dropout 0.3 | 0.6227 |
| 128 units, dropout 0.5 | 0.5831 |
| 256 units, dropout 0.3 | 0.5801 |
| 256 → 128 units, dropout 0.3 | 0.6269 |
| 128 units, dropout 0.3 + L2 | 0.7697 |
| **256 units, dropout 0.5** | **0.5433** |

The **256-unit layer with 0.5 dropout** achieved the lowest validation loss and was selected as the final classification head.

## Final Results

The tuned CNN was evaluated once on the held-out test set.

### Overall performance

- **Test accuracy:** 81.68%
- **Macro F1-score:** approximately 0.82

### Class recall

| Class | Recall |
|---|---:|
| Vegetation | 0.97 |
| Food Organics | 0.93 |
| Paper | 0.93 |
| Glass | 0.90 |
| Textile Trash | 0.84 |
| Cardboard | 0.78 |
| Metal | 0.78 |
| Plastic | 0.76 |
| Miscellaneous Trash | 0.57 |

### Strongest classes

The model performed particularly well on:

- Vegetation
- Food Organics
- Paper
- Glass

These categories tend to have more visually distinctive characteristics.

### Most difficult classes

**Miscellaneous Trash** was the weakest class. This is expected because it is a broad catch-all category containing objects with less consistent visual characteristics.

Plastic and Metal were also challenging. Their appearance can overlap significantly, particularly when objects are:

- Painted
- Printed
- Crushed
- Metallic-looking
- Made from plastic with reflective surfaces

## Error Analysis

The notebook investigates common Plastic ↔ Metal classification errors by displaying incorrectly classified examples.

The analysis found that some plastic objects can visually resemble metal, while some metal objects can resemble plastic because of coatings, paint, labels, shape, or damage.

The model also showed confusion between:

- Plastic and Metal
- Paper and Cardboard
- Glass and Plastic

This demonstrates why confusion matrices and individual misclassified images are useful in addition to overall accuracy.

## Comparison with Traditional Machine Learning

The project also evaluates traditional machine-learning models using the 1,280-dimensional features extracted by MobileNetV2.

The models include:

- Logistic Regression
- Support Vector Machine (RBF)
- Random Forest
- Tuned Keras classification head

Additional analysis includes:

- 5-fold cross-validation
- Logistic Regression grid search
- Comparison of class-weighted and unweighted training
- PCA visualization of the extracted feature space

The CNN remains the final model because it directly accepts an image as input and integrates the pretrained feature extractor with the classification head.

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

A recommended repository structure is:

```text
waste-material-classification/
│
├── RealWaste/
│   ├── Cardboard/
│   ├── Food Organics/
│   ├── Glass/
│   ├── Metal/
│   ├── Miscellaneous Trash/
│   ├── Paper/
│   ├── Plastic/
│   ├── Textile Trash/
│   └── Vegetation/
│
├── part2_cnn_classifier.ipynb
├── waste_cnn.keras
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Running the Notebook

Make sure the `RealWaste` dataset is located next to the notebook.

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
part2_cnn_classifier.ipynb
```

Run the notebook cells from top to bottom.

The notebook trains the classifier and saves the final model as:

```text
waste_cnn.keras
```

## Using the Saved Model

The trained model can be loaded with TensorFlow/Keras:

```python
from tensorflow import keras

model = keras.models.load_model("waste_cnn.keras")
```

The model expects an image input of approximately:

```text
224 × 224 × 3
```

and returns probabilities for the nine waste categories.

## Reproducibility

Random seeds are set to 42 for NumPy, Python, and TensorFlow.

However, training results can still vary between runs. The notebook reports that the final model achieved **81.68% test accuracy** in the documented run, while an earlier run reached approximately **84%**.

Therefore, the reported score should be treated as a representative result rather than a guaranteed result for every execution.

## Limitations

Several limitations should be considered:

- The dataset is relatively small for image classification.
- Textile Trash has only around 38 images in the test set.
- Results can vary between training runs.
- Training was performed on a CPU.
- Only a limited number of misclassified images were manually inspected.
- Only MobileNetV2 was used as the pretrained backbone.
- Miscellaneous Trash is visually heterogeneous and difficult to classify consistently.

## Future Improvements

Potential improvements include:

1. **Fine-tune MobileNetV2**
   - Unfreeze selected upper layers after training the classification head.
   - Use a smaller learning rate during fine-tuning.

2. **Data augmentation**
   - Random rotations
   - Horizontal flips
   - Zooming
   - Cropping
   - Brightness and contrast adjustments

3. **Try additional pretrained architectures**
   - EfficientNet
   - ResNet
   - DenseNet
   - MobileNetV3

4. **Collect more data**
   - Especially for Miscellaneous Trash and Textile Trash.

5. **Improve error analysis**
   - Investigate more misclassified examples.
   - Examine confusion patterns by object type and image quality.

6. **Deploy the model**
   - Build an API using FastAPI or Flask.
   - Create a web or mobile interface for waste classification.
   - Integrate the classifier into an intelligent waste-management assistant.

## Key Takeaway

The project demonstrates that transfer learning with a frozen **MobileNetV2** backbone can effectively classify waste materials with limited training data. The tuned model achieved **81.68% test accuracy** and a **macro F1-score of approximately 0.82**.

The main challenge is distinguishing visually similar materials such as **Plastic and Metal**, while the broad **Miscellaneous Trash** category remains the most difficult class.

The resulting `waste_cnn.keras` model provides a foundation for integrating image-based waste classification into a larger intelligent waste-management system.
