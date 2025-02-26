# Brain Tumor Classification via Deep Learning

## Project Overview
This project implements a brain tumor classification system using deep learning techniques. The model utilizes a pre-trained VGG16 architecture with transfer learning to classify brain MRI scans into four categories:
- Glioma tumor
- Meningioma tumor
- No tumor
- Pituitary tumor

## Dataset
The dataset consists of approximately 4,800 labeled MRI scans divided into pre-defined training and validation sets. The dataset can be downloaded from Kaggle (link to be added).

## Methodology

### Data Preprocessing
- Images are resized to a uniform size
- Pixel values are normalized to the range [0, 1]
- File paths are systematically created using os.path.join for platform-independent access

### Model Architecture
The model leverages the VGG16 architecture with transfer learning:
- The pre-trained VGG16 model serves as a feature extractor
- Initial layers capturing generic image features are kept frozen
- Final layers are replaced with new Dense layers using softmax activation
- The model is adapted for the four-class classification task

### Training Process
- Loss Function: Sparse categorical cross-entropy
- Optimizer: Adam
- Metrics: Accuracy
- Early stopping implemented to prevent overfitting
- TensorBoard visualization utilized for monitoring training

## Implementation Details

### Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib (for visualization)
- scikit-learn (for evaluation metrics)

### Code Structure
```
brain-tumor-classification/
├── data/
│   ├── training/
│   └── validation/
├── models/
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── README.md
```

## Usage

### Setup
```bash
# Clone the repository
git clone https://github.com/username/brain-tumor-classification.git
cd brain-tumor-classification

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

## Results
The model was trained for 12 epochs with a batch size of 32. Performance metrics include:
- Overall accuracy
- Precision, recall, and F1-score for each tumor class
- Test loss to evaluate generalization capability

## Future Work
- Explore different hyperparameter configurations
- Test alternative pre-trained models
- Implement more extensive data augmentation techniques
- Investigate model explainability for clinical applications

## References
1. Saleh, A., Sukaik, R., & Abu-Naser, S. S. (2020). Brain tumor classification using deep learning.
2. Paul, J. S., Plassard, A. J., Landman, B. A., & Fabbri, D. (2017). Deep learning for brain tumor classification.
3. Ari, A., & Hanbay, D. (2018). Deep learning based brain tumor classification and detection system.
4. Díaz-Pernas, F. J., Martínez-Zarzuela, M., Antón-Rodríguez, M., & González-Ortega, D. (2021). A deep learning approach for brain tumor classification and segmentation using a multiscale convolutional neural network.
5. Tandel, G. S., et al. (2019). A review on a deep learning perspective in brain cancer classification.

## Author
Muhammad Tahir Zia  
Bachelors Computer Engineering  
GIK Institute, Topi, Pakistan  
u2021465@giki.edu.pk

