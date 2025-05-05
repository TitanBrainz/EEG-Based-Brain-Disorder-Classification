# EEG Classification with Deep Learning Models

This project implements and compares various machine learning and deep learning models for classifying EEG data into 7 classes. The models include:

-   **LeNet**
-   **AlexNet**
-   **ResNet**
-   **GoogLeNet**

## Introduction

Electroencephalography (EEG) is a widely used technique to measure electrical activity in the brain. This project focuses on classifying EEG data into 7 distinct classes based on preprocessed features. The dataset undergoes preprocessing steps such as PCA for dimensionality reduction and SMOTE for class balancing. The models are trained and evaluated using metrics like accuracy, confusion matrices, and classification reports.

## Dataset

The dataset used in this project is `EEG.machinelearing_data_BRMH.csv`, which contains EEG features and corresponding labels. Key details:

-   **Features**: EEG signals processed into numerical features.
-   **Target**: `main.disorder` column, representing 7 classes of disorders.
-   **Preprocessing**:
    -   Removal of low-variance features.
    -   Imputation of missing values.
    -   Dimensionality reduction using PCA (99% variance retained).
    -   Class balancing using SMOTE.

## Project Workflow

1. **Data Preprocessing**:

    - Load the dataset.
    - Encode categorical columns.
    - Apply PCA for dimensionality reduction.
    - Balance the dataset using SMOTE.

2. **Model Training**:

    - Train deep learning models (LeNet, AlexNet, ResNet, GoogLeNet) on the preprocessed data.
    - Use data augmentation during training to improve generalization.

3. **Evaluation**:
    - Evaluate models using accuracy, confusion matrices, and classification reports.
    - Compare model performance using training and validation accuracy/loss plots.

## How to Run the Project

1. Clone the repository or download the project files.
2. Ensure the following dependencies are installed:
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `tensorflow`
    - `matplotlib`
    - `seaborn`
    - `imbalanced-learn`
3. Run the following command to start training and evaluation:
    ```bash
    python train_compare_models.py
    ```

## Output Plots

### Training and Validation Accuracy Comparison

![Figure 5](output_plots/run4/Figure_5.png)

### Confusion Matrices

![Figure 6](output_plots/run4/Figure_6.png)

## Results

The models were evaluated on their ability to classify EEG data into 7 classes. Key metrics include:

-   **Accuracy**: Training and validation accuracy for each model.
-   **Confusion Matrices**: Visual representation of true vs. predicted labels.
-   **Classification Reports**: Precision, recall, and F1-score for each class.

## Conclusion

This project demonstrates the effectiveness of deep learning models in classifying EEG data. By leveraging advanced architectures like ResNet and GoogLeNet, the models achieve high accuracy and robust performance across all classes.

## Acknowledgments

Special thanks to the contributors of the dataset and the developers of the libraries used in this project.
