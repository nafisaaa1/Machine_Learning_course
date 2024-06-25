# Supervised Machine Learning for Spotting and Sorting Cancer: An Entertaining Comparison

## Project Overview

This project delves into cancer detection and classification using supervised machine learning techniques. The model aims to predict cell health status and identify specific cancer types if present.

## Objectives

1. **Predicting Tissue Health Status:**
   - Train the model to predict whether a cell is healthy or cancerous based on labeled data comprising cellular characteristics and gene expressions.

2. **Discerning Specific Cancer Types:**
   - Distinguish between various cancer types if the cell is identified as cancerous.

3. **Exploring Diverse Machine Learning Algorithms:**
   - Conduct a comparative analysis of supervised machine learning algorithms, including Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and Neural Networks.

## Dataset Preparation

- **Dataset:** [Subset of TCGA & GTEx](https://zenodo.org/records/7828660) published by Iñigo Prada-Luengo.
- **Description:** The dataset integrates data from GTEx and TCGA projects, providing genetic and expression data from various healthy and cancerous human tissues.

## Preprocessing the Data

1. **Data Loading:**
   - Load the dataset and filter for high-count tissues.
   - Encode labels with 1 if cancerous and 0 if healthy.

2. **Splitting the Data:**
   - Split the dataset into training, validation, and test sets.

3. **Visualizing the Data:**
   - Visualize tissue counts and perform PCA for dimensionality reduction and visualization.

## Models and Training

### Logistic Regression, Random Forest, and K-Nearest Neighbors

- **Training and Validation:**
  - Train and validate models using the training and validation sets.
  - Evaluate models using cross-validation and additional metrics.

- **Testing:**
  - Test models on a separate test set to ensure they generalize well to unseen data.
  - Implement regularization and use robust evaluation metrics.

### Neural Network

- **Model Definition:**
  - Define a neural network with specific hyperparameters: 4 layers, ReLU activation, 10 epochs, learning rate of 1e-4, CrossEntropyLoss, and Adam optimizer.
  - Include batch normalization and dropout for regularization.

- **Training and Evaluation:**
  - Train and validate the neural network on lung, colon, and kidney datasets.
  - Evaluate the model on test sets.

## Evaluation Metrics

- **Accuracy:** Overall correctness of the model.
- **Precision:** Correctness of cancer predictions.
- **Recall:** Ability to identify all actual cancer cases.
- **F1-Score:** Balance between precision and recall.
- **AUC-ROC:** Area under the ROC curve for each class.

## Visualization

- **Loss and Accuracy Curves:**
  - Plot training and validation loss and accuracy for each model, including test metrics.

- **ROC Curves:**
  - Plot ROC curves for multi-class classification to evaluate the ability of models to distinguish between different classes.

## Conclusion

The models show good learning behavior with high accuracies and low losses, indicating good generalization. The findings offer valuable insights into optimizing healthcare interventions in cancer management, potentially enhancing early detection and treatment outcomes.


