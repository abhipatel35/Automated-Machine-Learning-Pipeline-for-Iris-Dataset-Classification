# Automated Machine Learning Pipeline for Iris Dataset Classification

This project implements an automated machine learning pipeline for classifying the Iris dataset using a Decision Tree classifier. The pipeline includes dimensionality reduction using Principal Component Analysis (PCA), standard scaling of features, and training the classifier. The project serves as a demonstration of how to create an end-to-end machine learning workflow using scikit-learn pipelines.

## Requirements

- Python 3.x
- scikit-learn
- numpy

## Installation

You can install the required packages using pip:

```bash
pip install scikit-learn numpy
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/abhipatel35/automated-ml-pipeline-iris.git
```

2. Navigate to the project directory:

```bash
cd automated-ml-pipeline-iris
```

3. Run the script:

```bash
python main.py
```

## Pipeline Overview

1. **Data Loading:** The Iris dataset is loaded using scikit-learn's datasets module.
2. **Data Splitting:** The dataset is split into training and testing sets.
3. **Pipeline Creation:** A scikit-learn pipeline is created, which includes:
   - Dimensionality reduction using PCA.
   - Standard scaling of features.
   - Training a Decision Tree classifier.
4. **Model Training:** The pipeline is fitted to the training data.
5. **Model Evaluation:** The accuracy score of the model on the test set is computed.
