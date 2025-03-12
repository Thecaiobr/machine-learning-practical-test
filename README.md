#Machine Learning Developer Test Caio Massote

## Overview
This project is designed to analyze embeddings derived from images to classify genetic syndromes. The embeddings are outputs from a pre-trained classification model, and the goal is to improve the understanding of the data distribution and enhance classification accuracy.

## Project Structure
```
apollo-solutions-ml-test
├── data
│   └── mini_gm_public_v0.1.p          # Pickle file containing the dataset
├── src
│   ├── data_processing.py               # Functions for loading and preprocessing data
│   ├── classification.py                # Implementation of K-Nearest Neighbors (KNN)
├── requirements.txt                     # Required Python packages
├── README.md                            # Documentation for the project
└── report
    ├── ML Practical Test Caio Massote Code Report.pdf  # PDF documentation of the created code
    └── ML Practical Test Caio Massote Interpretation.pdf   # PDF answering interpretation questions
```

## Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines
- **Data Processing**: Use `data_processing.py` to preprocess the data and visualize the embeddings using t-SNE, run the following command: `python src/data_processing.py`
This script will:
   - Load the dataset from the pickle file.
   - Preprocess the data by normalizing the embeddings and analyzing data imbalance.
   - Visualize the embeddings using t-SNE and plot the results.

- **Classification**: Use `classification.py` To train and evaluate the KNN model with different distance metrics, run the following command: `python src/classification.py`
This script will:
   - Load and preprocess the data.
   - Train the KNN model with Euclidean and Cosine distance metrics.
   - Evaluate the model using cross-validation.
   - Plot the ROC curves for the best k values for both distance metrics.

## Report
The project includes a detailed code documentation located in the `report` directory, summarizing methodologies, results, and insights gained from the analysis.

## Interpretation
An additional PDF document answering interpretation questions is also included in the `report` directory.
