# Fake News Detection

This repository contains a machine learning pipeline for detecting fake news, including feature extraction, model training, metrics comparison, and a prediction system. Multiple classification models are compared to evaluate performance and reliability.

## Features

- **Dataset Handling**  
  - Upload and process datasets from Kaggle.  
  - Import and load the fake news dataset.  

- **Feature Engineering**  
  - Perform feature selection for identifying relevant attributes.  
  - Extract features using TF-IDF for text data representation.

- **Model Development**  
  - Train and evaluate the following classifiers:
    - **Gradient Boosting Classifier**
    - **Decision Tree Classifier**
    - **Logistic Regression Classifier**
    - **Random Forest Classifier**

- **Model Comparison**  
  - Evaluate models based on key metrics, including:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-Score**
    - **Confusion Matrix**
  - Visualize model performance for better insight.

- **Prediction System**  
  - Build a prediction system to classify news articles as fake or real using the **Gradient Boosting Classifier**.

## Getting Started

### Prerequisites

- Python 3.x  
- Kaggle API credentials (`kaggle.json`) for dataset download.  
- Required Python libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

### Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/SubhanMalik1/fake-news-detection.git
   ```
2. Navigate to the project directory:  
   ```bash
   cd fake-news-detection
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Upload `kaggle.json` for dataset retrieval if needed.  
2. Run the Jupyter Notebook to execute the workflow:  
   ```bash
   jupyter notebook FakeNewsDetection.ipynb
   ```
3. Follow these steps in the notebook:
   - Preprocess the dataset.
   - Extract features using **TF-IDF**.
   - Train multiple classifiers.
   - Evaluate and compare model metrics.
   - Use the **Gradient Boosting Classifier** for predictions.

## Model Evaluation

The following classifiers are compared to find the best-performing model:

1. **Gradient Boosting Classifier**  
   - Known for high accuracy and robustness.  
2. **Decision Tree Classifier**  
   - Simple and interpretable, but prone to overfitting.  
3. **Logistic Regression Classifier**  
   - A baseline linear model for binary classification.  
4. **Random Forest Classifier**  
   - An ensemble technique for reducing overfitting and improving performance.

### Metrics Evaluated

- Logistic Regression: 98.89%
- Decision Tree Classifier: 99.61%
- Random Forest Classifier: 98.95%
- Gradient Boosting Classifier: 99.47%

### Visualization

- Performance metrics and comparisons are visualized using bar plots and confusion matrices to understand the models' strengths and weaknesses.

## Contributing

Contributions are welcome! Please fork this repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
