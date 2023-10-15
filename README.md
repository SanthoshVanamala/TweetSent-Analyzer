# TweetSent Analyzer

## Description
The TweetSent Analyzer is a machine learning project that performs sentiment analysis on Twitter data. The model was trained using a RandomForestClassifier and achieved an accuracy of 87%. The project involved several steps including data preprocessing, model training, and model evaluation.

## Table of Contents
1. Installation
2. Usage
3. Results
4. Contributing

## Installation
This project requires Python 3.x and several Python libraries including pandas, numpy, matplotlib, seaborn, and scikit-learn. All the dependencies can be installed by running `pip install -r requirements.txt` in the command line.

## Usage
The project is implemented in a Google Colab and can be run in any environment that supports Google Colab, including Jupyter notebooks . Here are the steps to run the project:

1. Load the dataset: `df = pd.read_csv('text.csv')`
2. Preprocess the data: `X = preprocess_text(df['text'])`
3. Train the model: `classifier.fit(X_train, y_train)`
4. Evaluate the model: `print(classification_report(y_test,y_pred))`

## Results
The model achieved an accuracy of 87% on the test set. The confusion matrix and classification report are as follows:

Confusion Matrix:
[[708  44]
 [ 85 126]]
              precision    recall  f1-score   support

           0       0.89      0.94      0.92       752
           1       0.74      0.60      0.66       211

    accuracy                           0.87       963
   macro avg       0.82      0.77      0.79       963
weighted avg       0.86      0.87      0.86       963

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have any suggestions to improve the project.
