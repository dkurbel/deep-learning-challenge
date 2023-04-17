# deep-learning-challenge

### This repository contains two Jupyter Notebook files, each with Python code for building and training a neural network model to predict the success of charitable donations using a given dataset. The dataset is loaded from a CSV file ("charity_data.csv") hosted online.

## The first notebook file ("AlphabetSoup_classifier.ipynb") includes the following steps:

### - Preprocessing the dataset by dropping non-beneficial ID columns, binning application types and classifications with low frequency, and converting categorical data to numeric using one-hot encoding.

### - Splitting the preprocessed data into training and testing datasets.

### - Scaling the numerical data using StandardScaler.

### - Building a deep neural network model with three layers (two hidden layers and one output layer) using the Sequential model from Keras. The model uses the Rectified Linear Unit (ReLU) activation function for the hidden layers and the Sigmoid activation function for the output layer.

### - Compiling the model with binary cross-entropy as the loss function and Adam optimizer, and training the model on the training dataset with 10 epochs and batch size of 32.

### - Evaluating the trained model on the testing dataset and printing the loss and accuracy metrics.

### - Saving the trained model to an HDF5 file ("model.h5") for future use.

## The second notebook file ("AlphabetSoupCharity_Optimization.ipynb") includes similar steps as the first notebook, but with some slight differences in the preprocessing and training steps. However, the overall workflow and purpose of the code remain the same.

# My written report can be viewed here or at the end of the AlphabetSoupCharity_Optimization file.

# Report on the Neural Network Model for Alphabet Soup

## Overview of the Analysis:

### The purpose of this analysis is to create a deep learning model using neural networks to predict whether applicants for funding from Alphabet Soup, a charitable organization, will be successful or not. The dataset used for this analysis is a CSV file called charity_data.csv, which contains various features about each applicant, such as application type, classification, and other relevant information.

## Results:

## Data Preprocessing:

### - The target variable for the model is the "IS_SUCCESSFUL" column, which indicates an applicant’s success or failure in receiving funding.
### - The features for the model include all the columns in the original dataset, except for "EIN" and "NAME".
### - The "APPLICATION_TYPE" and "CLASSIFICATION" columns were one-hot encoded using pd.get_dummies() to convert the categorical data to numeric.

## Compiling, Training, and Evaluating the Model:

### - The neural network model was compiled using the Adam optimizer and binary crossentropy loss function, as this is a binary classification problem.
### - The base model architecture consists of three layers: one input layer with 10 units and ReLU activation, one hidden layer with 5 units and ReLU activation, and one output layer with 1 unit and sigmoid activation.
### - The optimized models were trained using different variations of units per layer and I utilized L1 regularization to prevent overfitting by adding a penalty term to the loss function.
### - Different activation functions and layer configurations were also experimented with, but the target model performance of 75% accuracy was not achieved.
### - The best model achieved an accuracy of about 72.5% on the test data, which is slightly below the target model performance of 75% accuracy.


## Summary:

### The deep learning neural network model developed for predicting successful applicants for funding from Alphabet Soup achieved a best accuracy of about 72.5% on the test data. Although this falls slightly short of the target model performance of 75% accuracy, the model may still provide valuable insights and predictions. To improve the model performance, additional experimentation with different model architectures, hyperparameter tuning, and feature engineering techniques could be performed. It may also be beneficial to explore other machine learning algorithms, such as decision trees, random forests, or support vector machines, to compare their performance with these neural network models. Updating with new data may also help to improve the model's accuracy.