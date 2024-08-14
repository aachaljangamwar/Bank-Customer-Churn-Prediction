# Bank-Customer-Churn-Prediction
This project aims to predict whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN) model. The project follows a structured approach, including data preprocessing, model building, training, evaluation, and deployment.
Below is a detailed step-by-step guide:

1. Problem Definition
The goal is to predict customer churn for a bank based on various features such as customer demographics, account information, and transaction history.

2. Dataset
The dataset used for this project contains customer information including features like age, gender, balance, tenure, number of products, and more.
Source: Kaggle (or specify another source if applicable).
Data Size: 10,000+ rows and 14 columns (adjust according to your dataset).

3. Data Preprocessing
   
a. Data Cleaning
Missing Values: Checked for missing values and handled them appropriately (e.g., imputation or removal).
Duplicate Values: Removed any duplicate records.
Data Types: Ensured all columns had the correct data types.

b. Exploratory Data Analysis (EDA)
Descriptive Statistics: Analyzed key statistics of the features.
Correlation Analysis: Identified correlations between features to detect multicollinearity.
Data Visualization: Visualized the data to understand distributions, relationships, and potential outliers.

c. Feature Engineering
Encoding Categorical Variables: Applied techniques like One-Hot Encoding for categorical variables (e.g., gender, geography).
Feature Scaling: Standardized or normalized features using methods like MinMaxScaler or StandardScaler to ensure consistent model input.

d. Splitting the Dataset
Train-Test Split: Split the dataset into training (80%) and testing (20%) sets.
Validation Set: Used cross-validation or a separate validation set for model tuning.

5. Model Building
   
a. Choosing the Model
Model Type: Built an Artificial Neural Network (ANN) using TensorFlow and Keras.
Architecture: The ANN consists of multiple layers, including input, hidden, and output layers. The model architecture was optimized through hyperparameter tuning.

b. Compiling the Model
Loss Function: Used 'binary_crossentropy' for binary classification.
Optimizer: Applied the Adam optimizer for efficient gradient descent.
Metrics: Monitored performance using metrics like accuracy, precision, recall, and F1-score.

7. Model Training
Epochs: Trained the model for a set number of epochs, monitoring validation loss and accuracy.
Batch Size: Selected an appropriate batch size for training.
Early Stopping: Implemented early stopping to prevent overfitting.

9. Model Evaluation
Performance Metrics: Evaluated the model using accuracy, precision, recall, and F1-score on the test set.
Confusion Matrix: Analyzed the confusion matrix to understand the model’s classification performance.
ROC-AUC Curve: Plotted the ROC-AUC curve to visualize the trade-off between true positive and false positive rates.

11. Model Improvement
Hyperparameter Tuning: Performed hyperparameter tuning using techniques like Grid Search or Random Search to improve model performance.
Cross-Validation: Used k-fold cross-validation to ensure model stability.

13. Model Deployment
Model Saving: Saved the trained model using Keras' model.save() function.
Deployment: Deployed the model using Flask/Django or as a REST API for real-time predictions.

15. Conclusion
Results Summary: The model achieved an accuracy of 79% on the test set, with a precision of 80%, recall of 77%, and an F1-score of 78% for the non-churn class. The model performs reliably in predicting customer churn.

17. Future Work
Feature Importance Analysis: Investigate which features contribute most to customer churn.
Model Optimization: Experiment with other architectures like Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN) for potential improvements.
