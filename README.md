# Breast Cancer Survivability Prediction

## Introduction
This project focuses on using different machine learning algorithms to predict the survivability of women with breast cancer at a certain progression of the disease. The dataset used was adapted from Kaggle and includes various features such as age, race, marital status, tumor stage, lymph node stage, cancer stage, grade, differentiation, estrogen status, progesterone status, number of regional nodes examined, number of positive regional nodes, months survived, and survival status of the patient.

## Preprocessing

### Data Description
- T Stage: Indicates the size and extent of the primary tumor.
- N Stage: Indicates the number of neighboring lymph nodes that are cancerous.
- 6th Stage: Combines T, N, and M classifications, tumor grade, and results of ER/PR and HER2 testing.
- Grade: Determines the grade of invasive breast cancer based on the appearance of cancer cells.
- A Stage: Summarizes the stage of cancer, indicating if it has spread outside the breast or to distant parts of the body.
- Estrogen Status: Indicates if cancer cells need estrogen to grow.
- Progesterone Status: Indicates if breast cancer is sensitive to progesterone.

### Encoding
Ordinal coding was used for features like T Stage, N Stage, 6th Stage, A Stage, differentiation, and grade. Binary encoding was used for features like estrogen status, progesterone status, and survival status.

### Unsupervised Learning
- Dealing with Unbalanced Data: The target and features had imbalanced data. Oversampling was performed using the Synthetic Minority Oversampling Technique (SMOTE) to evenly distribute the data.
- Data Correlation: A correlation matrix was used to identify relationships between variables.

### Data Splitting
The data was divided into training (80%) and validation (20%) sets.

### Data Scaling
The data was scaled using the MinMaxScaler() package from scikit-learn to transform the values to a range between 0 and 1.

## Logistic Regression

### Initial Model
The initial logistic regression model was trained without any regularization or feature transformations. It achieved a training accuracy of approximately 80.4% and a testing accuracy of 78.5%.

### Regularization and Feature Transformations
Different regularization techniques (L1 and L2) and feature transformations (X2 and X3) were applied to improve the model's accuracy. The accuracy results for each combination of regularization and feature transformations were plotted.

- L1 No Feature Transformations:
###
![My Image](https://raw.githubusercontent.com/JackShkifati28/ML-Breast-Cancer/main/images/L1Linear.png)


- L2 No Feature Transformations:
###
![My Image](https://raw.githubusercontent.com/JackShkifati28/ML-Breast-Cancer/main/images/L2Linear.png)


- L1 x^2 Feature Transformations:
###
![My Image](https://raw.githubusercontent.com/JackShkifati28/ML-Breast-Cancer/main/images/L1LinearX2.png)


- L2 x^2 Feature Transformations:
###
![My Image](https://raw.githubusercontent.com/JackShkifati28/ML-Breast-Cancer/main/images/L2LinearX2.png)

- L1 x^3 Feature Transformations:
###
![My Image](https://raw.githubusercontent.com/JackShkifati28/ML-Breast-Cancer/main/images/L1linearx3.png)

- L2 x^3 Feature Transformations:
###
![My Image](https://raw.githubusercontent.com/JackShkifati28/ML-Breast-Cancer/main/images/L2linearx3.png)


The best logistic regression model was achieved with L1 regularization and X3 feature transformation, resulting in a testing accuracy of 91.3% with a C value of 0.014563.

## Support Vector Machines (SVM)

### Initial Model
The initial SVM model was trained without any regularization or kernel transformations. It achieved a training accuracy of approximately 80.5% and a testing accuracy of 78.7%.

### Regularization and Kernel Transformations
Different regularization techniques (L2 squared) and kernel transformations (linear, polynomial, and RBF) were applied to improve the model's accuracy. The accuracy results for each combination of regularization and kernel transformations were plotted.

![L2 Regularization with Linear Kernel](images/svm_l2_linear_kernel.png)

![L2 Regularization with Polynomial Kernel](images/svm_l2_poly_kernel.png)

![L2 Regularization with RBF Kernel](images/svm_l2_rbf_kernel.png)

The best SVM model was achieved with L2 regularization and RBF kernel transformation, resulting in a testing accuracy of 83.7% with C values ranging from 0.000010 to 0.000215.

## Neural Networks

### Initial Model
The initial neural network model was trained with a simple architecture of two hidden layers with five nodes each. It achieved a training accuracy of approximately 80.4% and a testing accuracy of 78.3%.

### Architecture and Activation Functions
Different neural network architectures with varying numbers of hidden layers and nodes were tested. Activation functions such as ReLU, Sigmoid, and tanh were applied. The accuracy results for each architecture and activation function were plotted.

![ReLU Activation Function](images/nn_relu.png)

![Sigmoid Activation Function](images/nn_sigmoid.png)

![tanh Activation Function](images/nn_tanh.png)

The best-performing neural network model was achieved with ReLU activation function and L1 regularization, resulting in a testing accuracy of 85.09% with four hidden layers, each containing seven nodes.

## Conclusion
In this project, we applied logistic regression, support vector machines, and neural networks to predict the survivability of women with breast cancer. The best-performing models achieved accuracies ranging from 83.7% to 91.3%.

For logistic regression, the best model used L1 regularization and a cubed feature transformation (X3). For support vector machines, the best model used RBF kernel transformation and L2 squared regularization. Finally, for neural networks, the best model used ReLU activation function and L1 regularization.

These models can potentially assist healthcare professionals in predicting the survivability of breast cancer patients at different stages, allowing for more informed treatment decisions.

