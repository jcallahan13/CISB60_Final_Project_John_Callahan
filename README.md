# CISB60_Final_Project_John_Callahan
SpringfieldVision: Machine and Deep Learning for Simpsons Character Classification
Project Overview
SpringfieldVision is a machine learning and deep learning project aimed at classifying images of characters from The Simpsons TV show. Using a dataset of labeled images featuring five characters—Abraham Grampa Simpson, Bart Simpson, Homer Simpson, Lisa Simpson, and Marge Simpson—the project explores the development, training, and evaluation of Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) for multiclass image classification.

This project demonstrates the application of modern deep learning techniques to real-world image recognition problems while addressing challenges like class imbalance and model generalization.

Problem Statement
The objective of this project is to correctly classify images of The Simpsons characters into one of five categories:

Abraham Grampa Simpson
Bart Simpson
Homer Simpson
Lisa Simpson
Marge Simpson
The project involves building robust ML and DL pipelines to preprocess the data, train models, and evaluate performance using metrics like accuracy and confusion matrices.

Methodology
1. Data Preparation
The dataset contains 7,156 labeled images resized to dimensions 128x128x3.
The images were normalized to a [0, 1] range to ensure consistent input for the models.
The dataset was split into:
Training Set: 80% (5,724 images)
Testing Set: 20% (1,432 images)
2. Models
Artificial Neural Network (ANN)
A baseline model with fully connected layers, dropout for regularization, and softmax activation for multiclass classification.
Optimized using the Adam optimizer.
Convolutional Neural Network (CNN)
A deep learning model with convolutional layers, max pooling, batch normalization, and dropout for feature extraction and spatial pattern recognition.
Demonstrated superior performance over the ANN.
3. Evaluation Metrics
Loss and Accuracy: Monitored for both training and validation datasets.
Confusion Matrix: Provided insights into class-specific performance and highlighted misclassifications.
Mean Absolute Error (MAE): Used to assess prediction precision.
Results
ANN:
Training Accuracy: 66.83%
Training Loss: 0.8417
Most accurate for Homer Simpson (Class 2) with 357 correct predictions.
Struggled with Abraham Grampa Simpson and Bart Simpson due to misclassifications.
CNN:
Training Accuracy: 90.22%
Training Loss: 0.3402
Most accurate for Homer Simpson (Class 2) with 439 correct predictions.
Showed occasional confusion between Bart Simpson (Class 1) and Lisa Simpson (Class 3).
Challenges
Class Imbalance:
Certain characters (e.g., Homer Simpson) dominated predictions, suggesting a need for balancing techniques.
Misclassification:
Confusion between visually similar characters (e.g., Bart and Lisa Simpson) highlighted areas for model refinement.
Tools and Libraries
Core Libraries:
TensorFlow/Keras
Pandas, NumPy
Matplotlib, Seaborn
PIL (Python Imaging Library)
Key Methods:
Convolutional Neural Networks
Artificial Neural Networks
Data Augmentation
Confusion Matrix Evaluation
How to Run the Project
Clone the repository:

bash
Copy code
git clone https://github.com/<your-repo>/SpringfieldVision.git
cd SpringfieldVision
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook:

bash
Copy code
jupyter notebook CISB60_Final_Project_John_Callahan.ipynb
Ensure the dataset is in the appropriate directory for loading during execution.

Future Improvements
Implement data augmentation to address class imbalance.
Explore transfer learning techniques using pretrained models like ResNet or VGG.
Optimize hyperparameters using tools like Keras Tuner or GridSearchCV.
Add weighted loss functions to improve performance on underrepresented classes.
Acknowledgments
This project was completed as part of CISB 60: Machine and Deep Learning (Fall 2024) under the guidance of Professor Angel Martinon Hernandez. Special thanks to the contributors of the Simpsons dataset.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
