Problem Statement: Iris Flower Classification

The Iris Flower Classification problem is a supervised machine learning task where the objective is to classify iris flowers into one of three species:
1.Setosa
2.Versicolor
3.Virginica

Each flower is described using four numerical features:
1.Sepal Length (cm)
2.Sepal Width (cm)
3.Petal Length (cm)
4.Petal Width (cm)

Goal:
The goal is to develop a machine learning model that can accurately predict the species of an iris flower based on these four features. This is a classification problem, meaning the output will be one of the three predefined categories (species).

Why is this important?
The Iris dataset is widely used in machine learning for learning and benchmarking classification models.
It serves as an excellent beginner-friendly problem to understand data preprocessing, visualization, model training, and evaluation.
The problem is simple yet effective in demonstrating the power of supervised learning algorithms like Decision Trees, Random Forests, and Support Vector Machines.

Challenges in the Problem
Some features may have overlapping values between different species, making classification slightly challenging.
Choosing the right classification algorithm can impact model accuracy.

Code:

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target  # Add target variable
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first 5 rows of the dataset
print("Dataset Sample:")
print(df.head())

# Visualize the dataset
sns.pairplot(df, hue='species')
plt.show()

# Split data into features and target variable
X = df.drop(columns=['species'])  # Features
y = df['species']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to predict species for custom inputs
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    return prediction[0]

# Example prediction
example_prediction = predict_species(5.1, 3.5, 1.4, 0.2)
print(f"Predicted Species: {example_prediction}")

