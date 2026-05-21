# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Load the dataset
Load the Iris dataset using a suitable library.

### STEP 2: Preprocess the data
Preprocess the data by handling missing values and normalizing features.

### STEP 3: Split the dataset
Split the dataset into training and testing sets.

### STEP 4: Train the model
Train a classification model using the training data.

### STEP 5: Evaluate the model
Evaluate the model on the test data and calculate accuracy.

### STEP 6: Display results
Display the test accuracy, confusion matrix, and classification report.

## PROGRAM


### Name: Saravanan G

### Register Number:212223230194

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/customers.csv')
data.head()

from google.colab import drive
drive.mount('/content/drive')

# Drop ID column as it's not useful for classification
data = data.drop(columns=["ID"])

# Handle missing values
data.fillna({"Work_Experience": 0, "Family_Size": data["Family_Size"].median()}, inplace=True)

# Encode categorical variables
categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1"]
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])  # A, B, C, D -> 0, 1, 2, 3

# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 =nn.Linear(input_size, 32)
        self.fc2 =nn.Linear(32,16)
        self.fc3 =nn.Linear(16, 8)
        self.fc4 =nn.Linear(8,4)

    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x
# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()

  #Include your code here



    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# Initialize model
model =PeopleClassifier(input_size=X_train.shape[1])

# Define loss function and optimizer
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=100)

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=[str(i) for i in label_encoder.classes_])
print("Name: Saravanan G ")
print("Register No: 212223230194")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# Prediction for a sample input
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)
with torch.no_grad():
    output = model(sample_input)
    # Select the prediction for the sample (first element)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
print("Name: Saravanan G ")
print("Register No: 212223230194")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {label_encoder.inverse_transform([y_test[12].item()])[0]}')
```

### Dataset Information
<img width="1238" height="255" alt="image" src="https://github.com/user-attachments/assets/8bff1817-bfd1-4eb0-b9ac-4c43d458eae3" />


### OUTPUT
<img width="346" height="236" alt="image" src="https://github.com/user-attachments/assets/00850cd1-6938-45cd-9d20-61efc5f6d32d" />

## Confusion Matrix

<img width="673" height="561" alt="image" src="https://github.com/user-attachments/assets/7844be59-154c-4b76-9c6f-4ad28239ff46" />

## Classification Report
<img width="621" height="413" alt="image" src="https://github.com/user-attachments/assets/8802bfdd-ad89-4344-803d-89d6b355fa38" />


### New Sample Data Prediction
<img width="363" height="151" alt="image" src="https://github.com/user-attachments/assets/a77475ae-d0f6-4ad8-9455-232b1112e18e" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch.
