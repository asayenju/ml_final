import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Ensure the library module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the dataset
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'credit_approval.csv'))

print(f"Dataset shape: {data.shape}")
print(f"Column names: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head())

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify numeric and categorical columns
# Numeric columns: attr2_num (1), attr3_num (2), attr8_num (7), attr14_num (13), attr15_num (14)
numeric_cols = {1, 2, 7, 13, 14}
categorical_cols = {0, 3, 4, 5, 6, 8, 9, 10, 11, 12}

print(f"\nNumeric columns (indices): {numeric_cols}")
print(f"Categorical columns (indices): {categorical_cols}")

# Encode categorical columns using LabelEncoder
X_processed = X.copy().astype(object)
label_encoders = {}

for col_idx in categorical_cols:
    le = LabelEncoder()
    X_processed[:, col_idx] = le.fit_transform(X[:, col_idx].astype(str))
    label_encoders[col_idx] = le
    print(f"Column {col_idx}: {len(le.classes_)} unique values")

# Convert to float
X_processed = X_processed.astype(float)

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Data Summary: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
print(f"Testing Data Summary: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

print(f"\nClass distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} samples ({c/len(y_train)*100:.1f}%)")
