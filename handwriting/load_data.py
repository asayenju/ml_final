from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.random_forest import RandomForestClassifierScratch

digits = datasets.load_digits(return_X_y=True)
digits_dataset_X = digits[0]
digits_dataset_y = digits[1]


X_train, X_test, y_train, y_test = train_test_split(
    digits_dataset_X, digits_dataset_y, test_size=0.2, random_state=42
)

print(f"Training Data Summary: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
print(f"Testing Data Summary: X_test shape {X_test.shape}, y_test shape {y_test.shape}")



num_features = X_train.shape[1]
numeric_cols = set(range(num_features))

print("\nInitializing and training Custom Random Forest...")
model = RandomForestClassifierScratch(
    n_trees=15,
    max_depth=15,
    numeric_cols=numeric_cols,
    random_state=42
)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"Custom Random Forest Accuracy on Test Set: {accuracy * 100:.2f}%\n")


N_test = len(X_test)
digit_to_show = np.random.choice(range(N_test), 1)[0]
sample_x = X_test[digit_to_show]
true_class = y_test[digit_to_show]


predicted_class = model.predict(np.array([sample_x]))[0]



print("Attributes shape:", sample_x.shape)
print("True Class:", true_class)
print("Predicted Class:", predicted_class)

plt.imshow(np.reshape(sample_x, (8,8)), cmap='gray')
plt.title(f"True: {true_class} | Pred: {predicted_class}")
plt.show()
