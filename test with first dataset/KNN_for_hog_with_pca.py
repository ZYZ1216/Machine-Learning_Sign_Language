import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the reduced features and labels
data_path = 'sign_language_digits_reduced.csv'
df = pd.read_csv(data_path)

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and testing sets (e.g., 70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can choose a different number of neighbors
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("KNN Classifier Performance:")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Evaluate using cross-validation (e.g., 5-fold cross-validation)
cv_scores = cross_val_score(knn, X, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")
# Check for duplicates in the dataset
duplicates = df.duplicated()
print(f"Number of duplicate samples: {duplicates.sum()}")