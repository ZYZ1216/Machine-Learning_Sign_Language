import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the LDA-reduced features and labels
train_df = pd.read_csv('train_lda_features.csv')
test_df = pd.read_csv('test_lda_features.csv')

# Separate features and labels
X_train_lda = train_df.drop('label', axis=1)
y_train_lda = train_df['label']

X_test_lda = test_df.drop('label', axis=1)
y_test_lda = test_df['label']

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_lda, y_train_lda)

# Predict on the test set
y_pred_knn = knn.predict(X_test_lda)

# Evaluate performance
accuracy_knn = accuracy_score(y_test_lda, y_pred_knn)
report_knn = classification_report(y_test_lda, y_pred_knn)

print("KNN Classifier Performance:")
print(f"Accuracy: {accuracy_knn}")
print("Classification Report:")
print(report_knn)