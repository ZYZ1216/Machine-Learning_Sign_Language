import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split


# Load the dataset
data_path = 'sign_language_digits_hog.csv'
df = pd.read_csv(data_path)

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply LDA to reduce to a lower number of components
lda = LDA(n_components=9)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Combine the reduced features with their labels into dataframe
train_lda_df = pd.DataFrame(X_train_lda, columns=[f'lda_component_{i+1}' for i in range(X_train_lda.shape[1])])
train_lda_df['label'] = y_train.reset_index(drop=True)

test_lda_df = pd.DataFrame(X_test_lda, columns=[f'lda_component_{i+1}' for i in range(X_test_lda.shape[1])])
test_lda_df['label'] = y_test.reset_index(drop=True)

# Save the reduced features by LDA with labels to CSV files
train_lda_df.to_csv('train_lda_features.csv', index=False)
test_lda_df.to_csv('test_lda_features.csv', index=False)
print("LDA reduced features saved successfully.")

