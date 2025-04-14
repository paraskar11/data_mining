# prompt: write a code  Predict User Class (Subscriber vs Customer)
# Build a classification model (like Logistic Regression, Random Forest, or XGBoost)
# to predict user class using engineered features. Evaluate using accuracy, precision,
# recall, and F1 score.
# dataset contails two column 
# category and message

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset (replace 'your_dataset.csv' with your actual file path)
try:
  df = pd.read_csv('/content/classification_for_subscriber_customer.csv')
except FileNotFoundError:
  print("Error: 'your_dataset.csv' not found. Please provide the correct file path.")
  exit()

# Preprocessing (handle missing values if any)
df.dropna(inplace=True)

# Feature Engineering (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(f"Predicted class: {y_pred}")

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted') # Use 'weighted' for multiclass
recall = recall_score(y_test, y_pred, average='weighted') # Use 'weighted' for multiclass
f1 = f1_score(y_test, y_pred, average='weighted') # Use 'weighted' for multiclass

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
