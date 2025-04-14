import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset (replace 'your_dataset.csv' with your actual file path)

df = pd.read_csv('/content/naive_bayes.csv')


# Preprocessing (handle missing values if any)
df.dropna(inplace=True)

# Feature Engineering (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Message'])
y = df['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted') # Use 'weighted' for multiclass
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean Cross-validation F1 score: {cv_scores.mean()}")

# Evaluation
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
