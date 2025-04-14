import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('/content/bicycle.csv')


# Assuming 'CyclistCount' is the target variable
# Identify non-numeric columns for encoding
non_numeric_cols = data.select_dtypes(exclude=['number']).columns

# Create a LabelEncoder object
le = LabelEncoder()

# Encode non-numeric columns
for col in non_numeric_cols:
    data[col] = le.fit_transform(data[col])

# Define features (X) and target variable (y)
X = data.drop('CyclistCount', axis=1)
y = data['CyclistCount']

# Handle missing values (optional, but recommended)
X.fillna(X.mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")