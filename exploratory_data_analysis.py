import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Load the dataset
file_path = '/content/exploratory_data_analysis.csv'  # Replace with your dataset file path
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Visualize the distribution of numerical features
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

    
# Correlation heatmap for numerical features ONLY
plt.figure(figsize=(10, 6))

# Select only numerical features for correlation calculation
numerical_df = df.select_dtypes(include=['number']) 

correlation_matrix = numerical_df.corr()  # Calculate correlation for numerical features
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Count plots for categorical features
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=column)
    plt.title(f'Count Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Pairplot for numerical features
sns.pairplot(df[numerical_columns])
plt.show()