import pandas as pd
import numpy as np

# Load dataset
try:
    # use trip history training dataset from kaggle
    df = pd.read_csv('/content/trip_history.csv')
except FileNotFoundError:
    print("Error: File not found.")
    exit()

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Convert date/time columns
df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
df['End date'] = pd.to_datetime(df['End date'], errors='coerce')

# Drop rows with invalid dates or missing essential data
df.dropna(subset=['Start date', 'End date', 'Start station number', 'End station number', 'Member type'], inplace=True)

# Calculate trip duration in minutes
df['trip_duration_min'] = (df['End date'] - df['Start date']).dt.total_seconds() / 60

# Remove entries with zero or negative duration
df = df[df['trip_duration_min'] > 0]

# Create time-based features
df['day_of_week'] = df['Start date'].dt.dayofweek       # Monday = 0
df['hour_of_day'] = df['Start date'].dt.hour
df['month'] = df['Start date'].dt.month

# Binary encode 'Member type' if it's numeric (as in your sample)
# If it's text ('Member', 'Casual'), use: df['is_member'] = df['Member type'].apply(lambda x: 1 if x.strip().lower() == 'member' else 0)
df['is_member'] = df['Member type']

# Categorize trip durations into bins (e.g., short, medium, long)
df['duration_bin'] = pd.cut(df['trip_duration_min'],
                            bins=[0, 5, 15, 30, 60, np.inf],
                            labels=['very short', 'short', 'medium', 'long', 'very long'])

# Final cleaned & processed DataFrame preview
print("\nProcessed DataFrame:")
print(df.head())

# Optional: Save the cleaned dataset
# df.to_csv('cleaned_bike_data.csv', index=False)
